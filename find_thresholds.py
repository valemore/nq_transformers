'''
Finding optimal thresholds for short and long answers

At inference time, the model gives a confidence score for short and long answer.
This script evaluates the model on a dev set and finds the optimal cutoff-points for
the confidence scores.

Usage:
    find_thresholds.py MODEL_DIR VOCAB_FILE DEV_FILE DEV_FEATURES OUTPUT_DIR

Example:
    python find_thresholds.py checkpoint_dir vocab-nq.txt nq_dev.jsonl nq_dev.hdf5 ./find_thresholds/

Options:
    -h --help                               show this screen.
    MODEL_DIR                               directory contained trained model weights (pytorch_model.bin)
    VOCAB_FILE                              vocabulary file to use for the tokenizer
    DEV_FILE                                example dev set (jsonl file)
    DEV_FEATURES                            where to save/load the corresponding features
    OUTPUT_DIR                              where to save the predictions and optimal thresholds as json
'''
from docopt import docopt
import numpy as np
import json
import logging
from pathlib import Path

import torch

from nq_model import NQBert
import nq_metric
import nq_config
import google_tokenization
from run_nq import evaluate

logger = logging.getLogger(__name__)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class ArgsDummy():
    # This is a dummy class used for passing arguments to evaluate in run_nq.py
    def __init__(self,
                    model_type, model_name_or_path, output_dir, vocab_file,
                    **kwargs):
        # Store true
        self.only_final_layers=False
        self.do_predict=False
        self.do_train=False
        self.do_eval=False
        self.evaluate_during_training=False
        self.do_lower_case=False
        self.verbose_logging=False
        self.eval_all_checkpoints=False
        self.no_cuda=False
        self.overwrite_output_dir=False
        self.overwrite_cached_input_features=False
        self.fp16=False
        
        # With defaults
        self.train_file=None
        self.dev_file=None
        self.test_file=None
        self.train_features=None
        self.dev_features=None
        self.test_features=None
        self.undersampling_factor=50
        self.from_checkpoint=''
        self.long_answer_threshold=3.0
        self.short_answer_threshold=3.0
        self.train_subset_file=None
        self.train_subset_features=None
        self.config_name=""
        self.tokenizer_name=""
        self.cache_dir=""
        self.null_score_diff_threshold=0.0,
        self.max_seq_length=384
        self.doc_stride=128
        self.max_query_length=64
        self.max_windows=48
        self.per_gpu_train_batch_size=8
        self.per_gpu_eval_batch_size=8
        self.learning_rate=5e-5
        self.gradient_accumulation_steps=1
        self.weight_decay=0.0
        self.adam_epsilon=1e-8
        self.max_grad_norm=1.0
        self.num_train_epochs=3.0
        self.max_steps=-1
        self.warmup_steps=0
        self.n_best_size=20
        self.max_answer_length=30
        self.logging_steps=50
        self.save_steps=50
        self.seed=42
        self.local_rank=-1
        self.fp16_opt_level='O1'
        self.server_ip=''
        self.server_port=''
        
        #Required:
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.vocab_file = vocab_file

        #Needed in eval script
        self.n_gpu = 1
        
        self.__dict__.update(kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_on_dev(checkpoint_dir, vocab_file, dev_file, dev_features, output_dir):
    args = ArgsDummy(model_type='bert',
                    model_name_or_path='nq_model',
                    output_dir=output_dir,
                    from_checkpoint=checkpoint_dir,
                    vocab_file=vocab_file,
                    dev_file=dev_file,
                    dev_features=dev_features,
                    do_lower_case=True,
                    per_gpu_eval_batch_size=32,
                    max_seq_length=512,
                    doc_stride=128,
                    max_windows=48,
                    device=device)

    # Set NQA prep variables
    nq_config.max_context_length = args.max_seq_length
    nq_config.max_question_length = args.max_query_length
    nq_config.max_windows = args.max_windows
    nq_config.window_stride = args.doc_stride
    nq_config.include_unknowns = 1.0 / args.undersampling_factor
    nq_config.n_best_size = args.n_best_size
    nq_config.max_answer_length = args.max_answer_length

    tokenizer = google_tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)

    # Load model
    model = NQBert.from_pretrained(args.from_checkpoint)
    model.to(args.device)

    raw_results = evaluate(args, model, tokenizer, dataset_type='dev', output_raw=True)
    return raw_results

def raw_preds_to_f1(raw_results, dev_file, la_threshold, sa_threshold):
    nq_pred_df = nq_metric.get_test_df(raw_results, la_threshold, sa_threshold)
    nq_pred_df = nq_pred_df.sort_values(by='example_id').set_index('example_id', drop=False)

    labelled_df = nq_metric.get_labelled_df(dev_file).sort_values(by='example_id').set_index('example_id', drop=False)
    f1_scores = nq_metric.get_f1(nq_pred_df, labelled_df)
    results = {'micro_f1':f1_scores[0] * 100,
                'long_f1':f1_scores[1] * 100,
                'short_f1':f1_scores[2] * 100}
    return results

def best_thresh(raw_results, dev_file):
    min_sa_score = min([ r['short_answer_score'] for r in raw_results.values() ])
    max_sa_score = max([ r['short_answer_score'] for r in raw_results.values() ])
    
    min_la_score = min([ r['long_answer_score'] for r in raw_results.values() ])
    max_la_score= max([ r['long_answer_score'] for r in raw_results.values() ])
    
    min_thresh, max_thresh = min(min_sa_score, min_la_score), max(max_sa_score, max_la_score)
    
    thresholds = np.linspace(start=min_thresh, stop=max_thresh, num=50)
    scores = []
    for i, thresh in enumerate(thresholds):
        scores.append(raw_preds_to_f1(raw_results, dev_file, thresh, thresh))
        
    short_f1 =np.array([ s['short_f1'] for s in scores ])
    best_short_f1 = ( short_f1.argmax(), short_f1.max() )

    long_f1 =np.array([ s['long_f1'] for s in scores ])
    best_long_f1 = ( long_f1.argmax(), long_f1.max() )

    micro_f1 =np.array([ s['micro_f1'] for s in scores ])
    best_micro_f1 = ( micro_f1.argmax(), micro_f1.max() )
    
    default_f1 = {
        'threshold':thresholds[0],
        'short_f1':short_f1[0],
        'long_f1':long_f1[0],
        'micro_f1':micro_f1[0]
    }
    
    res = {
        'short':{
            'threshold':thresholds[best_short_f1[0]],
            'f1':best_short_f1[1]
        },
        'long':{
            'threshold':thresholds[best_long_f1[0]],
            'f1':best_long_f1[1]
        },
        'micro':{
            'threshold':thresholds[best_micro_f1[0]],
            'f1':best_micro_f1[1]
        },
        'default_f1':default_f1
    }
    return res

def main(checkpoint_dir, vocab_file, dev_file, dev_features, output_dir):
    logger.info('Finding optimal thresholds ...')
    raw_results = eval_on_dev(checkpoint_dir, vocab_file, dev_file, dev_features, output_dir)
    res = best_thresh(raw_results, dev_file)
    logger.info(f'Optimal thresholds:\n{res}')
    outfile = Path(output_dir) / 'optimal_thresholds.json'
    logger.info(f'Writing to {str(outfile)}...')
    with open(outfile, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    args = docopt(__doc__)
    checkpoint_dir = args['MODEL_DIR']
    vocab_file = args['VOCAB_FILE']
    dev_file = args['DEV_FILE']
    dev_features = args['DEV_FEATURES']
    output_dir = args['OUTPUT_DIR']
    main(checkpoint_dir, vocab_file, dev_file, dev_features, output_dir)
