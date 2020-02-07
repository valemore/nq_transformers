'''
Finetuning Huggingface's models for question-answering on Natural Questions (NQ) datasety by Google

For the full list of options, type python run_nq.py -h 
'''

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import glob
import sys
import timeit

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # Following import causes annoying warning
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup

from nq_model import NQBert
import nq_metric

from nq_features import load_or_precompute_nq_features
from nq_eval import RawResult, read_candidates, compute_pred_dict
import google_tokenization

import nq_config

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append((param_group, param_group['lr']))
    return lrs

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # Allow for different learning rate for final layers
    final_layers = ['span_outputs.weight',
                    'span_outputs.bias',
                    'type_output.weight',
                    'type_output.bias']
    if args.final_layers_lr == -1.0:
        args.final_layers_lr = args.learning_rate
    if args.final_layers_wd == -1.0:
        args.final_layers_wd = args.weight_decay

    final_layer_params = [(n, p) for n, p in model.named_parameters() if n in final_layers]
    non_final_layer_params = [(n, p) for n, p in model.named_parameters() if n not in final_layers]

    no_decay = ['bias', 'LayerNorm.weight']
    final_layer_decaying_params = [p for n, p in final_layer_params if not any(nd in n for nd in no_decay)]
    final_layer_nondecaying_params = [p for n, p in final_layer_params if any(nd in n for nd in no_decay)]

    non_final_layer_decaying_params = [p for n, p in non_final_layer_params if not any(nd in n for nd in no_decay)]
    non_final_layer_nondecaying_params = [p for n, p in non_final_layer_params if any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': final_layer_decaying_params,
            'lr':args.final_layers_lr,
            'weight_decay':args.final_layers_wd},
        {'params': final_layer_nondecaying_params,
            'lr':args.final_layers_lr,
            'weight_decay':0.0},
        {'params': non_final_layer_decaying_params,
            'lr':args.learning_rate,
            'weight_decay':args.weight_decay},
        {'params': non_final_layer_nondecaying_params,
            'lr':args.learning_rate,
            'weight_decay':0.0},
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Allow choice between lr schedules
    if args.constant_lr and args.warmup_steps == 0:
        scheduler = get_constant_schedule(optimizer)
    elif args.constant_lr and args.warmup_steps > 0:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids':       batch['input_ids'].to(args.device),
                      'attention_mask':  batch['attention_mask'].to(args.device),
                      'token_type_ids':  batch['token_type_ids'].to(args.device),
                      'start_positions': batch['start_positions'].to(args.device),
                      'end_positions':   batch['end_positions'].to(args.device),
                      'instance_types':  batch['instance_types'].to(args.device)}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, dataset_type='dev', prefix=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('lr_final_layers', scheduler.get_lr()[1], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, dataset_type='test', prefix='', output_preds=False, output_raw=False):
    '''Predict answer spans using model and tokenizer (Google tokenizer)'''
    # I believe the enclosing barriers can be deleted. dataset_type should be test or train
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = load_or_precompute_nq_features(args, tokenizer, dataset_type)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    nq_config.predict_batch_size = args.eval_batch_size

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    all_results = []

    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device)
                      }
            
            outputs = model(**inputs)

        for b, unique_id in enumerate(batch['unique_id']):
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][b]),
                    end_logits=to_list(outputs[1][b]),
                    answer_type_logits=to_list(outputs[2][b])))

    # Computing predictions
    dataset_jsonl_file = args.test_file if dataset_type == 'test' else args.dev_file if dataset_type == 'dev' else args.train_file
    logger.info(f"Loading candidates from {dataset_jsonl_file} ...")
    candidates_dict = read_candidates(dataset_jsonl_file)
    logger.info(f"Computing predictions for {dataset_jsonl_file} ...")
    nq_pred_dict = compute_pred_dict(candidates_dict,
                                     dataset,
                                     [r._asdict() for r in all_results])

    evalTime = timeit.default_timer() - start_time
    logger.info("Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    if output_raw:
        return nq_pred_dict

    output_prediction_file = os.path.join(args.output_dir, f"predictions_{dataset_type}_{prefix}.json")
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    with open(output_prediction_file, 'w') as f:
        json.dump(list(nq_pred_dict.values()), f, indent=4)

    nq_pred_df = nq_metric.get_test_df(nq_pred_dict, args.long_answer_threshold, args.short_answer_threshold)
    nq_pred_df = nq_pred_df.sort_values(by='example_id').set_index('example_id', drop=False)

    if not output_preds:
        labelled_df = nq_metric.get_labelled_df(dataset_jsonl_file).sort_values(by='example_id').set_index('example_id', drop=False)
        f1_scores = nq_metric.get_f1(nq_pred_df, labelled_df)
        results = {'micro_f1':f1_scores[0] * 100,
                    'long_f1':f1_scores[1] * 100,
                    'short_f1':f1_scores[2] * 100}
        return results
    else:
        submission_df = nq_pred_df[['example_id', 'long_answer', 'short_answer']]
        submission_df = submission_df.rename(columns={'long_answer':'long', 'short_answer':'short'})
        submission_df = submission_df.melt(id_vars=['example_id'], var_name='answer_length')
        submission_df['example_id'] = submission_df['example_id'].apply(str) + '_' + submission_df['answer_length']
        submission_df = submission_df.rename(columns={'value':'PredictionString'}).drop(columns='answer_length')

        return submission_df

def main(args):
    # One must be chosen:
    if not args.do_train and not args.do_eval and not args.do_predict:
        sys.exit("Must use at least one of --do_train, --do_eval, --do_predict")
    if args.do_train and (not args.train_file or not args.train_features):
        sys.exit("When training, must specify both --train_file, --train_features")
    if args.do_eval and (not args.dev_file or not args.dev_features):
        sys.exit("When evaluating on dev set, must specify both --dev_file, --dev_features")
    if args.do_predict and (not args.test_file or not args.test_features):
        sys.exit("When predicting on test set, must specify both --test_file, --test_features")
    if args.evaluate_during_training and (not args.dev_file):
        sys.exit("When evaluating during training, specify both --dev_file, --dev_features")

    # Set NQA prep variables
    nq_config.max_context_length = args.max_seq_length
    nq_config.max_question_length = args.max_query_length
    nq_config.max_windows = args.max_windows
    nq_config.window_stride = args.doc_stride
    nq_config.include_unknowns = 1.0 / args.undersampling_factor
    nq_config.n_best_size = args.n_best_size
    nq_config.max_answer_length = args.max_answer_length

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    # Google tokenizer:
    tokenizer = google_tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)

    # Resume from checkpoint
    if args.from_checkpoint:
        model = NQBert.from_pretrained(args.from_checkpoint)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=2,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
        model = NQBert.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    
    # Only tune last layer: Last four parameters (answer type weight & bias, span start/end weight and bias)
    if args.only_final_layers:
        for i_param, param in enumerate(model.parameters()):
            if i_param in range(len(model.parameters())-4):
                continue
            else:
                param.requires_grad = False
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Using  Google's feature computation
        train_dataset = load_or_precompute_nq_features(args, tokenizer, dataset_type='train')
    
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = NQBert.from_pretrained(args.output_dir)
        tokenizer = google_tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        model.to(args.device)

    # When checkpoint is supplied and output_dir does not contain mode, copy it over from checkpoint
    output_model_path = Path(args.output_dir) / 'pytorch_model.bin'
    if not args.do_train and (not output_model_path.exists()):
        if not args.from_checkpoint:
            sys.exit("When not training, evaluation model bust either resider in output directory or checkpoint mus be provided.")
        eval_model_dir = args.from_checkpoint
    else:
        eval_model_dir = args.output_dir


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [eval_model_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(eval_model_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = NQBert.from_pretrained(checkpoint)
            tokenizer = google_tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, dataset_type='dev', prefix=global_step)
            result = dict((k + '_dev' + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Predicting on test data set...")
        
        # Reload the model
        model = NQBert.from_pretrained(eval_model_dir)   
        model.to(args.device)

        tokenizer = google_tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        submission_df = evaluate(args, model, tokenizer, dataset_type='test', prefix='', output_preds=True)

        submission_df.to_csv("./submission.csv", index=False)

        logger.info("submission_df:")
        logger.info(submission_df)
        
    logger.info("Results: {}".format(results))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=False,
                        help="Natural Questions jsonl for training. E.g., simplified-nq-train.jsonl")
    parser.add_argument("--dev_file", default=None, type=str, required=False,
                        help="Natural Questions jsonl for cross-validation. E.g., simplified-nq-dev.jsonl")
    parser.add_argument("--test_file", default=None, type=str, required=False,
                        help="Natural Questions jsonl for prediction. E.g., simplified-nq-test.jsonl")
    parser.add_argument("--train_features", default=None, type=str, required=False,
                        help="Train features are computed and saved to, or loaded from this file")
    parser.add_argument("--dev_features", default=None, type=str, required=False,
                        help="Dev features are computed and saved to, or loaded from this file")
    parser.add_argument("--test_features", default=None, type=str, required=False,
                        help="Test features are computed and saved to, or loaded from this file")
    parser.add_argument("--undersampling_factor", default=50, type=int, required=False,
                        help="Undersample null instances by this factor")
    parser.add_argument("--only_final_layers", action='store_true',
                        help="Whether to only tune layers on top of BERT representation")
    parser.add_argument("--from_checkpoint", default='', type=str, required=False,
                        help="Start training from the provided checkpoint")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="Vocabulary file for the tokenizer")
    parser.add_argument('--long_answer_threshold', type=float, default=-1*float('inf'),
                        help="Threshold for confidence score for long answer.")
    parser.add_argument('--short_answer_threshold', type=float, default=-1*float('inf'),
                        help="Threshold for confidence score for short answer.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--final_layers_wd", default=-1.0, type=float,
                        help="Weight decay  parameeters for final layers.")

    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to perform prediction on test set")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_windows", default=48, type=int,
                        help="The maximum number of windows / instances to get per question.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--final_layers_lr", default=-1.0, type=float,
                        help="The initial learning rate for the final layers")
                        
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--constant_lr", action='store_true',
                        help="If true, use a constant learning rate (after a potential warmup phase)")
                    
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cached_input_features', action='store_true',
                        help="Overwrite the precomputed input features")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    main(args)
