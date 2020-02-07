'''
Convert Google's pre-trained checkpoint to PyTorch

Usage:
    convert_tf.py --tf_checkpoint_path=<str> --bert_config_file=<str> --save_dir_bert_part=<str> --save_dir=<str>

Example:
    python convert_tf.py --tf_checkpoint_path=./bert-joint-baseline/bert_joint.ckpt --bert_config_file=./bert-joint-baseline/bert_config.json --save_dir_bert_part=./bert_part --save_dir=./pytorch_model

Options:
    -h --help                               show this screen.
    --tf_checkpoint_path=<str>              path to TensorFlow checkpoint to convert, e.g. bert_joint.ckpt
                                            (Specifying like this will also work when checkpoint is given as multiple files bert_joint.ckpt.index and ckpt.data files)
    --bert_config_file=<str>                path to BERT config file, e.g. bert_config.json
    --save_dir_bert_part=<str>              directory where to save the model (only the Bert part)
    --save_dir=<str>                        directory where to save the whole model in Pytorch format (config.json and pytorch_model.bin)
'''

from collections import OrderedDict
from docopt import docopt
import logging
import os
from pathlib import Path

import torch
from transformers import BertConfig, BertForPreTraining

from nq_model import NQBert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    # Function adapted from Huggingface's transformers
    # We keep track of additional layers in extra_list
    extra_list = []
    try:
        import re
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore',category=FutureWarning) # This supresses a warning caused by old TF version
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if m_name in ['answer_type_output_bias', 'answer_type_output_weights', 'output_bias', 'output_weights']:
                extra_list.append((m_name, array))
                continue
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        if m_name not in ['answer_type_output_bias', 'answer_type_output_weights', 'output_bias', 'output_weights']:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        else:
            logger.info(f"Added {m_name} to extra_list")
    return extra_list

def main(tf_checkpoint_path, bert_config_file, save_dir_bert_part, save_dir):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    logger.info("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    extra_list = load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save the Bert part
    Path(save_dir_bert_part).mkdir(exist_ok=True)
    model.save_pretrained(save_dir_bert_part)

    # Reload
    nq_model = NQBert.from_pretrained(save_dir_bert_part)

    # Check if the Bert parameters match
    tf_model_parameters = list(model.parameters())
    all( ( torch.equal(p, tf_model_parameters[i])
            for i, p in enumerate(nq_model.bert.parameters()) ) )

    # Copy over parameters of final layers
    tf_model_final_layer_params = { x[0]:x[1] for x in extra_list}
    nq_model_final_layer_params = OrderedDict({
        'span_outputs.weight':torch.tensor(tf_model_final_layer_params['output_weights']),
        'span_outputs.bias':torch.tensor(tf_model_final_layer_params['output_bias']),
        'type_output.weight':torch.tensor(tf_model_final_layer_params['answer_type_output_weights']),
        'type_output.bias':torch.tensor(tf_model_final_layer_params['answer_type_output_bias']),
        })

    nq_model.load_state_dict(nq_model_final_layer_params, strict=False)

    # Check if final parameters match
    all( [ torch.equal(x[0], x[1])
            for x in zip(list(nq_model.parameters())[-4:], list(nq_model_final_layer_params.values())) ] )

    # Save whole model
    Path(save_dir).mkdir(exist_ok=True)
    nq_model.save_pretrained(save_dir)

if __name__ == '__main__':
    args = docopt(__doc__)
    tf_checkpoint_path = args['--tf_checkpoint_path']
    bert_config_file = args['--bert_config_file']
    save_dir_bert_part = args['--save_dir_bert_part']
    save_dir = args['--save_dir']

    main(tf_checkpoint_path, bert_config_file, save_dir_bert_part, save_dir)
