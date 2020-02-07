'''
Convert Google's pre-computed features to hdf5

You don't need to use this script, but precomputing for yourself would take > 1 day with a single CPU.

Usage:
    convert_tfrecord_hdf5.py TFRECORDS_FILE OUTPUT_FILE

Example:
    python convert_tfrecord_hdf5.py nq-train.tfrecords-00000-of-00001 google_features.hdf5

Options:
    -h --help                               show this screen.
    TFRECORDS_FILE                          file containing pre-computed features for BERT
    OUTPUT_FILE                             where to save the input features in hdf5 format
'''

from docopt import docopt
import h5py
import logging
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # This supresses a warning caused by old TF version
import tensorflow as tf
import numpy as np
from tqdm import tqdm

tf.compat.v1.enable_eager_execution()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_tfrecord(input_file, output_file):
    tf_file = tf.data.TFRecordDataset(input_file)

    # As precomputed by Google
    seq_length = 512

    # Count number of records
    n_records = 0
    for record in tf_file:
        n_records += 1

    name_to_features = {
        "answer_types": tf.FixedLenFeature([], tf.int64),
        "end_positions": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "start_positions": tf.FixedLenFeature([], tf.int64),
        "unique_ids": tf.FixedLenFeature([], tf.int64),      
        }

    with h5py.File(output_file, "w") as outfile:
        unique_ids = outfile.create_dataset("unique_ids", (n_records,), dtype='int64') # Necessary to prevent overflow
        input_ids = outfile.create_dataset("input_ids", (n_records, seq_length,), dtype='i')
        input_mask = outfile.create_dataset("input_mask", (n_records, seq_length,), dtype='i')
        segment_ids = outfile.create_dataset("segment_ids", (n_records, seq_length,), dtype='i')
        start_positions = outfile.create_dataset("start_positions", (n_records,), dtype='i')
        end_positions = outfile.create_dataset("end_positions", (n_records,), dtype='i')
        answer_types = outfile.create_dataset("answer_types", (n_records,), dtype='i')

        with tqdm(total=n_records) as pbar:
            logger.info(f'Creating {output_file} ...')
            for i, example in enumerate(tf_file):
                parsed_example = tf.parse_single_example(example, name_to_features)
                unique_ids[i] = np.array(parsed_example['unique_ids'])
                input_ids[i,:] = np.array(parsed_example['input_ids'])
                input_mask[i,:] = np.array(parsed_example['input_mask'])
                segment_ids[i,:] = np.array(parsed_example['segment_ids'])
                start_positions[i] = np.array(parsed_example['start_positions'])
                end_positions[i] = np.array(parsed_example['end_positions'])
                answer_types[i] = np.array(parsed_example['answer_types'])
                pbar.update(1)

    logger.info(f'Created {output_file}')

if __name__ == '__main__':
    args = docopt(__doc__)
    convert_tfrecord(args['TFRECORDS_FILE'], args['OUTPUT_FILE'])
