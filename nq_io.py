'''
IO helper functions for Google's Natural Questions (NQ) dataset for
creating train-dev splits and other subsets of the training data.

Usage:
    nq_io.py train-dev FILE [--lnum=<int> --dev-frac=<float> --output-dir=DIR --seed=<int>]
    nq_io.py create-subset FILE OUTFILE [--lnum=<int> --frac=<float> --seed=<int>]

Examples:
    python nq_io.py create-subset simplified-nq-train.jsonl tiny_data.jsonl --frac=0.00001
    python nq_io.py train-dev simplified-nq-train.jsonl

Options:
    -h --help                               show this screen.
    --lnum=<int>                            number of lines in the jsonl file [default: 307373]
    --frac=<float>                          fraction of lines in the jsonl to randomly sample and use [default: 0.01]
    --dev-frac=<float>                      fraction to use for train-dev split [default: 0.2]
    --output-dir=DIR                        output directory for generated files [default: ./]
    --seed=<int>                            random seed for sampling of lines to keep [default: 5293]
'''

from docopt import docopt
from pathlib import Path
import random
from tqdm import tqdm

from typing import List

def create_train_dev_split(reader, dir, dev_lines: List[int], max_lnum: int):
    '''Creates train-dev split in dir with lines with numbers in dev_lines (indexed from 1, sorted in descending order) in the dev set'''
    outfile_train_path = Path(dir / 'nq_train.jsonl')
    outfile_dev_path = Path(dir / 'nq_dev.jsonl')
    with open(outfile_train_path, 'w') as outfile_train, open(outfile_dev_path, 'w') as outfile_dev:
        line = reader.readline()
        current_lnum = 1
        print('Creating train-dev split ...')
        with tqdm(total=max_lnum) as pbar:
            while line:
                if len(dev_lines) == 0 or current_lnum != dev_lines[-1]:
                    outfile_dev.write(line)
                else:
                    outfile_train.write(line)
                    dev_lines.pop()
                
                line = reader.readline()
                current_lnum += 1
                pbar.update(1)

def create_subset(reader, outfile, subset_lines: List[int], max_lnum: int):
    '''Creates subset of file read by reader with lines with numbers in subset_lines, writes subset to outfile'''
    outfile_path = Path(outfile)
    with open(outfile_path, 'w') as outfile:
        line = reader.readline()
        current_lnum = 1
        print('Creating subset ...')
        with tqdm(total=max_lnum) as pbar:
            while line:
                if len(subset_lines) > 0 and current_lnum == subset_lines[-1]:
                    outfile.write(line)
                    subset_lines.pop()
                line = reader.readline()
                current_lnum += 1
                pbar.update(1)

def main():
    args = docopt(__doc__)

    file_name = args['FILE']
    file_path = Path(file_name)

    if args['train-dev']:
        output_dir_name = args['--output-dir']
        output_dir = Path(output_dir_name)
        seed = int(args['--seed'])
        max_lnum = int(args['--lnum'])
        dev_frac = float(args['--dev-frac'])

        random.seed(seed)
        dev_lines = random.sample(range(1, max_lnum+1), int((1-dev_frac) * max_lnum))
        dev_lines.sort(reverse=True)

        output_dir.mkdir(exist_ok=True)
        
        with open(file_path, 'r') as reader:
            create_train_dev_split(reader, output_dir, dev_lines, max_lnum)
        print('Created train-dev split [nq_train.jsonl, nq_dev.jsonl] in {}'.format(str(output_dir)))
    elif args['create-subset']:
        outfile_name = args['OUTFILE']
        output_path = Path(outfile_name)
        seed = int(args['--seed'])
        max_lnum = int(args['--lnum'])
        subset_frac = float(args['--frac'])

        random.seed(seed)
        subset_lines = random.sample(range(1, max_lnum+1), int(subset_frac * max_lnum))
        subset_lines.sort(reverse=True)

        with open(file_path, 'r') as reader:
            create_subset(reader, output_path, subset_lines, max_lnum)
        print(f"Created subset of {outfile_name} : {outfile_name}")
    else:
        raise RuntimeError('invalid mode')

if __name__ == '__main__':
    main()
