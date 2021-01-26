import sys, os
from argparse import ArgumentParser

from allennlp.commands import main as allen_main


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('dataset')
    parser.add_argument('--model-name', default=None)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    dataset_name = args.dataset
    model_name = args.model_name or dataset_name

    os.environ['dataset'] = dataset_name

    sys.argv = [
        'allennlp', 
        'train', 'config/custom.jsonnet',
        '-s', f'models/trained/{model_name}',
        '--include-package', 'src.allen_elements'
    ]

    allen_main()
