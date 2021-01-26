import argparse
import logging

from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


logger = logging.getLogger(__name__)


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)

    model_name = '/model' if args.model_name.startswith('trained') else ''

    archive = load_archive(
        f"models/{args.model_name}{model_name}.tar.gz",
        cuda_device=args.cuda_device,
        overrides='{"model": {"type": "allen_elements.custom.CustomClassifier"}, "dataset_reader": {"type": "allen_elements.custom_reader.CustomDatasetReader"} }',
    )

    return Predictor.from_archive(archive, predictor_name="allen_elements.custom.CustomPredictor")


def parse_args():
    description = """Serve up a simple model."""
    parser = argparse.ArgumentParser('serve', description=description)

    parser.add_argument('model_name', type=str, help='name of the model to load')

    parser.add_argument('--top-n', type=int, default=3, dest='top_n')

    parser.add_argument('--cuda-device', type=int, default=-1, help="id of GPU to use (if any)")

    return parser.parse_args()


def serve(args: argparse.Namespace) -> None:

    print('loading model')
    predictor = _get_predictor(args)
    vocab = predictor._model.vocab.get_index_to_token_vocabulary('labels')
    # print(vocab)

    print("Type EXIT to stop")
    model_input = None

    while model_input != 'EXIT':
        model_input = input("Type text to classify: ")
        fake_input = dict(input=model_input, label='NODOMAIN', synset='null')
        out = predictor.predict_json(fake_input)
        probs = out['probs']
        print(f"model predicted {out['label']} with confidence {max(probs):.4f}")
        top = sorted(range(1, len(probs)), key=lambda i: probs[i], reverse=True)
        if args.top_n and args.top_n > 1:
            for i, idx in enumerate(top[1:args.top_n], 2):
                print(f"\tprediction {i}: {vocab[idx]} with confidence {probs[idx]:.4f}")


if __name__ == '__main__':
    args = parse_args()
    serve(args)
