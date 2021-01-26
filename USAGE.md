
# Introduction

## NOTE
Make sure to have followed the installation instructions contained in README before proceeding.

## DISCLAIMER

This project is based off of the AllenNLP library (version 1.3.0).

- [AllenNLP GitHub](https://github.com/allenai/allennlp/)
- [AllenNLP Website](https://allennlp.org/)
- [AllenNLP Docs](https://docs.allennlp.org/v1.3.0/)


# Training a model

In order to train a model, place `train.tsv`, `dev.tsv`, `test.tsv` files under a subfolder of `data/` (e.g., `data/my-dataset/train.tsv` etc).
The file structure is pretty simple: every line must contain the label and the lemma/gloss pair to be fed to the model, separated by the tab character. For example:
```
POLITICS_GOVERNMENT_AND_NOBILITY [TAB] royal family | Royal persons collectively
```
Note that the usage of lemmas is not mandatory, since the model works even with glosses only. In the example above, lemma and gloss are separated by means of a pipe character, otherwise, the example would look like the following:
```
POLITICS_GOVERNMENT_AND_NOBILITY [TAB] Royal persons collectively
```
Once the training data is ready, you can start training by executing:
```
python src/main.py <data-folder-name>
```
Where, in case your dataset files were placed under `data/my-dataset`, `<data-folder-name>` is my-dataset.

## Running a demo of the trained model

Once the model has finished training, all the training files will be saved under `models/trained/<data-folder-name>`. In case you wish to try an interactive version of the trained model, you can launch it via `python src/serve.py trained/<data-folder-name>`.

# Using a released / trained model

Download the WordNet-based model at http://TODO (link will be available soon) and place it under the `models/released/` folder.

To run a simple interactive command-line demo, run the following command:
```
python src/serve.py released/wn
```
To tag a file in the tsv format described above (in case you only have raw sentences, simply prepend `NODOMAIN\t` to every line of the file), run the following command:
```
allennlp predict models/released/wn.tar.gz <path/to/file.tsv> --output-file <path/to/output.jsonl> --batch-size <batch-size> --cuda-device 0 --use-dataset-reader --include-package src.allen_elements --silent
```
For further information about the command, check out the [AllenNLP Documentation page](https://docs.allennlp.org/v1.3.0/api/commands/predict/) or type `allennlp predict --help`.
