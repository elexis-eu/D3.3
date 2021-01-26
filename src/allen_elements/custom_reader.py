from typing import Iterable

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


@DatasetReader.register('custom')
class CustomDatasetReader(DatasetReader):
    def __init__(self, model_name: str,
                 namespace: str = "tokens"):
        super().__init__()
        self.namespace = namespace
        self.indexer = PretrainedTransformerIndexer(model_name, namespace)
        self.tokenizer = PretrainedTransformerTokenizer(model_name, max_length=500, 
                                                        tokenizer_kwargs=dict(use_fast=True, truncation=True))

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f:
                parts = line.rstrip().split('\t')
                if len(parts) != 2:
                    print(line)
                label, _input = parts
                yield self.text_to_instance(dict(label=label, input=_input))

    def text_to_instance(self, data):
        model_input = self.tokenizer.tokenize(data['input'])
        content = TextField(model_input, {self.namespace: self.indexer})
        label = LabelField(data['label'])
        return Instance(dict(
            tokens=content,
            label=label
        ))
