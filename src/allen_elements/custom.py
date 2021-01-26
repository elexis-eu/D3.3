from typing import Any, Dict

import torch
from allennlp.common import JsonDict
from allennlp.data import TextFieldTensors
from allennlp.models import Model, BasicClassifier
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy


@Predictor.register('custom')
class CustomPredictor(TextClassifierPredictor):
    def dump_line(self, outputs: JsonDict) -> str:
        outputs.pop('token_ids')
        outputs.pop('tokens')
        outputs.pop('logits')
        return super().dump_line(outputs)
    
    def _json_to_instance(self, data):
        return self._dataset_reader.text_to_instance(data)


@Model.register('custom')
class CustomClassifier(BasicClassifier):
    _f1 = FBetaMeasure(average='macro')
    _wf1 = FBetaMeasure(average='weighted')
    _acc3 = CategoricalAccuracy(top_k=3)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(tokens, label)
        
        if label is not None:
            self._f1(out['logits'], label)
            self._wf1(out['logits'], label)
            self._acc3(out['logits'], label)
        
        return out

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics.update(**self._f1.get_metric(reset))
        metrics['wf1'] = self._wf1.get_metric(reset)['fscore']
        metrics['acc3'] = self._acc3.get_metric(reset)
        return metrics

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    default_predictor = 'custom'
