local pretrained_model = 'bert-base-multilingual-cased';

local dataset = std.extVar('dataset');

local model = {
  type: 'custom',
  text_field_embedder: {
    type: 'basic',
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: pretrained_model,
        train_parameters: false,
      }
    }
  },
  seq2vec_encoder: {
    type: 'bert_pooler',
    pretrained_model: pretrained_model,
  },
};

{
  dataset_reader: {
    type: 'custom',
    model_name: pretrained_model,
  },
  model: model,
  train_data_path: std.format('data/%s/train.tsv', dataset),
  validation_data_path: std.format('data/%s/dev.tsv', dataset),
  test_data_path: std.format('data/%s/test.tsv', dataset),
  evaluate_on_test: true,
  data_loader: {
    batch_size: 32,
    shuffle: true,
  },
  trainer: {
    cuda_device: 0,
    grad_clipping: 1,
    num_epochs: 100,
    num_gradient_accumulation_steps: 8,
    optimizer: {
        type: "adam",
        eps: 1e-08,
        lr: 5e-05,
        parameter_groups: [
            [
                [
                    "bias",
                    "LayerNorm.weight"
                ],
                {
                    "weight_decay": 0
                }
            ]
        ]
    },
    patience: 3,
    validation_metric: "-loss"
  },
}
