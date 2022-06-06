# Pre-trained Models

# <img src="https://img.icons8.com/color/48/undefined/1-circle--v1.png"/> BiomedNLP-PubMedBERT

`git clone https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

```yaml
config.json
flax_model.msgpack
pytorch_model.bin
tokenizer_config.json
vocab.txt
```

config.json

```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "model_type": "bert",
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

# <img src="https://img.icons8.com/color/48/undefined/2-circle--v1.png"/> Biobert

`git clone https://huggingface.co/dmis-lab/biobert-v1.1`

```yaml
config.json
flax_model.msgpack
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

config.json

```json
{
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 28996
}
```
