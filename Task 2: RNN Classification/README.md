# RNN Classification with PyTorch

> [feat(hw2): recurrent neural network #2](https://github.com/colorsquare/ml-for-nlp/pull/2)

```python
class RNN(nn.Module):
    
    def __init__(self, num_embeddings, padding_idx, embedding_dim,
        hidden_dim,  num_layers, dropout, bidirectional):
        # TODO: Build a RNN model
    
    def forward(self, text, text_lengths):
        # TODO: return `output`

def train(model, iterator, optimizer, criterion):
    # TODO: return `(epoch_loss, epoch_acc)`

def evaluate(model, iterator, criterion):
    # TODO: return `(epoch_loss, epoch_acc)`

def set_hyperparameter_dict():
    # TODO: return `param_dict`
```

Instructions are available at [hw2](https://github.com/uilab-kaist/cs475-mlnlp-fall-2022-hw/tree/main/hw2).

## Problem

**Goal:** Validation accuracy *83.40%*.

- Q1. Design RNN model with different neural networks(GRU, LSTM).
- Q2. How would you tune the hyperparameters?
