# Pooling Token-level Representation in BERT

> [Report](https://github.com/colorsquare/ml-for-nlp/blob/main/hw3/docs/report.pdf)

```python
class MyBertPooler(nn.Module):
    def __init__(self, config):
        # TODO

    def forward(self, hidden_states, *args, **kwargs):
        # TODO
```

Instructions are available at [hw3](https://github.com/uilab-kaist/cs475-mlnlp-fall-2022-hw/blob/main/hw3/report/report.pdf).

## Problem

Different pooling strategy for sentences shows varying performance on different tasks.

- Q1. Compare `class BertPooler` which only takes `[CLS]`, with `class MeanMaxTokensBertPooler`.
- Q2. Build `class MyBertPooler` for a chosen task.

### Data

Microsoft Research Paraphrase Corpus (MRPC) was used to evaluate semantic textual similarity between sentences.

### Method

**TopKMeanBertPooler**

In paraphrasing tasks, retaining several important features in a sentence is important.  
Therefore, instead of focusing on one best feature, mean of top K features were taken.

### Result & Discussion

`TopKMeanBertPooler` outperformed `MeanMaxTokensBertPooler` and the pooler with `K = 20` showed the best performance among our experiment.

For future approaches:
- Analysis on changes in the performance for different K values
- Relationship of K with sentence length L to extend to other corpus
- Training 'Generalized Pooling Operator'
- Ensembles of different pooling methods
