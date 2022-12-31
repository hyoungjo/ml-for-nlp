# N-Gram Bag-of-Words Classification

> [feat(hw1): n-gram bag-of-words classification #1](https://github.com/colorsquare/ml-for-nlp/pull/1)

```python
ArrayLike = Union[list, tuple, np.ndarray]

def preprocess_and_split_to_tokens(sentences: ArrayLike, n_gram: int) -> ArrayLike:
    # TODO: return `tokens_per_sentence`

def create_bow(sentences: ArrayLike, n_gram: int, vocab: Dict[str, int] = None,  
               msg_prefix="\n") -> Tuple[Dict[str, int], ArrayLike]:
    # TODO: return `(vocab, bow_array)`
```

Instructions are available at [hw1](https://github.com/uilab-kaist/cs475-mlnlp-fall-2022-hw/tree/main/hw1).

## Problem

**Goal:** Validation accuracy *0.857 for 1-gram* and *0.814 for 2-gram* (num samples: 5000).

- Q1. Which words do you want?
- Q2. Which words do you want to join as n-gram?

## Getting started

Environment setup with Anaconda.

### Creating a new environment

```sh
$ conda create --name ENV_NAME python=3.x
$ conda activate ENV_NAME
```

### Installing dependencies

```sh
$ conda install joblib=1.0.1 numpy=1.20.1 scikit-learn=0.24.1 scipy=1.6.1 termcolor=1.1.0 threadpoolctl=2.1.0
$ conda install -c conda-forge tqdm=4.58.0
```

### Execution

```sh
$ python3 bow_classification_with_sklearn.py --num-samples 5000 --verbose True --n_gram 1
```
