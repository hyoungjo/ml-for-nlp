# Machine Learning for Natural Language Processing

Different tasks of machine learning for natural language processing. Based on CS475, 2022 Fall, KAIST.

## About the Course and Tasks

Refer to the [course homepage](https://uilab-kaist.github.io/cs475-mlnlp-fall-2022/), and [coursework repository](https://github.com/uilab-kaist/cs475-mlnlp-fall-2022-hw).

## Getting Started

### Environment setup with anaconda

#### Create a new environment
```sh
$ conda create --name ENV_NAME python=3.x
$ conda activate ENV_NAME
```

#### Install dependencies

For hw1,

```sh
$ conda install joblib=1.0.1 numpy=1.20.1 scikit-learn=0.24.1 scipy=1.6.1 termcolor=1.1.0 threadpoolctl=2.1.0
$ conda install -c conda-forge tqdm=4.58.0
```

### Execution

For hw1,

```sh
$ python3 bow_classification_with_sklearn.py --num-samples 5000 --verbose True --n_gram 1
```
