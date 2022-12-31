# Explore Bias in Language Models with Fill Mask

> [Report](https://github.com/colorsquare/ml-for-nlp/blob/main/hw4/docs/report.pdf)

Review a paper, *Ahn & Oh, Mitigating Language-Dependent Ethnic Bias in BERT (EMNLP 2021)*.  
Using fill-mask, explore biases in BERT models of different languages.

Instructions are available at [hw4](https://github.com/uilab-kaist/cs475-mlnlp-fall-2022-hw/blob/main/hw4/report/report.pdf).

## Problem

Choose different attributes, targets, or templates to extract bias from BERT models.

- Q1. Explore bias found by the modification of attributes, targets, or templates.
- Q2. Discuss potential bias mitigation methods along with ones suggested in the paper above.

### Experiment

#### Attributes

Religious bias, one of which holds deep relationship with ethnicity is chosen.  
Therefore, the four most common religions (> 1%) are chosen as attributes: Christianity, Muslim, Hinduism, and Buddism.

#### Targets

Top 10 countries for each religion were set as targets.  
Removed countries with strong correlation with one religion. e.g. Vatican City (100% Christianity)

### Result & Discussion

Categorical Bias (CB) Score shows the level of bias in the BERT model for given targets (countries) and attributes (religion or others).

| LANG | Original Targets & Attributes | New Targets | New Targets & Attributes |
| ---- | ----------------------------- | ----------- | ------------------------ |
| EN   | 0.928                         | 1.019       | 2.103                    |
| ES   | 3.013                         | 2.856       | 5.229                    |

- Baseline (Original Targets & Attributes) successfully reproduced.
- Consistency verified for newly chosen targets (not deliberately chosen).
- Higher degree of bias observed for new attributes (religions).

Bias mitigation methods (Multilingual Model (M-BERT), and Contextual Word Alignment) from the original paper will not be as effective. 

- While M-BERT is expected to counterbalance different area of biases in different language models, religious bias is reproduced in different language models.
- Base language should be assumed to have low level of bias in contextual word alignment. However, religious bias is 'statistical' and 'universal', which means even languages with rich resources are likely to have high level of bias.
