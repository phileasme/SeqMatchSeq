# SeqMatchSeq (PyTorch)

Greatful for the original work provided by [pcgreat](https://github.com/pcgreat) .

This is a simplified, "more readable" version of the code. Emphasises on the different sections described in the paper applied to WikiQA.

Pytorch implementation of the model described in the papers related to sequence matching:

- [A Compare-Aggregate Model for Matching Text Sequences](https://arxiv.org/abs/1611.01747) by Shuohang Wang, Jing Jiang


### Requirements
- [Pytorch v0.1.12](http://pytorch.org/)
- tqdm
- Python 3

### Datasets
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage

WikiQA task:
```
sh preprocess.sh wikiqa (Please first dowload the file "WikiQACorpus.zip" to the path SeqMatchSeq/data/wikiqa/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419)

PYTHONPATH=. python3 main/main.py --task wikiqa --model compAggWikiqa --learning_rate 0.004 --dropoutP 0.04 --batch_size 10 --mem_dim 150

- `model` (model name) : CompAggWikiQA
```