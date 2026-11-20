# Sentiment classification: pretrained embeddings and LSTMs

We classify IMDB movie reviews as *negative* or *positive* with pretrained word embeddings:

- **Embedding:** GloVe 42B (300 dimensions, trained on Common Crawl web text)
- **2 encoders:** embed and pool, and an LSTM
- **Evaluation:** 5-fold cross-validation on the same folds as project 1

Results: embed and pool reaches **85.7%**, the LSTM **90.0%** (project 1's bpe-logcounts-mlp: 88.2%).

## Contents

```
├── notebooks/
│   ├── project.ipynb            the project assignment
│   └── theory.ipynb             theory: word embeddings, vector semantics
├── src/
│   ├── embed_pool.py            embed and pool: data, folds, model and training
│   ├── lstm.py                  the LSTM and its step-by-step search
│   ├── error_analysis.py        out-of-fold predictions of both models and review features
│   └── compare.py               figures of the slides
├── docs/
│   ├── presentation.pdf         the slides (PDF)
│   └── presentation.html        the slides (HTML, open in a browser)
└── data/                        downloaded embeddings (not in git)
```

## Setup

We are using python 3.15, here is how you reproduce our setup.

```bash
python3.15 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To run the experiments (hours on a laptop; finished runs are reused):

```bash
python src/embed_pool.py combos            # slide 4: 3 MLPs x 100/200/300 dims, 15 runs each
python src/embed_pool.py regularization    # slide 4: dropout, weight decay, batch size
python src/lstm.py grid --cv               # slide 5, step 1: LSTM 32 / 64 / 128 units
python src/lstm.py step2 "linear · 64 · frozen" --cv   # step 2: bidirectional, average of states
python src/lstm.py step3 "linear · 64 · frozen" --cv   # step 3: words read x word dropout
python src/lstm.py step4 --cv --emb frozen             # step 4: data size, frozen GloVe
python src/lstm.py step4 --cv --emb fine-tuned         #         and fine-tuned GloVe
python src/error_analysis.py               # slide 6: predictions and features
python src/compare.py                      # figures
```

Before the first run, download GloVe 42B ([glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip), about 1.9 GB)
into `data/raw/`. The IMDB dataset ([mteb/imdb](https://huggingface.co/datasets/mteb/imdb)) is downloaded from Hugging Face on the first run.

## License and acknowledgements

The code is adapted from [nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student),
released under the Apache License 2.0. This project is released under the same license; see [`LICENSE`](LICENSE).
