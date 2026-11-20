"""Q6: error analysis of the best embed-and-pool model vs the best LSTM.

Out-of-fold predictions: for each seed (0, 3, 5) the 5 cross-validation models together predict every one of the
24,904 training reviews exactly once, each by the model that did NOT train on it. No training happens here: the
saved weights of the 15 runs of each system are loaded and only predict.

  embed and pool: frozen GloVe 300d, mean pooling, MLP 3x128         results/models/mlp-3-128/300-dims/
  LSTM:           frozen GloVe 300d, LSTM 64, linear, 800 words, word dropout 0.1
                                                                      results/models/lstm/linear-64-frozen-word-dropout-0-1-800-words/

Then a few features of each review and how they relate to each system being right (majority vote of the 3 seeds).

Usage (from the project root):
    python src/error_analysis.py           # -> results/q6_predictions.json (cached), results/q6_analysis.json
    python src/error_analysis.py figures   # results/q6_wording.png, q6_contrast.png, q6_length.png
    python src/error_analysis.py test      # slide 6: the IMDB test set (24,678 reviews); each of the 15 models of each
                                           # system predicts every test review -> results/q6_test_analysis.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_pool as ep
import lstm as L

RESULTS = ep.RESULTS
EP_DIR = ep.MODELS / "mlp-3-128" / "300-dims"
LSTM_DIR = L.MODELS / "linear-64-frozen-word-dropout-0-1-800-words"
SYSTEMS = ["embed and pool", "LSTM"]

NEGATIONS = {"not", "n't", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without", "hardly", "cannot"}
CONTRAST = {"but", "however", "although", "though", "yet", "despite", "nevertheless", "whereas", "unfortunately"}
RATING = re.compile(r"^\d+(\.\d+)?/10$")


@torch.no_grad()
def predict_embed_pool(ids, idx, vectors, setting, state):
    model = ep.EmbedPoolMLP(vectors, setting["pooling"], False, setting["hidden"], setting["layers"], setting["dropout"])
    model.mlp.load_state_dict(state)
    model.eval()
    out = []
    for b in range(0, len(idx), 512):
        flat, offsets = ep.batch_ids([ids[i] for i in idx[b:b + 512]])
        out.append(torch.softmax(model(flat, offsets), -1)[:, 1])
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_lstm(ids, idx, vectors, setting, state):
    model = L.LSTMClassifier(vectors, setting).to(L.DEVICE)
    model.load_state_dict(state, strict=False)  # the frozen embedding is GloVe itself (not stored)
    model.eval()
    out = []
    for b in range(0, len(idx), 256):
        x, lengths = L.batch(ids, np.asarray(idx[b:b + 256]), setting["maxlen"])
        out.append(torch.softmax(model(x.to(L.DEVICE), lengths), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def predictions(tokens, ids, vectors):
    """P(positive) for every review, per system and seed (cached in results/q6_predictions.json)."""
    path = RESULTS / "q6_predictions.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {s: {} for s in SYSTEMS}
    for seed in ep.CV_SEEDS:
        folds = ep.cross_validation_ids(len(tokens), ep.NUM_FOLDS, seed)
        p = {s: np.full(len(tokens), np.nan) for s in SYSTEMS}
        for k in range(ep.NUM_FOLDS):
            val = list(folds[k]["validation"])
            d = EP_DIR / f"seed-{seed}" / f"fold-{k}"
            s_ep = json.loads((d / "run.json").read_text())["setting"]
            p["embed and pool"][val] = predict_embed_pool(ids, val, vectors, s_ep, torch.load(d / "model.pt", map_location="cpu"))
            d = LSTM_DIR / f"seed-{seed}" / f"fold-{k}"
            s_l = {**L.BASE, **json.loads((d / "run.json").read_text())["setting"]}
            p["LSTM"][val] = predict_lstm(ids, val, vectors, s_l, torch.load(d / "model.pt", map_location="cpu"))
            print(f"seed {seed} fold {k}: predicted {len(val)} reviews", flush=True)
        for s in SYSTEMS:
            assert not np.isnan(p[s]).any()
            out[s][str(seed)] = [round(float(v), 5) for v in p[s]]
    path.write_text(json.dumps(out))
    return out


def features(tokens, ids):
    """One row of features per review (see FEATURES for what each one is)."""
    from nltk.corpus import opinion_lexicon
    import nltk
    nltk.download("opinion_lexicon", quiet=True)
    pos, neg = set(opinion_lexicon.positive()), set(opinion_lexicon.negative())
    rows = []
    for t, i in zip(tokens, ids):
        n = len(t)
        p = sum(w in pos for w in t)
        q = sum(w in neg for w in t)
        rows.append(dict(
            length=n,
            unknown=1 - len(i) / n,
            negations=100 * sum(w in NEGATIONS for w in t) / n,
            contrast=100 * sum(w in CONTRAST for w in t) / n,
            pos_words=p, neg_words=q,
            lexicon_balance=(p - q) / (p + q) if p + q else 0.0,
            rating=int(any(RATING.match(w) for w in t)),
        ))
    return rows


def main():
    tokens, labels = ep.load_imdb()
    setting = {**ep.BASE, "embedding": "glove", "dims": 300, "filter_tokens": False}
    ids, vectors = ep.prepare(tokens, setting, {})
    y = np.array(labels)
    preds = predictions(tokens, ids, vectors)
    feats = features(tokens, ids)
    out = dict(n=len(y), systems={}, reviews=[])
    correct = {}
    for s in SYSTEMS:
        P = np.array([preds[s][str(seed)] for seed in ep.CV_SEEDS])         # 3 x n
        right = ((P > 0.5) == y[None, :])                                  # 3 x n
        correct[s] = right.mean(0) >= 0.5                                  # majority of the 3 seeds
        out["systems"][s] = dict(accuracy_per_seed=[float(r.mean()) for r in right],
                                 accuracy_majority=float(correct[s].mean()),
                                 predicted_positive=float((P > 0.5).mean()))
        print(f"{s:15s} accuracy per seed {[round(100 * r.mean(), 2) for r in right]}, "
              f"majority {100 * correct[s].mean():.2f}%, predicts positive {100 * (P > 0.5).mean():.1f}%")
    both = correct["embed and pool"] & correct["LSTM"]
    print(f"both right {100 * both.mean():.1f}%, only LSTM {100 * (correct['LSTM'] & ~correct['embed and pool']).mean():.1f}%, "
          f"only embed and pool {100 * (correct['embed and pool'] & ~correct['LSTM']).mean():.1f}%, "
          f"both wrong {100 * (~correct['LSTM'] & ~correct['embed and pool']).mean():.1f}%")
    for i, f in enumerate(feats):
        out["reviews"].append(dict(label=int(y[i]), ep_right=bool(correct["embed and pool"][i]),
                                   lstm_right=bool(correct["LSTM"][i]), **f))
    (RESULTS / "q6_analysis.json").write_text(json.dumps(out))
    print("saved", RESULTS / "q6_analysis.json")


# ---- slide 6 figures: accuracy of both systems per bin of one feature (all 24,904 reviews, out-of-fold)
BINS = {
    "wording": ("lex_agree", [-1.01, -0.2, 0, 0.2, 0.5, 1.01],
                ["against", "slightly\nagainst", "slightly\nwith", "with", "clearly\nwith"],
                "opinion words vs the label", (55, 100), 10),
    "contrast": ("contrast", [0, 0.5, 1, 1.5, 2, 1e9], ["<0.5", "0.5–1", "1–1.5", "1.5–2", ">2"],
                 "contrast words per 100 words", (80, 95), 5),
    "length": ("length", [0, 100, 200, 400, 800, 1e9], ["<100", "100–\n200", "200–\n400", "400–\n800", ">800"],
               "review length (words)", (80, 95), 5),
}


def binned_figure(out, key):
    import matplotlib
    import matplotlib.pyplot as plt
    import compare as C
    feat, edges, ticks, xlabel, ylim, step = BINS[key]
    R = json.loads((RESULTS / "q6_analysis.json").read_text())["reviews"]
    v = np.array([r[feat] if feat != "lex_agree" else (r["lexicon_balance"] if r["label"] else -r["lexicon_balance"])
                  for r in R])
    right = {"embed and pool": np.array([r["ep_right"] for r in R]), "LSTM": np.array([r["lstm_right"] for r in R])}
    C.setup_fonts()
    fig, ax = plt.subplots(figsize=(4.6, 6.0), dpi=200)
    x = np.arange(len(ticks))
    for name, c in (("embed and pool", C.PINK), ("LSTM", C.PURPLE)):
        acc = [100 * right[name][(v >= lo) & (v < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(x, acc, color=c, linewidth=2.6, marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=1.4, zorder=3, label=name)
        print(f"{key:9s} {name:15s}", [round(a, 1) for a in acc])
    shares = [100 * ((v >= lo) & (v < hi)).mean() for lo, hi in zip(edges[:-1], edges[1:])]
    for xi, sh in zip(x, shares):  # how many reviews fall in each bin
        ax.annotate(f"{sh:.0f}%", (xi, ylim[0]), xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=11.5, color=C.FAINT)
    if key == "length":  # the LSTM reads only the first 800 words
        ax.axvline(3.5, color=C.GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.6, zorder=1)
        ax.annotate("LSTM reads\n800 words", (3.5, ylim[1]), xytext=(-5, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=12, color=C.GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=13, color=C.INK, linespacing=1.0)
    ax.set_xlim(-0.4, len(ticks) - 0.6)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:.0f}%"))
    ax.set_xlabel(xlabel, color=C.INK, fontsize=15, weight="bold", labelpad=10)
    ax.set_ylabel("accuracy", color=C.MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=C.GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=C.MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    if key == "wording":
        leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), frameon=False, fontsize=13, handlelength=1.4)
        for t, c in zip(leg.get_texts(), (C.PINK, C.PURPLE)):
            t.set_color(c)
            t.set_fontweight("bold")
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


# ---- the IMDB test set: never used in training, early stopping or choosing a model

def load_imdb_test():
    """Test texts, tokenized like the training set (lowercase + NLTK), cached."""
    path = ep.CACHE / "imdb_test_tokens.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d["tokens"], d["labels"]
    import nltk
    from datasets import load_dataset
    ds = load_dataset("mteb/imdb")["test"]
    tokens = [nltk.word_tokenize(ep.clean_text(t).lower()) for t in ds["text"]]
    labels = list(ds["label"])
    path.write_text(json.dumps(dict(tokens=tokens, labels=labels)))
    return tokens, labels


def test_vectors(test_tokens):
    """GloVe vectors of the training vocabulary plus the words that occur only in the test reviews (one pass over
    GloVe, cached). The embeddings are frozen and not stored in the models, so the order of the matrix does not matter."""
    words, vectors = ep.load_vectors("glove", set())  # the cached training vocabulary
    path = ep.CACHE / "glove_imdb_test_extra.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        extra_words, extra_vecs = list(d["words"]), d["vectors"]
    else:
        need = {t for toks in test_tokens for t in toks} - set(words)
        extra_words, extra_vecs, seen = [], [], set()
        for line in ep.read_lines(ep.EMBEDDINGS["glove"]):
            w, rest = line.rstrip("\n").split(" ", 1)
            w = w.lower()
            if w in need and w not in seen:
                v = np.asarray(rest.split(" "), dtype=np.float32)
                if v.shape[0] == 300:
                    seen.add(w)
                    extra_words.append(w)
                    extra_vecs.append(v)
        extra_vecs = np.vstack(extra_vecs) if extra_vecs else np.zeros((0, 300), np.float32)
        np.savez(path, words=np.array(extra_words), vectors=extra_vecs)
    return list(words) + list(extra_words), np.vstack([vectors, extra_vecs]).astype(np.float32)


def test_main():
    """Every one of the 15 models of each system predicts every test review."""
    tokens, labels = load_imdb_test()
    words, vectors = test_vectors(tokens)
    index = {w: i for i, w in enumerate(words)}
    ids = [[index[t] for t in toks if t in index] for toks in tokens]
    ids = [x if x else [index.get("the", 0)] for x in ids]  # never leave a review empty (as in training)
    y = np.array(labels)
    idx = list(range(len(y)))
    path = RESULTS / "q6_test_predictions.npz"
    if path.exists():
        P = dict(np.load(path))
    else:
        P = {s: [] for s in SYSTEMS}
        for seed in ep.CV_SEEDS:
            for k in range(ep.NUM_FOLDS):
                d = EP_DIR / f"seed-{seed}" / f"fold-{k}"
                s_ep = json.loads((d / "run.json").read_text())["setting"]
                P["embed and pool"].append(predict_embed_pool(ids, idx, vectors, s_ep, torch.load(d / "model.pt", map_location="cpu")))
                d = LSTM_DIR / f"seed-{seed}" / f"fold-{k}"
                s_l = {**L.BASE, **json.loads((d / "run.json").read_text())["setting"]}
                P["LSTM"].append(predict_lstm(ids, idx, vectors, s_l, torch.load(d / "model.pt", map_location="cpu")))
                print(f"seed {seed} fold {k}: predicted {len(idx)} test reviews", flush=True)
        P = {s: np.array(v, dtype=np.float32) for s, v in P.items()}
        np.savez(path, **P)
    out = dict(n=len(y), systems={}, reviews=[])
    right = {}
    for s in SYSTEMS:
        r = (P[s] > 0.5) == y[None, :]  # 15 models x reviews
        right[s] = r.mean(0)  # share of the 15 models that are right on each review
        acc = 100 * r.mean(1)
        out["systems"][s] = dict(accuracy_per_model=acc.round(3).tolist(), mean=float(acc.mean()), std=float(acc.std()))
        print(f"{s:15s} test accuracy {acc.mean():.2f} ± {acc.std():.2f} (15 models)")
    for i, f in enumerate(features(tokens, ids)):
        out["reviews"].append(dict(label=int(y[i]), ep_right=float(right["embed and pool"][i]),
                                   lstm_right=float(right["LSTM"][i]), **f))
    (RESULTS / "q6_test_analysis.json").write_text(json.dumps(out))
    print("saved", RESULTS / "q6_test_analysis.json")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test_main()
    elif sys.argv[1:] == ["figures"]:
        sys.argv = sys.argv[:1]
        for key in BINS:
            binned_figure(RESULTS / f"q6_{key}.png", key)
    else:
        main()
