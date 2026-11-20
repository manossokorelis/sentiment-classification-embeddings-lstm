"""Q3: embed and pool on IMDB with pretrained word embeddings.

Encoder: look up the vector of every token (tokens without a vector are dropped),
then mean- or sum-pool them into one 300-d vector. Classifier: an MLP softmax regressor.

Usage (from the project root):
    python src/embed_pool.py search   # fold 0, one change at a time -> results/q3_search.json
    python src/embed_pool.py cv NAME  # 5 folds x 3 seeds for a setting from SETTINGS -> results/q3_cv.json
    python src/embed_pool.py grid     # GRID, one change at a time, 15 runs each -> results/q3_grid.json
    python src/embed_pool.py combos   # COMBOS, 3 GloVe sizes x 3 MLPs, 15 runs each -> results/q3_combos.json
    python src/embed_pool.py regularization  # dropout, weight decay, batch size (3 each) -> results/q3_regularization.json
    python src/embed_pool.py reg-grid        # 2 dropout x 2 weight decay x 2 batch sizes -> results/q3_reg_grid.json
"""
import gzip
import html
import itertools
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path

import nltk
import numpy as np
import torch
from datasets import load_dataset
from nltk.corpus import stopwords
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

EMBEDDINGS = {
    "glove": dict(file="glove.42B.300d.zip", member="glove.42B.300d.txt", header=False),
}

# training setup of project 1's final model (MLP 1x16, AdamW lr 1e-4, weight decay 1e-4, dropout 0.1, batch 64),
# with early stopping: at least 20 epochs, then stop when the validation loss has not improved for `patience`
# epochs (at most `epochs`), keeping the epoch with the lowest validation loss
BASE = dict(embedding="glove", dims=300, pooling="mean", filter_tokens=False, finetune=False,
            hidden=16, layers=1, dropout=0.1, lr=1e-4, weight_decay=1e-4, batch_size=64,
            epochs=100, min_epochs=20, patience=3)

# 5-fold x 3-seed comparison, one change at a time from BASE, grouped by the kind of choice
ARCHITECTURES = {"1×16 units (base)": {}, "2×64 units": dict(hidden=64, layers=2),
                 "2×128 units": dict(hidden=128, layers=2), "3×128 units": dict(hidden=128, layers=3)}
GRID = {
    "Embedding size": {
        "50 dims": dict(dims=50),
        "100 dims": dict(dims=100),
        "200 dims": dict(dims=200),
        "300 dims (base)": {},
    },
    "Architecture": dict(ARCHITECTURES),
    "Optimisation": {
        "batch 64 (base)": {},
        "batch 32": dict(batch_size=32),
        "batch 16": dict(batch_size=16),
    },
}
# 3 x 3 combinations, as in project 1's 3 models x 3 representations: GloVe size x MLP size
COMBOS = {
    f"{mlp}": {f"{d} dims": dict(dims=d, **change) for d in (100, 200, 300)}
    for mlp, change in {"MLP 1×16": {}, "MLP 2×64": dict(hidden=64, layers=2),
                        "MLP 3×128": dict(hidden=128, layers=3)}.items()
}
# regularization and batch size, one change at a time from the best combination (300 dims, MLP 3x128)
BEST = dict(hidden=128, layers=3)
REGULARIZATION = {
    "Dropout": {"0": dict(BEST, dropout=0.0), "0.1 (base)": dict(BEST), "0.3": dict(BEST, dropout=0.3)},
    "Weight decay": {"0": dict(BEST, weight_decay=0.0), "1e-4 (base)": dict(BEST), "1e-2": dict(BEST, weight_decay=1e-2)},
    "Batch size": {"32": dict(BEST, batch_size=32), "64 (base)": dict(BEST), "128": dict(BEST, batch_size=128)},
}
# every combination of 2 dropout x 2 weight decay x 2 batch sizes (8 settings), on the best combination
REG_GRID = {
    f"dropout {d} · decay {w}": {f"batch {b}": dict(BEST, dropout=d, weight_decay=w, batch_size=b) for b in (32, 64)}
    for d in (0.1, 0.3) for w in (1e-4, 1e-2)
}
MODELS = RESULTS / "models"  # saved weights, one folder per run (not in git)

# fold-0 search: change one choice at a time from BASE
SEARCH = {
    "base": {},
    "sum pooling": dict(pooling="sum"),
    "no stopwords/punctuation": dict(filter_tokens=True),
    "fine-tuned": dict(finetune=True),
    "MLP 2x128": dict(hidden=128),
    "dropout 0.3": dict(dropout=0.3),
}

CV_SEEDS = [0, 3, 5]  # the seeds of project 1
NUM_FOLDS = 5


# ---------------------------------------------------------------- data

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb():
    """Training texts (tokenized as in the notebook: lowercase + NLTK) and labels, cached."""
    path = CACHE / "imdb_train_tokens.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d["tokens"], d["labels"]
    nltk.download("punkt_tab", quiet=True)
    ds = load_dataset("mteb/imdb")["train"]
    tokens = [nltk.word_tokenize(clean_text(t).lower()) for t in ds["text"]]
    labels = list(ds["label"])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(tokens=tokens, labels=labels)))
    return tokens, labels


def cross_validation_ids(n, k, seed):
    """Same shuffling and splitting as project 1's experiment.cross_validation_split."""
    random.seed(seed)
    ids = list(range(n))
    random.shuffle(ids)
    parts = np.array_split(np.array(ids, dtype=object), k)
    folds = {}
    for i in range(k):
        train = list(itertools.chain.from_iterable(p for j, p in enumerate(parts) if j != i))
        folds[i] = dict(training=[int(x) for x in train], validation=[int(x) for x in parts[i]])
    return folds


def read_lines(spec):
    path = RAW / spec["file"]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open(spec["member"]) as f:
            yield from TextIOWrapper(f, encoding="utf-8")
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f


def load_vectors(name, vocab):
    """Vectors for the IMDB word types that the embedding knows (lowercased lookup), cached."""
    path = CACHE / f"{name}_imdb.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return list(d["words"]), d["vectors"]
    spec = EMBEDDINGS[name]
    lines = read_lines(spec)
    if spec["header"]:
        next(lines)
    words, vecs = [], []
    seen = set()
    for line in lines:
        w, rest = line.rstrip("\n").split(" ", 1)
        w = w.lower()
        if w in vocab and w not in seen:
            v = np.asarray(rest.split(" "), dtype=np.float32)
            if v.shape[0] == 300:
                seen.add(w)
                words.append(w)
                vecs.append(v)
    vectors = np.vstack(vecs)
    np.savez(path, words=np.array(words), vectors=vectors)
    return words, vectors


# ---------------------------------------------------------------- model

class EmbedPoolMLP(nn.Module):
    def __init__(self, vectors, pooling, finetune, hidden, layers, dropout, num_classes=2):
        super().__init__()
        self.bag = nn.EmbeddingBag.from_pretrained(torch.tensor(vectors), freeze=not finetune, mode=pooling)
        dims = [vectors.shape[1]] + [hidden] * layers
        mlp = []
        for a, b in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        mlp.append(nn.Linear(dims[-1], num_classes))
        self.mlp = nn.Sequential(*mlp)

    def pool(self, flat, offsets):
        return self.bag(flat, offsets)

    def forward(self, flat, offsets):
        return self.mlp(self.pool(flat, offsets))


def batch_ids(seqs):
    lengths = [len(s) for s in seqs]
    flat = torch.tensor([i for s in seqs for i in s], dtype=torch.long)
    offsets = torch.tensor([0] + list(itertools.accumulate(lengths))[:-1], dtype=torch.long)
    return flat, offsets


def set_seed(seed):
    """Make a run reproducible: same seed -> same initialisation, batches and dropout masks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run(setting, ids, labels, train_idx, val_idx, vectors, seed):
    """Train one model with early stopping; return the metrics and the weights of the best epoch."""
    set_seed(seed)
    s = {"min_epochs": 0, **setting}
    model = EmbedPoolMLP(vectors, s["pooling"], s["finetune"], s["hidden"], s["layers"], s["dropout"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=s["lr"], weight_decay=s["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor(labels, dtype=torch.long)

    if not s["finetune"]:  # frozen: pool every review once, then train the MLP on 300-d features
        with torch.no_grad():
            feats = torch.cat([model.pool(*batch_ids([ids[i] for i in chunk]))
                               for chunk in np.array_split(np.arange(len(ids)), 50)])

    def forward(idx):
        if s["finetune"]:
            return model(*batch_ids([ids[i] for i in idx]))
        return model.mlp(feats[idx])

    gen = np.random.default_rng(seed)
    history, best, best_state, since_best = [], None, None, 0
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        order = gen.permutation(train_idx)
        total = 0.0
        for b in range(0, len(order), s["batch_size"]):
            idx = order[b:b + s["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(forward(idx), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl, correct = 0.0, 0
            for b in range(0, len(val_idx), 512):
                idx = np.asarray(val_idx[b:b + 512])
                out = forward(idx)
                vl += loss_fn(out, y[idx]).item() * len(idx)
                correct += (out.argmax(1) == y[idx]).sum().item()
        h = dict(epoch=epoch, train_loss=total / len(train_idx), val_loss=vl / len(val_idx),
                 val_acc=correct / len(val_idx))
        history.append(h)
        if best is None or h["val_loss"] < best["val_loss"]:
            best, since_best = h, 0
            # frozen: the embedding is GloVe itself, so only the MLP needs saving
            state = model.state_dict() if s["finetune"] else model.mlp.state_dict()
            best_state = {k: v.detach().clone() for k, v in state.items()}
        else:
            since_best += 1
        if epoch >= s["min_epochs"] and since_best >= s.get("patience", s["epochs"]):
            break  # early stopping
    return dict(best_epoch=best["epoch"], val_acc=best["val_acc"], val_loss=best["val_loss"],
                epochs_run=len(history), history=history, state=best_state)


# ---------------------------------------------------------------- experiments

STOP = None


def prepare(tokens, setting, cache):
    """Token ids per review for a setting's embedding and token filter (cached per combination)."""
    key = (setting["embedding"], setting["filter_tokens"], setting.get("dims", 300))
    if key in cache:
        return cache[key]
    if key[2] < 300:  # smaller embedding: PCA of the full vectors, keeping the top `dims` directions
        ids, vectors = prepare(tokens, {**setting, "dims": 300}, cache)
        centered = vectors - vectors.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        cache[key] = (ids, (centered @ vt[: key[2]].T).astype(np.float32))
        return cache[key]
    global STOP
    if STOP is None:
        nltk.download("stopwords", quiet=True)
        STOP = set(stopwords.words("english"))
    vocab = set(t for toks in tokens for t in toks)
    words, vectors = load_vectors(setting["embedding"], vocab)
    index = {w: i for i, w in enumerate(words)}
    keep = (lambda t: t not in STOP and any(c.isalnum() for c in t)) if setting["filter_tokens"] else (lambda t: True)
    ids = [[index[t] for t in toks if t in index and keep(t)] for toks in tokens]
    ids = [x if x else [index.get("the", 0)] for x in ids]  # never leave a review empty
    cache[key] = (ids, vectors)
    return cache[key]


def search(model_seeds=(0, 1, 2)):
    tokens, labels = load_imdb()
    fold = cross_validation_ids(len(tokens), NUM_FOLDS, CV_SEEDS[0])[0]
    out, cache = {}, {}
    for name, change in SEARCH.items():
        setting = {**BASE, **change}
        ids, vectors = prepare(tokens, setting, cache)
        runs = []
        for seed in model_seeds:
            t = time.time()
            r = run(setting, ids, labels, fold["training"], fold["validation"], vectors, seed)
            r["seconds"] = round(time.time() - t, 1)
            runs.append(r)
            print(f"{name:26s} seed {seed}: acc {r['val_acc']:.4f} (epoch {r['best_epoch']}, {r['seconds']}s)", flush=True)
        accs = [r["val_acc"] for r in runs]
        out[name] = dict(setting=setting, mean=float(np.mean(accs)), std=float(np.std(accs)), runs=runs)
        print(f"{name:26s} mean {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "q3_search.json").write_text(json.dumps(out, indent=1))


def cross_validate(name, change):
    tokens, labels = load_imdb()
    setting = {**BASE, **change}
    ids, vectors = prepare(tokens, setting, {})
    path = RESULTS / "q3_cv.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    res = dict(setting=setting, runs=[])
    for seed in CV_SEEDS:
        folds = cross_validation_ids(len(tokens), NUM_FOLDS, seed)
        for k in range(NUM_FOLDS):
            r = run(setting, ids, labels, folds[k]["training"], folds[k]["validation"], vectors, seed)
            res["runs"].append(dict(seed=seed, fold=k, val_acc=r["val_acc"], best_epoch=r["best_epoch"],
                                    history=r["history"]))
            print(f"{name} seed {seed} fold {k}: acc {r['val_acc']:.4f} (epoch {r['best_epoch']})", flush=True)
    accs = [r["val_acc"] for r in res["runs"]]
    res["mean"], res["std"] = float(np.mean(accs)), float(np.std(accs))
    allres[name] = res
    path.write_text(json.dumps(allres, indent=1))
    print(f"{name}: {res['mean']:.4f} ± {res['std']:.4f} over {len(accs)} runs", flush=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def grid(groups=None, path=RESULTS / "q3_grid.json"):
    """Every GRID setting on the 15 folds of project 1 (5 folds x seeds 0, 3, 5) -> results/q3_grid.json.

    Each run is saved in results/models/<group>/<setting>/seed-<s>/fold-<k>/: the best weights (model.pt),
    the setting and per-epoch history (run.json). Finished runs are loaded instead of retrained, so the
    script can be stopped and restarted. Identical settings (the base appears in two groups) are trained once.
    """
    tokens, labels = load_imdb()
    folds = {seed: cross_validation_ids(len(tokens), NUM_FOLDS, seed) for seed in CV_SEEDS}
    out, done, cache = {}, {}, {}
    for group, settings in (groups or GRID).items():
        out[group] = {}
        for name, change in settings.items():
            setting = {**BASE, **change}
            key = json.dumps(setting, sort_keys=True)
            if key not in done:
                folder = MODELS / slug(group) / slug(name)
                runs = []
                for seed in CV_SEEDS:
                    for k in range(NUM_FOLDS):
                        run_dir = folder / f"seed-{seed}" / f"fold-{k}"
                        if (run_dir / "run.json").exists():
                            saved = json.loads((run_dir / "run.json").read_text())
                            if saved["setting"] == setting:  # reuse only an identical run
                                runs.append(saved)
                                continue
                        ids, vectors = prepare(tokens, setting, cache)
                        t = time.time()
                        r = run(setting, ids, labels, folds[seed][k]["training"], folds[seed][k]["validation"], vectors, seed)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(r.pop("state"), run_dir / "model.pt")
                        info = dict(setting=setting, seed=seed, fold=k, seconds=round(time.time() - t, 1),
                                    torch=torch.__version__, **r)
                        (run_dir / "run.json").write_text(json.dumps(info, indent=1))
                        runs.append(info)
                        print(f"{group} | {name}: seed {seed} fold {k} acc {r['val_acc']:.4f} "
                              f"(best epoch {r['best_epoch']} of {r['epochs_run']}, {info['seconds']}s)", flush=True)
                accs = [r["val_acc"] for r in runs]
                summary = [{k: r[k] for k in ("seed", "fold", "val_acc", "best_epoch", "epochs_run", "seconds")} for r in runs]
                done[key] = dict(setting=setting, mean=float(np.mean(accs)), std=float(np.std(accs)), runs=summary)
            out[group][name] = done[key]
            print(f"== {group} | {name}: {done[key]['mean']:.4f} ± {done[key]['std']:.4f}", flush=True)
            RESULTS.mkdir(exist_ok=True)
            path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(8)
    if sys.argv[1] == "search":
        search()
    elif sys.argv[1] == "grid":
        grid()
    elif sys.argv[1] == "combos":
        grid(COMBOS, RESULTS / "q3_combos.json")
    elif sys.argv[1] == "regularization":
        grid(REGULARIZATION, RESULTS / "q3_regularization.json")
    elif sys.argv[1] == "reg-grid":
        grid(REG_GRID, RESULTS / "q3_reg_grid.json")
    elif sys.argv[1] == "cv":
        name = sys.argv[2]
        cross_validate(name, json.loads(sys.argv[3]) if len(sys.argv) > 3 else SEARCH[name])
