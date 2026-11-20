"""Q4: a recurrent encoder (LSTM) on IMDB with pretrained GloVe embeddings.

Encoder (as in tutorial 2): embed every token (tokens without a vector are dropped, as in Q3), read the first
`maxlen` tokens with a 1-layer LSTM and use its last hidden state. Classifier: the same MLP head as Q3 (1x16).
Training keeps the epoch with the lowest validation loss and stops after `patience` epochs without improvement.

Usage (from the project root):
    python src/lstm.py NAME [--cv]   # one setting from SETTINGS on fold 0 x seeds 0, 3, 5 (or all 15 folds with --cv)
    python src/lstm.py grid --cv     # step 1: LSTM 32 / 64 / 128 units, frozen GloVe, linear head, 15 runs each
    python src/lstm.py step2 "SETTING" --cv      # step 2 on the best size: bidirectional, mean of hidden states
    python src/lstm.py step3 "SETTING" --cv      # step 3 on the best of step 2: word dropout 0 / 0.1 / 0.25 x 200 / 400 / 800 words
    python src/lstm.py step4 [--cv | --one]      # step 4 on the best of step 3 (800 words, word dropout 0.1):
                                                 # 1k / 5k / all training reviews x frozen / fine-tuned / random vectors
    --one: only seed 0, fold 0 (a quick first look, saved under "seed 0 fold 0")
    --emb fine-tuned: step 4 for one kind of embedding only (frozen / fine-tuned / random)

Every run is saved in results/models/lstm/<setting>/seed-<s>/fold-<k>/ (weights + run.json with the per-epoch
history); finished runs are reused. Seeds are fixed; on the Mac GPU (MPS) PyTorch does not guarantee
bit-identical results, so reruns can differ in the last decimals.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_pool as ep

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS = ep.RESULTS
MODELS = RESULTS / "models" / "lstm"

# base LSTM (experiment 1): Q3's tokenizer, GloVe 300d frozen and MLP head 1x16; lr 1e-3 (an LSTM learns
# very slowly at 1e-4), reviews cut to their first 400 tokens, early stopping with patience 3 (at most 30 epochs)
BASE = dict(embedding="glove", dims=300, filter_tokens=False, finetune=False, maxlen=400, hidden=64,
            bidirectional=False, readout="last", word_dropout=0.0, mlp_hidden=16, dropout=0.1, lr=1e-3, weight_decay=1e-4,
            batch_size=64, epochs=30, min_epochs=0, patience=3, train_size=None, emb_lr=None, emb_from=None,
            init="glove")  # init="random": same vocabulary, vectors drawn at random (scale of GloVe) and trained

SETTINGS = {
    "base": {},
    "fine-tuned": dict(finetune=True),
    "word dropout 0.25": dict(word_dropout=0.25),
    "1,000 reviews": dict(train_size=1000),
    "5,000 reviews": dict(train_size=5000),
    # slide 5 grid: 300 dims, one LSTM layer, a linear layer straight to the 2 classes (mlp_hidden=None);
    # LSTM 64 or 128 units x GloVe frozen or fine-tuned (vectors at lr 1e-4, 10x below the rest)
    "linear · 32 · frozen": dict(mlp_hidden=None, hidden=32),
    "linear · 64 · frozen": dict(mlp_hidden=None),
    "linear · 128 · frozen": dict(mlp_hidden=None, hidden=128),
    "linear · 64 · fine-tuned": dict(mlp_hidden=None, finetune=True, emb_lr=1e-4),
    "linear · 128 · fine-tuned": dict(mlp_hidden=None, hidden=128, finetune=True, emb_lr=1e-4),
}
# step 2, after the grid, on the best of frozen / fine-tuned (see `step2` below):
WORD_DROPOUT = [0.1, 0.25]  # share of word vectors zeroed during training (base 0)
WORD_COUNT = [200, 800]     # how many words of each review the LSTM reads (base 400)
# plus: bidirectional (base: no) and the readout, average of all hidden states (base: last hidden state)
GRID = ["linear · 32 · frozen", "linear · 64 · frozen", "linear · 128 · frozen"]  # step 1: hidden units
# step 4, on the best of step 3; frozen + all reviews IS that setting (already run, reused under its own name)
STEP4_BASE = "linear · 64 · frozen · word dropout 0.1 · 800 words"
SETTINGS[STEP4_BASE] = dict(mlp_hidden=None, word_dropout=0.1, maxlen=800)
STEP4_SIZES = {"1k reviews": 1000, "5k reviews": 5000, "all reviews": None}
STEP4_EMB = {"frozen": {},
             "fine-tuned": dict(finetune=True, emb_lr=1e-4),          # GloVe vectors at lr 1e-4, 10x below the rest
             "random": dict(finetune=True, init="random")}            # random vectors learn at the full lr 1e-3


class LSTMClassifier(nn.Module):
    def __init__(self, vectors, s):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(vectors), freeze=not s["finetune"])
        self.word_dropout = s["word_dropout"]
        self.readout = s.get("readout", "last")
        self.lstm = nn.LSTM(vectors.shape[1], s["hidden"], batch_first=True, bidirectional=s["bidirectional"])
        enc = s["hidden"] * (2 if s["bidirectional"] else 1)
        if s["mlp_hidden"] is None:  # linear layer straight to the 2 classes
            self.mlp = nn.Sequential(nn.Dropout(s["dropout"]), nn.Linear(enc, 2))
        else:
            self.mlp = nn.Sequential(nn.Linear(enc, s["mlp_hidden"]), nn.ReLU(), nn.Dropout(s["dropout"]),
                                     nn.Linear(s["mlp_hidden"], 2))

    def forward(self, x, lengths):
        e = self.emb(x)
        if self.training and self.word_dropout > 0:  # zero whole word vectors at random
            e = e * (torch.rand(x.shape, device=x.device) >= self.word_dropout).unsqueeze(-1)
        out, (h, _) = self.lstm(pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False))
        if self.readout == "mean":  # average of the hidden states at every word (padding excluded)
            out, lens = pad_packed_sequence(out, batch_first=True)
            mask = (torch.arange(out.shape[1])[None, :] < lens[:, None]).to(out.device).unsqueeze(-1)
            h = (out * mask).sum(1) / lens.to(out.device).unsqueeze(-1)
        else:  # last hidden state (of both directions when bidirectional)
            h = torch.cat([h[-2], h[-1]], dim=-1) if self.lstm.bidirectional else h[-1]
        return self.mlp(h)


def batch(ids, idx, maxlen):
    seqs = [torch.tensor(ids[i][:maxlen]) for i in idx]
    return pad_sequence(seqs, batch_first=True), torch.tensor([len(q) for q in seqs])


def run(s, ids, labels, train_idx, val_idx, vectors, seed):
    ep.set_seed(seed)
    model = LSTMClassifier(vectors, s).to(DEVICE)
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("emb.")]
    groups = [dict(params=rest)]
    if s["finetune"]:  # the GloVe vectors get their own (smaller) learning rate when emb_lr is set
        groups.append(dict(params=list(model.emb.parameters()), lr=s.get("emb_lr") or s["lr"]))
    opt = torch.optim.AdamW(groups, lr=s["lr"], weight_decay=s["weight_decay"])
    y = torch.tensor(labels)
    gen = np.random.default_rng(seed)
    history, best, best_state, since = [], None, None, 0
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        total = 0.0
        order = gen.permutation(train_idx)
        for b in range(0, len(order), s["batch_size"]):
            idx = order[b:b + s["batch_size"]]
            x, lengths = batch(ids, idx, s["maxlen"])
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x.to(DEVICE), lengths), y[idx].to(DEVICE))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        vl, correct = 0.0, 0
        with torch.no_grad():
            for b in range(0, len(val_idx), 256):
                idx = np.asarray(val_idx[b:b + 256])
                x, lengths = batch(ids, idx, s["maxlen"])
                out = model(x.to(DEVICE), lengths).cpu()
                vl += nn.functional.cross_entropy(out, y[idx], reduction="sum").item()
                correct += (out.argmax(1) == y[idx]).sum().item()
        h = dict(epoch=epoch, train_loss=total / len(train_idx), val_loss=vl / len(val_idx), val_acc=correct / len(val_idx))
        history.append(h)
        print(f"  epoch {epoch}: train {h['train_loss']:.4f} val {h['val_loss']:.4f} acc {h['val_acc']:.4f}", flush=True)
        if best is None or h["val_loss"] < best["val_loss"]:
            best, since = h, 0
            state = model.state_dict()
            if not s["finetune"]:  # the frozen embedding is GloVe itself: don't store it
                state = {k: v for k, v in state.items() if not k.startswith("emb.")}
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        else:
            since += 1
        if epoch >= s["min_epochs"] and since >= s["patience"]:
            break
    return dict(best_epoch=best["epoch"], val_acc=best["val_acc"], val_loss=best["val_loss"],
                epochs_run=len(history), history=history, state=best_state)


def finetuned_vectors(setting_name, seed, fold):
    """The GloVe vectors as fine-tuned by an earlier setting, for the SAME seed and fold.

    Each fold has its own copy: vectors fine-tuned on one fold's training reviews have seen other folds'
    validation reviews, so reusing them across folds would leak validation data into training.
    """
    path = MODELS / ep.slug(setting_name) / f"seed-{seed}" / f"fold-{fold}" / "model.pt"
    return torch.load(path, map_location="cpu")["emb.weight"].numpy()


def subset(train_idx, size, seed):
    """A fixed random subset of the training part (the same one for every setting with this seed)."""
    if size is None:
        return list(train_idx)
    return sorted(np.random.default_rng(seed).choice(train_idx, size, replace=False).tolist())


def random_vectors(vectors, seed):
    """Random vectors for the same vocabulary, with the same per-dimension mean and spread as GloVe."""
    rng = np.random.default_rng(seed)
    return (vectors.mean(0) + vectors.std(0) * rng.standard_normal(vectors.shape)).astype(vectors.dtype)


def run_setting(name, folds_per_seed, seeds=None):
    s = {**BASE, **SETTINGS[name]}
    tokens, labels = ep.load_imdb()
    ids, vectors = ep.prepare(tokens, {**ep.BASE, **{k: s[k] for k in ("embedding", "dims", "filter_tokens")}}, {})
    runs = []
    for seed in (seeds or ep.CV_SEEDS):
        folds = ep.cross_validation_ids(len(tokens), ep.NUM_FOLDS, seed)
        for k in folds_per_seed:
            run_dir = MODELS / ep.slug(name) / f"seed-{seed}" / f"fold-{k}"
            if (run_dir / "run.json").exists():
                saved = json.loads((run_dir / "run.json").read_text())
                if {**BASE, **saved["setting"]} == s:  # runs saved before a new BASE key was added still count
                    runs.append(saved)
                    continue
            print(f"{name}: seed {seed} fold {k}", flush=True)
            t = time.time()
            vecs = vectors if s["emb_from"] is None else finetuned_vectors(s["emb_from"], seed, k)
            if s["init"] == "random":
                vecs = random_vectors(vectors, seed)
            r = run(s, ids, labels, subset(folds[k]["training"], s["train_size"], seed), folds[k]["validation"], vecs, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(r.pop("state"), run_dir / "model.pt")
            info = dict(setting=s, seed=seed, fold=k, device=DEVICE, seconds=round(time.time() - t, 1), torch=torch.__version__, **r)
            (run_dir / "run.json").write_text(json.dumps(info, indent=1))
            runs.append(info)
            print(f"{name}: seed {seed} fold {k} acc {r['val_acc']:.4f} (best epoch {r['best_epoch']} of "
                  f"{r['epochs_run']}, {info['seconds']}s)", flush=True)
    accs = [r["val_acc"] for r in runs]
    res = dict(setting=s, mean=float(np.mean(accs)), std=float(np.std(accs)),
               runs=[{k: r[k] for k in ("seed", "fold", "val_acc", "best_epoch", "epochs_run", "seconds")} for r in runs])
    path = RESULTS / "q4_lstm.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    key = "seed 0 fold 0" if seeds == [0] else "15 folds" if len(folds_per_seed) > 1 else "fold 0"
    out.setdefault(key, {})[name] = res
    path.write_text(json.dumps(out, indent=1))
    print(f"== {name}: {res['mean']:.4f} ± {res['std']:.4f} over {len(accs)} runs", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    folds = range(ep.NUM_FOLDS) if "--cv" in sys.argv else [0]
    seeds = [0] if "--one" in sys.argv else None
    if sys.argv[1] == "step4":  # data size x embeddings; frozen + all reviews is STEP4_BASE itself (reused)
        only = sys.argv[sys.argv.index("--emb") + 1] if "--emb" in sys.argv else None  # e.g. --emb fine-tuned
        for emb, e in STEP4_EMB.items():
            if only and emb != only:
                continue
            for size, n in STEP4_SIZES.items():
                name = STEP4_BASE if (emb, n) == ("frozen", None) else f"{STEP4_BASE} · {emb} · {size}"
                SETTINGS[name] = dict(SETTINGS[STEP4_BASE], train_size=n, **e)
                run_setting(name, folds, seeds)
    elif sys.argv[1] in ("step2", "step3"):  # on top of the chosen setting, e.g. "linear · 64 · frozen"
        chosen = sys.argv[2]
        if sys.argv[1] == "step2":  # architecture: bidirectional, and the readout (mean of all hidden states)
            changes = {"bidirectional": dict(bidirectional=True), "mean of hidden states": dict(readout="mean")}
        else:  # tuning: every combination of word dropout (0 / 0.1 / 0.25) x word count (200 / 400 / 800)
            changes = {}
            for p in [0.0] + WORD_DROPOUT:
                for n in sorted(WORD_COUNT + [400]):
                    if (p, n) != (0.0, 400):  # the base itself is already done
                        changes[f"word dropout {p} · {n} words"] = dict(word_dropout=p, maxlen=n)
        for label, change in changes.items():
            base = dict(SETTINGS[chosen])
            if base.get("finetune"):  # reuse the vectors fine-tuned earlier (same seed and fold), now frozen
                base.update(finetune=False, emb_lr=None, emb_from=chosen)
            SETTINGS[f"{chosen} · {label}"] = dict(base, **change)
            run_setting(f"{chosen} · {label}", folds)
    else:
        for name in (GRID if sys.argv[1] == "grid" else [sys.argv[1]]):
            run_setting(name, folds)
