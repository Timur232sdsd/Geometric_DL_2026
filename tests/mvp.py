from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@dataclass
class Poset:
    name: str
    n: int
    edges: List[Tuple[int, int]]
    G: nx.DiGraph


def make_chain(n: int = 24) -> Poset:
    edges = [(i, i + 1) for i in range(n - 1)]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return Poset(name="chain", n=n, edges=edges, G=G)


def make_binary_tree(depth: int = 3) -> Poset:
    n = 2 ** (depth + 1) - 1
    edges = []
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            edges.append((i, left))
        if right < n:
            edges.append((i, right))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return Poset(name="tree", n=n, edges=edges, G=G)


def make_grid(h: int = 5, w: int = 5) -> Poset:
    edges = []
    n = h * w
    for i in range(h):
        for j in range(w):
            u = i * w + j
            if i + 1 < h:
                edges.append((u, (i + 1) * w + j))
            if j + 1 < w:
                edges.append((u, i * w + (j + 1)))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return Poset(name="grid", n=n, edges=edges, G=G)


def get_poset(name: str) -> Poset:
    if name == "chain":
        return make_chain(24)
    if name == "tree":
        return make_binary_tree(3)
    if name == "grid":
        return make_grid(5, 5)
    raise ValueError(f"Unknown poset: {name}")


def default_sources(poset: Poset) -> List[int]:
    if poset.name == "chain":
        return [poset.n // 2]
    if poset.name == "tree":
        return [3, 6] if poset.n >= 15 else [1]
    if poset.name == "grid":
        return [12, 16] if poset.n >= 25 else [poset.n // 2]
    return [0]


def make_task(
    poset: Poset,
    sources: List[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
    positive = set()
    for s in sources:
        positive.add(s)
        positive.update(nx.descendants(poset.G, s))

    y = np.zeros(poset.n, dtype=np.int64)
    for v in positive:
        y[v] = 1

    rng = np.random.default_rng(seed)
    x_random = rng.normal(size=(poset.n, 8)).astype(np.float32)
    x_source = np.zeros((poset.n, 1), dtype=np.float32)
    for s in sources:
        x_source[s, 0] = 1.0

    x = np.concatenate([x_source, x_random], axis=1)
    idx = np.arange(poset.n)

    try:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.4, random_state=seed, stratify=y
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx]
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    masks: Dict[str, torch.Tensor] = {}
    for name, arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        mask = torch.zeros(poset.n, dtype=torch.bool)
        mask[arr] = True
        masks[name] = mask
    return x, y, masks


def build_gcn_adjacency(edges: List[Tuple[int, int]], n: int) -> torch.Tensor:
    a = np.eye(n, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    deg = a.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    a_hat = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    return torch.from_numpy(a_hat.astype(np.float32))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, *_unused):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, h


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, a):
        h = F.relu(torch.mm(a, self.w1(x)))
        logits = self.w2(h)
        return logits, h


class SimpleSheafNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        stalk_dim: int = 8,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.r_lower = nn.Linear(hidden_dim, stalk_dim, bias=False)
        self.r_upper = nn.Linear(hidden_dim, stalk_dim, bias=False)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.alpha = float(alpha)

    def forward(self, x, edge_index):
        h = F.relu(self.proj(x))
        if edge_index.shape[1] > 0:
            lower = edge_index[0]
            upper = edge_index[1]
            z_lower = self.r_lower(h[lower])
            z_upper = self.r_upper(h[upper])
            diff = z_lower - z_upper

            msg_lower = diff @ self.r_lower.weight
            msg_upper = -diff @ self.r_upper.weight

            grad = torch.zeros_like(h)
            grad.index_add_(0, lower, msg_lower)
            grad.index_add_(0, upper, msg_upper)

            deg = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)
            deg.index_add_(0, lower, torch.ones(lower.size(0), dtype=h.dtype, device=h.device))
            deg.index_add_(0, upper, torch.ones(upper.size(0), dtype=h.dtype, device=h.device))
            deg = deg.clamp_min(1.0).unsqueeze(1)

            h = h - self.alpha * grad / deg

        h = F.relu(h)
        logits = self.out(h)
        return logits, h


def model_forward(model, name: str, x, a, edge_index):
    if name == "mlp":
        return model(x)
    if name == "gcn":
        return model(x, a)
    if name == "snn":
        return model(x, edge_index)
    raise ValueError(f"Unknown model: {name}")


def _clean_diagram(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    if d.shape[1] > 2:
        d = d[:, :2]
    finite = d[np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])]
    if finite.size == 0:
        return np.empty((0, 2), dtype=float)
    life = finite[:, 1] - finite[:, 0]
    return finite[life > 1e-12]


def compute_diagrams(
    embeddings: np.ndarray,
    maxdim: int = 1,
    subsample: int = 500,
    pca_dim: int = 10,
    seed: int = 0,
) -> List[np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.shape[0] < 2:
        return [np.empty((0, 2), dtype=float) for _ in range(maxdim + 1)]

    if x.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=subsample, replace=False)
        x = x[idx]

    x = StandardScaler().fit_transform(x)
    if x.shape[1] > pca_dim:
        n_components = min(pca_dim, x.shape[0], x.shape[1])
        if n_components >= 2:
            x = PCA(n_components=n_components, random_state=seed).fit_transform(x)

    try:
        from ripser import ripser

        result = ripser(x, maxdim=maxdim, thresh=np.inf)
        return [_clean_diagram(dgm) for dgm in result["dgms"]]
    except Exception:
        try:
            from gtda.homology import VietorisRipsPersistence

            homology_dimensions = list(range(maxdim + 1))
            vr = VietorisRipsPersistence(
                metric="euclidean",
                max_edge_length=np.inf,
                homology_dimensions=homology_dimensions,
            )
            diagrams = vr.fit_transform(x[None, :, :])[0]
            out = []
            for dim in homology_dimensions:
                dgm = diagrams[diagrams[:, 2] == dim, :2]
                out.append(_clean_diagram(dgm))
            return out
        except Exception:
            return [np.empty((0, 2), dtype=float) for _ in range(maxdim + 1)]


def total_persistence(dgm: np.ndarray, p: float = 1.0) -> float:
    if len(dgm) == 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    return float(np.sum(life ** p))


def persistence_entropy(dgm: np.ndarray) -> float:
    if len(dgm) == 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    total = float(np.sum(life))
    if total <= 0.0:
        return 0.0
    probs = life / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def long_lived_count(dgm: np.ndarray, rel_threshold: float = 0.1) -> int:
    if len(dgm) == 0:
        return 0
    life = dgm[:, 1] - dgm[:, 0]
    threshold = rel_threshold * float(np.max(life))
    return int(np.sum(life > threshold))


def safe_wasserstein(d1: np.ndarray, d2: np.ndarray) -> float:
    try:
        from persim import wasserstein
    except Exception:
        return float("nan")

    def prepare(d):
        if len(d) == 0:
            return np.array([[0.0, 0.0]], dtype=float)
        return d

    try:
        return float(wasserstein(prepare(d1), prepare(d2), matching=False))
    except Exception:
        return float("nan")


def diagram_metrics(
    diagrams: List[np.ndarray], first_diagrams: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2), dtype=float)
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2), dtype=float)
    metrics = {
        "tp_H0": total_persistence(h0, p=1.0),
        "tp_H1": total_persistence(h1, p=1.0),
        "entropy_H0": persistence_entropy(h0),
        "entropy_H1": persistence_entropy(h1),
        "long_H0": long_lived_count(h0),
        "long_H1": long_lived_count(h1),
    }
    if first_diagrams is not None:
        first_h0 = first_diagrams[0] if len(first_diagrams) > 0 else np.empty((0, 2), dtype=float)
        first_h1 = first_diagrams[1] if len(first_diagrams) > 1 else np.empty((0, 2), dtype=float)
        metrics["w1_H0_to_initial"] = safe_wasserstein(h0, first_h0)
        metrics["w1_H1_to_initial"] = safe_wasserstein(h1, first_h1)
    return metrics


def train_one(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    a: torch.Tensor,
    edge_index: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    save_every: int = 10,
    tda_subsample: int = 500,
    pca_dim: int = 10,
    seed: int = 0,
) -> List[Dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask = masks["train"]
    val_mask = masks["val"]
    test_mask = masks["test"]
    history = []
    first_diagrams = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model_forward(model, name, x, a, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, hidden = model_forward(model, name, x, a, edge_index)

        pred = logits.argmax(dim=1)
        train_acc = float((pred[train_mask] == y[train_mask]).float().mean().item())
        val_acc = float((pred[val_mask] == y[val_mask]).float().mean().item())

        if epoch % save_every == 0 or epoch == epochs - 1:
            diagrams = compute_diagrams(
                hidden.cpu().numpy(),
                maxdim=1,
                subsample=tda_subsample,
                pca_dim=pca_dim,
                seed=seed,
            )
            if first_diagrams is None:
                first_diagrams = diagrams
            tda = diagram_metrics(diagrams, first_diagrams)
            record = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            record.update(tda)
            history.append(record)
            print(
                f"[{name}] epoch {epoch:03d} "
                f"loss={loss.item():.4f} "
                f"train_acc={train_acc:.3f} "
                f"val_acc={val_acc:.3f} "
                f"tpH0={tda['tp_H0']:.3f} "
                f"tpH1={tda['tp_H1']:.3f} "
                f"entH1={tda['entropy_H1']:.3f}"
            )

    model.eval()
    with torch.no_grad():
        logits, _ = model_forward(model, name, x, a, edge_index)
    pred = logits.argmax(dim=1)
    test_acc = float((pred[test_mask] == y[test_mask]).float().mean().item())
    if history:
        history[-1]["test_acc"] = test_acc
    return history


def plot_histories(histories: Dict[str, List[Dict]], outdir: str) -> None:
    metrics = [
        ("val_acc", "Validation accuracy"),
        ("tp_H0", "Total persistence H0"),
        ("tp_H1", "Total persistence H1"),
        ("entropy_H1", "Persistence entropy H1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for name, hist in histories.items():
            xs = [h["epoch"] for h in hist]
            ys = [h.get(key, np.nan) for h in hist]
            ys = [np.nan if y is None else y for y in ys]
            ax.plot(xs, ys, marker="o", ms=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "metrics.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MVP: TDA diagnostics for MLP / GCN / simple SNN on posets"
    )
    parser.add_argument("--poset", default="tree", choices=["chain", "tree", "grid"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--stalk_dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--outdir", default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    poset = get_poset(args.poset)
    sources = default_sources(poset)
    x_np, y_np, masks = make_task(poset, sources, args.seed)

    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    a = build_gcn_adjacency(poset.edges, poset.n)
    edge_index = (
        torch.tensor(poset.edges, dtype=torch.long).t().contiguous()
        if len(poset.edges) > 0
        else torch.zeros((2, 0), dtype=torch.long)
    )

    in_dim = x.shape[1]
    out_dim = int(y.max().item() + 1)

    factories = {
        "mlp": lambda: MLP(in_dim, args.hidden_dim, out_dim),
        "gcn": lambda: GCN(in_dim, args.hidden_dim, out_dim),
        "snn": lambda: SimpleSheafNet(
            in_dim,
            args.hidden_dim,
            out_dim,
            stalk_dim=args.stalk_dim,
            alpha=args.alpha,
        ),
    }

    histories = {}
    for name, factory in factories.items():
        print(f"\n=== Training {name} on {args.poset} ===")
        set_seed(args.seed)
        model = factory()
        hist = train_one(
            name=name,
            model=model,
            x=x,
            y=y,
            masks=masks,
            a=a,
            edge_index=edge_index,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_every=args.save_every,
            tda_subsample=500,
            pca_dim=10,
            seed=args.seed,
        )
        histories[name] = hist

    with open(os.path.join(args.outdir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(histories), f, indent=2)

    summary = {name: hist[-1] if hist else {} for name, hist in histories.items()}
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)

    meta = {
        "poset": args.poset,
        "n_nodes": poset.n,
        "sources": sources,
        "args": vars(args),
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(meta), f, indent=2)

    plot_histories(histories, args.outdir)
    print(f"\nDone. Files saved to: {args.outdir}")


if __name__ == "__main__":
    main()