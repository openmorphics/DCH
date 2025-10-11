from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import random
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple

# YAML is imported lazily in _load_config to keep import-safe.


@dataclasses.dataclass
class BaselineConfig:
    seed: int = 0
    device: str = "cpu"
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 16
    input_size: int = 32
    hidden_sizes: List[int] = dataclasses.field(default_factory=lambda: [64])
    num_classes: int = 2
    max_steps: int = 10  # number of time steps in synthetic sequence and number of batches
    log_every: int = 50
    dataset: Literal["synthetic", "nmnist", "dvs_gesture"] = "synthetic"
    data_root: str = "./data"


def _optional_import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def _actionable_dependency_error(*missing: str, dataset: str) -> ImportError:
    missing_list = ", ".join(missing)
    base = (
        f"Missing optional dependency: {missing_list}. "
        "Install with: pip install 'torch>=2.2' 'norse>=0.0.9' "
        "and optionally 'tonic>=1.4.0' for event datasets "
        "(or use conda equivalents)."
    )
    if dataset != "synthetic":
        base += f" You selected dataset='{dataset}', which requires torch and tonic."
    else:
        base += " Or run with --dataset synthetic."
    return ImportError(base)


def _set_deterministic_seed(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    try:
        _np = importlib.import_module("numpy")  # type: ignore
        _np.random.seed(seed)  # type: ignore[attr-defined]
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def build_model(config: BaselineConfig) -> Any:
    torch = _optional_import("torch")
    norse = _optional_import("norse")
    if torch is None or norse is None:
        raise _actionable_dependency_error("torch (>=2.2)", "norse", dataset=config.dataset)
    nn = torch.nn  # type: ignore[attr-defined]
    lif_mod = importlib.import_module("norse.torch.module.lif")
    LIFCell = getattr(lif_mod, "LIFCell")
    # Deterministic init
    _set_deterministic_seed(config.seed, torch=torch)

    class TinySNN(nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            layers: List[Any] = []
            in_dim = config.input_size
            for hs in config.hidden_sizes:
                layers.append(nn.Linear(in_dim, hs))
                # LIFCell operates on (batch, features)
                layers.append(LIFCell())
                in_dim = hs
            self.fc_stack = nn.ModuleList(layers)
            self.readout = nn.Linear(in_dim, config.num_classes)

            # Deterministic initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

        def forward(self, x: Any) -> Any:
            # x shape: (T, B, input_size), binary spikes in float32
            T, B, _ = x.shape
            out_acc = torch.zeros((B, config.num_classes), dtype=torch.float32, device=x.device)
            # keep separate LIF states per LIF layer
            lif_states: List[Any] = [None] * sum(1 for l in self.fc_stack if isinstance(l, LIFCell))
            for t in range(int(T)):
                z = x[t]
                lif_idx = 0
                for layer in self.fc_stack:
                    if isinstance(layer, nn.Linear):
                        z = layer(z)
                    else:
                        z, st = layer(z, lif_states[lif_idx])
                        lif_states[lif_idx] = st
                        lif_idx += 1
                logits = self.readout(z.float())
                out_acc += logits
            return out_acc / float(T)

    return TinySNN()


def _synthetic_batches_torch(config: BaselineConfig, torch: Any) -> Iterator[Tuple[Any, Any]]:
    # Deterministic generator for reproducibility
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)
    T = int(config.max_steps)
    B = int(config.batch_size)
    D = int(config.input_size)
    C = int(config.num_classes)
    # We generate exactly max_steps batches to keep runtime modest.
    num_batches = max(1, min(10, config.max_steps))
    for _ in range(num_batches):
        y = torch.randint(low=0, high=C, size=(B,), generator=gen)
        # Class-dependent spike rates
        rate0 = 0.02
        rate1 = 0.08 if C > 1 else 0.05
        rates = torch.where(y == 0, torch.full((B,), rate0), torch.full((B,), rate1))
        rates = rates.view(1, B, 1).expand(T, B, D)
        x = torch.bernoulli(rates, generator=gen).to(torch.float32)
        yield x, y


def build_data(config: BaselineConfig) -> Iterable[Tuple[Any, Any]]:
    if config.dataset == "synthetic":
        torch = _optional_import("torch")
        if torch is None:
            # Torch-free synthetic fallback: returns Python lists so callers without torch can still inspect shapes
            T = int(config.max_steps)
            B = int(config.batch_size)
            D = int(config.input_size)
            C = int(config.num_classes)
            random.seed(config.seed)
            def _iter() -> Iterator[Tuple[List[List[List[float]]], List[int]]]:
                num_batches = max(1, min(10, config.max_steps))
                for _ in range(num_batches):
                    labels = [random.randrange(C) for _ in range(B)]
                    x = [[[1.0 if random.random() < (0.02 if labels[b] == 0 else 0.08) else 0.0 for _ in range(D)]
                          for b in range(B)] for ___ in range(T)]
                    yield x, labels
            return _iter()
        else:
            return _synthetic_batches_torch(config, torch)
    else:
        torch = _optional_import("torch")
        tonic = _optional_import("tonic")
        if torch is None or tonic is None:
            raise _actionable_dependency_error("torch (>=2.2)", "tonic (>=1.4.0)", dataset=config.dataset)
        raise NotImplementedError(
            f"Dataset '{config.dataset}' loading is intentionally not implemented in this minimal baseline. "
            "Use --dataset synthetic for a quick run."
        )


def train_and_eval(config: BaselineConfig) -> Dict[str, Any]:
    start = time.time()
    torch = _optional_import("torch")
    norse = _optional_import("norse")
    if torch is None or norse is None:
        raise _actionable_dependency_error("torch (>=2.2)", "norse", dataset=config.dataset)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    _set_deterministic_seed(config.seed, torch=torch)
    model = build_model(config).to(device)
    nn = torch.nn  # type: ignore
    optim = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    ce = nn.CrossEntropyLoss()

    # Train (tiny)
    model.train()
    seen: int = 0
    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0
    for _epoch in range(int(config.epochs)):
        for step, (x, y) in enumerate(build_data(config)):
            if not isinstance(x, torch.Tensor):
                raise _actionable_dependency_error("torch (>=2.2)", "norse", dataset=config.dataset)
            # x: (T,B,D)
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                total_correct += int((preds == y).sum().item())
                total_examples += int(y.shape[0])
            seen += 1
            if seen >= max(1, min(10, config.max_steps)):
                break  # keep it fast

    # Eval on one small pass
    model.eval()
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(build_data(config)):
            if not isinstance(x, torch.Tensor):
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            eval_correct += int((preds == y).sum().item())
            eval_total += int(y.shape[0])
            if step + 1 >= 2:
                break

    elapsed = time.time() - start
    result: Dict[str, Any] = {
        "baseline": "norse_sg",
        "dataset": config.dataset,
        "loss": total_loss / max(1, seen),
        "accuracy": (eval_correct / eval_total) if eval_total > 0 else None,
        "elapsed_s": elapsed,
        "seed": config.seed,
    }
    # Drop None-valued fields to match spec
    for k in [k for k, v in list(result.items()) if v is None]:
        del result[k]
    return result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Norse SG Baseline (tiny, deterministic)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--hidden-sizes", type=str, default=None, help="Comma-separated, e.g. 64,32")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["synthetic", "nmnist", "dvs_gesture"])
    p.add_argument("--data-root", type=str, default=None)
    return p.parse_args(argv)


def _load_config(ns: argparse.Namespace) -> BaselineConfig:
    cfg = BaselineConfig()
    if ns.config:
        try:
            yaml = importlib.import_module("yaml")
        except Exception:
            raise ImportError("PyYAML missing. Install with: pip install pyyaml")
        with open(ns.config, "r", encoding="utf-8") as f:
            data = getattr(yaml, "safe_load")(f) or {}
        cfg = BaselineConfig(**{**dataclasses.asdict(cfg), **data})
    # CLI overrides with special handling for hidden_sizes
    overrides: Dict[str, Any] = {}
    if getattr(ns, "hidden_sizes", None) is not None:
        hs = str(getattr(ns, "hidden_sizes"))
        overrides["hidden_sizes"] = [int(x) for x in hs.split(",") if x]
    for field in dataclasses.fields(BaselineConfig):
        key = field.name
        if key == "hidden_sizes":
            continue
        if hasattr(ns, key) and getattr(ns, key) is not None:
            overrides[key] = getattr(ns, key)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    ns = _parse_args(argv)
    try:
        config = _load_config(ns)
        result = train_and_eval(config)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    except (ImportError, NotImplementedError) as e:
        sys.stderr.write(str(e) + "\n")
        sys.stderr.flush()
        raise SystemExit(2)
    except Exception as e:  # unexpected errors: non-2 exit
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.stderr.flush()
        raise

if __name__ == "__main__":  # pragma: no cover
    main()
