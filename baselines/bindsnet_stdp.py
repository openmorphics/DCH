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
    # Unsupervised STDP params
    timesteps: int = 25
    dt: float = 1.0
    stdp_lr: float = 0.01
    batch_size: int = 16
    input_size: int = 64
    num_exc_neurons: int = 32
    epochs: int = 1
    max_steps: int = 10
    dataset: Literal["synthetic", "nmnist", "dvs_gesture"] = "synthetic"
    data_root: str = "./data"


def _optional_import(m: str) -> Any:
    try:
        return importlib.import_module(m)
    except ImportError:
        return None


def _actionable_dependency_error(*missing: str, dataset: str) -> ImportError:
    missing_list = ", ".join(missing)
    base = (
        f"Missing optional dependency: {missing_list}. "
        "Install with: pip install 'torch>=2.2' 'bindsnet>=0.3' "
        "and optionally 'tonic>=1.4.0' for event datasets."
    )
    if dataset != "synthetic":
        base += f" You selected dataset='{dataset}', which requires torch and tonic."
    else:
        base += " Or run with --dataset synthetic."
    return ImportError(base)


def _set_seed(seed: int, torch: Any | None) -> None:
    random.seed(seed)
    try:
        _np = importlib.import_module("numpy")  # type: ignore
        _np.random.seed(seed)  # type: ignore[attr-defined]
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_network(config: BaselineConfig) -> Any:
    torch = _optional_import("torch")
    bn = _optional_import("bindsnet")
    if torch is None or bn is None:
        raise _actionable_dependency_error("torch (>=2.2)", "bindsnet (>=0.3)", dataset=config.dataset)
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection
    from bindsnet.learning import PostPre

    _set_seed(config.seed, torch=torch)
    net = Network(dt=config.dt)
    in_layer = Input(n=int(config.input_size), traces=True)
    exc = LIFNodes(n=int(config.num_exc_neurons), traces=True)
    net.add_layer(in_layer, name="X")
    net.add_layer(exc, name="E")
    # Initialize small weights deterministically
    w = torch.full((int(config.input_size), int(config.num_exc_neurons)), 0.02, dtype=torch.float32)
    conn = Connection(source=in_layer, target=exc, w=w, update_rule=PostPre, nu=config.stdp_lr)
    net.add_connection(conn, source="X", target="E")
    return net


def _synthetic_batches_torch(config: BaselineConfig, torch: Any) -> Iterator[Tuple[Any, Any]]:
    # deterministic synthetic Poisson spikes conditioned on labels
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    T = int(config.timesteps)
    B = int(config.batch_size)
    D = int(config.input_size)
    C = 2  # proxy 2 clusters for purity notion
    num_batches = max(1, min(10, config.max_steps))
    for _ in range(num_batches):
        y = torch.randint(0, C, (B,), generator=gen)
        rate0 = 5.0  # Hz-ish units (proxy)
        rate1 = 15.0
        # convert rates to spike probability per dt assuming small dt
        p0 = min(0.2, rate0 * (config.dt / 1000.0) + 0.02)
        p1 = min(0.2, rate1 * (config.dt / 1000.0) + 0.05)
        probs = torch.where(y == 0, torch.full((B,), p0), torch.full((B,), p1)).view(1, B, 1).expand(T, B, D)
        x = torch.bernoulli(probs, generator=gen).to(torch.float32)
        yield x, y


def build_data(config: BaselineConfig) -> Iterable[Tuple[Any, Any]]:
    if config.dataset == "synthetic":
        torch = _optional_import("torch")
        if torch is None:
            # torch-free generator of lists (fallback)
            T = int(config.timesteps)
            B = int(config.batch_size)
            D = int(config.input_size)
            random.seed(config.seed)

            def _iter() -> Iterator[Tuple[List[List[List[float]]], List[int]]]:
                num_batches = max(1, min(10, config.max_steps))
                for _ in range(num_batches):
                    labels = [random.randrange(2) for _ in range(B)]
                    p = [0.05 if lbl == 0 else 0.15 for lbl in labels]
                    x = [[[1.0 if random.random() < p[b] else 0.0 for _ in range(D)] for b in range(B)] for _ in range(T)]
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


def run_unsupervised(config: BaselineConfig) -> Dict[str, Any]:
    start = time.time()
    torch = _optional_import("torch")
    bn = _optional_import("bindsnet")
    if torch is None or bn is None:
        raise _actionable_dependency_error("torch (>=2.2)", "bindsnet (>=0.3)", dataset=config.dataset)

    net = build_network(config)
    from bindsnet.network.monitors import Monitor

    # Monitor spikes of excitatory population
    exc = net.get_layer("E")
    mon = Monitor(obj=exc, state_vars=["s"], time=int(config.timesteps))
    net.add_monitor(mon, name="E_spikes")

    # tiny unsupervised loop
    total_rate_c0 = 0.0
    total_rate_c1 = 0.0
    total_c0 = 0
    total_c1 = 0

    for _epoch in range(int(config.epochs)):
        for step, (x, y) in enumerate(build_data(config)):
            if not isinstance(x, torch.Tensor):
                # cannot run without torch
                raise _actionable_dependency_error("torch (>=2.2)", "bindsnet (>=0.3)", dataset=config.dataset)
            net.reset_state_variables()
            # Run for timesteps with one-step inputs
            for t in range(int(config.timesteps)):
                net.run(inputs={"X": x[t]}, time=1, input_time_dim=False, learning=True)
            # Aggregate spikes
            s = mon.get("s")  # shape: time x batch x neurons
            # Compute average spike rate per sample
            rates = s.sum(dim=0).mean(dim=1)  # batch
            # Accumulate by class
            mask0 = (y == 0)
            mask1 = (y == 1)
            if mask0.any():
                total_rate_c0 += float(rates[mask0].mean().item())
                total_c0 += int(mask0.sum().item())
            if mask1.any():
                total_rate_c1 += float(rates[mask1].mean().item())
                total_c1 += int(mask1.sum().item())
            if step + 1 >= max(1, min(5, config.max_steps)):
                break

    # Proxy separation score (higher is better)
    purity_proxy: Optional[float] = None
    if total_c0 > 0 and total_c1 > 0:
        avg0 = total_rate_c0 / total_c0
        avg1 = total_rate_c1 / total_c1
        denom = max(1e-6, (avg0 + avg1))
        purity_proxy = abs(avg1 - avg0) / denom

    elapsed = time.time() - start
    result: Dict[str, Any] = {
        "baseline": "bindsnet_stdp",
        "dataset": config.dataset,
        "elapsed_s": elapsed,
        "seed": config.seed,
    }
    # Only include documented fields. Optionally map proxy to "accuracy" if it exists.
    if purity_proxy is not None:
        result["accuracy"] = purity_proxy  # proxy "purity-like" score
    return result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BindsNET STDP Baseline (tiny, deterministic)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--stdp-lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--num-exc-neurons", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
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
    # CLI overrides
    mapping = {
        "stdp-lr": "stdp_lr",
        "batch-size": "batch_size",
        "input-size": "input_size",
        "num-exc-neurons": "num_exc_neurons",
        "max-steps": "max_steps",
        "timesteps": "timesteps",
        "data-root": "data_root",
    }
    for cli_name, attr in mapping.items():
        key = cli_name.replace("-", "_")
        if hasattr(ns, key) and getattr(ns, key) is not None:
            setattr(cfg, attr, getattr(ns, key))
    for simple in ["seed", "device", "dt", "epochs", "dataset"]:
        if hasattr(ns, simple) and getattr(ns, simple) is not None:
            setattr(cfg, simple, getattr(ns, simple))
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    ns = _parse_args(argv)
    try:
        config = _load_config(ns)
        result = run_unsupervised(config)
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
