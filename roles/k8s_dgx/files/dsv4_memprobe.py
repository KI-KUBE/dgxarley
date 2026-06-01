"""DSV4 unified-memory load probe (GB10 / SM121).

Activated only when env DSV4_MEMPROBE=1 (sglang_launch.sh sets it from
SGLANG_MEMPROBE). Goal: on GB10 there is ONE unified pool — no host<->device
relocation — so the ~2x footprint AFTER all weight shards are read is a real
in-allocator duplication caused by some specific post-load action. This probe
pinpoints which one.

It logs torch.cuda.memory_allocated()/reserved() + process RSS + node swap:
  - bracketing ModelRunner.load_model (BEGIN = start of load, END = all layers
    read + post-processed) and init_memory_pool / cuda-graph capture,
  - the cuda-alloc DELTA of every Fp8(MoE|Linear)Method.process_weights_after_loading
    call (the prime suspect — per-layer weight repack/requant/contiguous copies),
  - plus a 0.2s background ticker that prints whenever cuda_alloc or RSS jumps
    >0.5 GB, so any unwrapped action that doubles memory is still caught.

All output goes to stderr -> pod log -> Loki. Grep:  |~ "\\[memprobe"
No-op (zero overhead, not even imported side effects) unless DSV4_MEMPROBE=1.
Wired via .pth in sglang_launch.sh; wraps targets lazily as their modules import
(works in the spawned scheduler/TP-worker processes, not just the launcher).
"""

import os
import sys
import threading
import time
from typing import Any

if os.environ.get("DSV4_MEMPROBE", "0") in ("1", "true", "yes"):
    import importlib.abc
    import importlib.util

    def _meminfo() -> tuple[float, float, float, float]:
        a = r = -1.0
        try:
            import torch

            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
        except Exception:
            pass
        rss = swap = -1.0
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) / 1e6
                    break
        except Exception:
            pass
        try:
            mi = {}
            for ln in open("/proc/meminfo"):
                k, v = ln.split(":", 1)
                mi[k] = int(v.split()[0])
            swap = (mi["SwapTotal"] - mi["SwapFree"]) / 1e6
        except Exception:
            pass
        return a, r, rss, swap

    def _emit(tag: str) -> None:
        a, r, rss, sw = _meminfo()
        sys.stderr.write(
            "[memprobe] %-40s cuda_alloc=%7.2fG cuda_resv=%7.2fG rss=%7.2fG node_swap=%7.2fG\n" % (tag, a, r, rss, sw)
        )
        sys.stderr.flush()

    _ticking = threading.Event()

    def _ticker() -> None:
        la = lr = -99.0
        while True:
            a, r, rss, sw = _meminfo()
            if abs(a - la) > 0.5 or abs(rss - lr) > 0.5:
                sys.stderr.write(
                    "[memprobe.tick] cuda_alloc=%7.2fG cuda_resv=%7.2fG rss=%7.2fG node_swap=%7.2fG\n" % (a, r, rss, sw)
                )
                sys.stderr.flush()
                la, lr = a, rss
            time.sleep(0.2)

    def _start_ticker_once() -> None:
        if not _ticking.is_set():
            _ticking.set()
            threading.Thread(target=_ticker, name="memprobe", daemon=True).start()
            _emit("ticker-started")

    def _wrap_bracket(cls: Any, name: str, label: str, start_ticker: bool = False) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__memprobe__", False):
            return

        def w(*a: Any, **k: Any) -> Any:
            if start_ticker:
                _start_ticker_once()
            _emit("BEGIN " + label)
            try:
                return orig(*a, **k)
            finally:
                _emit("END   " + label)

        setattr(w, "__memprobe__", True)
        setattr(cls, name, w)

    def _wrap_delta(cls: Any, name: str, label: str) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__memprobe__", False):
            return

        def w(*a: Any, **k: Any) -> Any:
            a0 = None
            try:
                import torch

                a0 = torch.cuda.memory_allocated()
            except Exception:
                pass
            try:
                return orig(*a, **k)
            finally:
                if a0 is not None:
                    try:
                        import torch

                        d = (torch.cuda.memory_allocated() - a0) / 1e9
                        if abs(d) > 0.2:
                            sys.stderr.write("[memprobe.delta] %-40s d_cuda_alloc=%+7.2fG\n" % (label, d))
                            sys.stderr.flush()
                    except Exception:
                        pass

        setattr(w, "__memprobe__", True)
        setattr(cls, name, w)

    # module -> [(class, method, kind)]; kind: bracket_tick | bracket | delta
    _TARGETS = {
        "sglang.srt.model_executor.model_runner": [
            ("ModelRunner", "load_model", "bracket_tick"),
            ("ModelRunner", "init_memory_pool", "bracket"),
            ("ModelRunner", "init_attention_backend", "bracket"),
        ],
        "sglang.srt.layers.quantization.fp8": [
            ("Fp8MoEMethod", "process_weights_after_loading", "delta"),
            ("Fp8MoEMethod", "process_weights_after_loading_block_quant", "delta"),
            ("Fp8LinearMethod", "process_weights_after_loading", "delta"),
            ("Fp8LinearMethod", "process_weights_after_loading_block_quant", "delta"),
        ],
        "sglang.srt.model_executor.cuda_graph_runner": [
            ("CudaGraphRunner", "capture", "bracket"),
        ],
    }

    def _apply(modname: str, mod: Any) -> None:
        for cls_name, meth, kind in _TARGETS.get(modname, []):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            label = cls_name + "." + meth
            if kind == "bracket_tick":
                _wrap_bracket(cls, meth, label, start_ticker=True)
            elif kind == "bracket":
                _wrap_bracket(cls, meth, label)
            else:
                _wrap_delta(cls, meth, label)
        sys.stderr.write("[memprobe] wrapped %s\n" % modname)
        sys.stderr.flush()

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
            if name not in _TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec and spec.loader:
                real = spec.loader.exec_module

                def exec_module(module: Any, _r: Any = real, _n: str = name) -> None:
                    _r(module)
                    try:
                        _apply(_n, module)
                    except Exception as e:  # noqa: BLE001
                        sys.stderr.write("[memprobe] wrap-failed %s: %s\n" % (_n, e))
                        sys.stderr.flush()

                spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    # modules already imported when the probe loads -> wrap immediately
    for _m in list(_TARGETS):
        if _m in sys.modules:
            try:
                _apply(_m, sys.modules[_m])
            except Exception:
                pass
    sys.meta_path.insert(0, _Hook())
    sys.stderr.write("[memprobe] armed (DSV4_MEMPROBE=1)\n")
    sys.stderr.flush()
