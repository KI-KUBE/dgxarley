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
import traceback
from typing import Any

if os.environ.get("DSV4_MEMPROBE", "0") in ("1", "true", "yes"):
    import importlib.abc
    import importlib.util

    # Capture BOTH layers — we do not yet know whether the swapped ~50 GB is the
    # CUDA/unified weights or transient HOST allocations:
    #   - HOST native+mmap: memray.Tracker around load_model (if installed by
    #     sglang_launch.sh) → which code malloc/mmap'd the transient. Output
    #     /tmp/dsv4_memray_<pid>.bin → analyse with `memray stats|flamegraph`.
    #   - CUDA/device: torch.cuda.memory._record_memory_history() + _dump_snapshot
    #     → /tmp/dsv4_cuda_<pid>.pickle (per-allocation stacks + timeline).
    #   - WHAT swaps: per-mapping Swap from /proc/self/smaps when swap is high.
    try:
        import memray  # type: ignore

        _HAVE_MEMRAY = True
    except Exception:
        _HAVE_MEMRAY = False

    # Full snapshot to find the swap-out TRIGGER: which memory category grows
    # while anon swaps out. torch's allocator view (cuda_alloc) is NOT enough on
    # unified memory — NCCL/driver/other allocations show only in the device
    # mem_get_info(used) and in the /proc/meminfo categories. /proc/vmstat gives
    # the real swap-out + reclaim counters (diff across ticks for rates). Values
    # in GB except *_ctr (raw cumulative page counters).
    def _snap() -> "dict[str, float]":
        d: "dict[str, float]" = {}
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            d["cuda_alloc"] = torch.cuda.memory_allocated() / 1e9
            d["cuda_resv"] = torch.cuda.memory_reserved() / 1e9
            d["dev_used"] = (total - free) / 1e9  # whole-device used (incl NCCL/driver/other)
            d["dev_free"] = free / 1e9
        except Exception:
            d.update(cuda_alloc=-1.0, cuda_resv=-1.0, dev_used=-1.0, dev_free=-1.0)
        try:
            mi = {}
            for ln in open("/proc/meminfo"):
                k, v = ln.split(":", 1)
                mi[k] = int(v.split()[0])  # kB

            def gb(key: str) -> float:
                return mi.get(key, 0) / 1024.0 / 1024.0

            d.update(
                node_used=gb("MemTotal")
                - gb("MemFree"),  # whole-node used (==dev_used on GB10 unified → confirms node-wide)
                memfree=gb("MemFree"),
                cached=gb("Cached"),
                anon=gb("AnonPages"),
                mapped=gb("Mapped"),
                shmem=gb("Shmem"),
                srecl=gb("SReclaimable"),
                sunrecl=gb("SUnreclaim"),
                pagetbl=gb("PageTables"),
                swapused=(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1024.0 / 1024.0,
            )
        except Exception:
            pass
        try:
            vs = {}
            for ln in open("/proc/vmstat"):
                p = ln.split()
                if len(p) == 2:
                    vs[p[0]] = int(p[1])
            d["pswpout_ctr"] = vs.get("pswpout", 0)
            d["pgst_kswapd_ctr"] = vs.get("pgsteal_kswapd", 0)
            d["pgst_direct_ctr"] = vs.get("pgsteal_direct", 0)
        except Exception:
            pass
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmSwap:"):
                    d["self_vmswap"] = int(ln.split()[1]) / 1024.0 / 1024.0
                    break
        except Exception:
            pass
        return d

    _ORDER = [
        "cuda_alloc",
        "cuda_resv",
        "dev_used",
        "node_used",
        "dev_free",
        "memfree",
        "cached",
        "anon",
        "mapped",
        "shmem",
        "srecl",
        "sunrecl",
        "pagetbl",
        "swapused",
        "self_vmswap",
        "pswpout_ctr",
        "pgst_kswapd_ctr",
        "pgst_direct_ctr",
    ]

    def _fmt(d: "dict[str, float]") -> str:
        out = []
        for k in _ORDER:
            v = d.get(k)
            if v is None:
                continue
            out.append(("%s=%d" % (k, v)) if k.endswith("_ctr") else ("%s=%.2f" % (k, v)))
        return " ".join(out)

    def _emit(tag: str) -> None:
        sys.stderr.write("[memprobe] %-40s %s\n" % (tag, _fmt(_snap())))
        sys.stderr.flush()

    # Sampling profiler: dump the live Python stack of every worker thread. This
    # is the whole point — during the memory-explode phase sglang logs NOTHING,
    # so we sample sys._current_frames() ourselves. The most frequent stack over
    # the phase = the operation driving memory. A thread parked in a C call
    # (NCCL all_reduce, cudaMalloc, a torch op) shows the Python frame that
    # entered it, which still names the responsible action.
    def _stacks(maxframes: int = 9) -> None:
        me = threading.get_ident()
        try:
            main_id = threading.main_thread().ident
        except Exception:
            main_id = None
        for tid, frame in list(sys._current_frames().items()):
            if tid == me:  # don't profile the profiler
                continue
            try:
                st = traceback.extract_stack(frame)[-maxframes:]
                chain = " <- ".join(
                    "%s:%d:%s" % (f.filename.rsplit("/", 1)[-1], f.lineno or 0, f.name) for f in reversed(st)
                )
            except Exception:
                continue
            tag = "MAIN" if tid == main_id else ("tid=%d" % tid)
            sys.stderr.write("[memprobe.stack] %-8s %s\n" % (tag, chain))
        sys.stderr.flush()

    def _self_vmswap_gb() -> float:
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmSwap:"):
                    return int(ln.split()[1]) / 1024.0 / 1024.0
        except Exception:
            pass
        return 0.0

    def _swap_smaps() -> None:
        # Per-mapping Swap breakdown — answers WHAT is swapped (host-anon vs
        # safetensors-mmap vs /dev/nvidia vs libs), independent of any guess.
        agg: "dict[str, float]" = {}
        cur = "[anon]"
        try:
            for ln in open("/proc/self/smaps"):
                if ln and ln[0] != " " and "-" in ln.split(" ", 1)[0]:
                    parts = ln.split()
                    nm = parts[5] if len(parts) > 5 else "[anon]"
                    if nm.startswith("/dev/nvidia"):
                        nm = "/dev/nvidia*"
                    elif nm.endswith(".so") or ".so." in nm:
                        nm = "shared-libs"
                    elif "huggingface" in nm or nm.endswith(".safetensors"):
                        nm = "safetensors-mmap"
                    elif nm and nm[0] != "[" and "/" in nm:
                        nm = nm.rsplit("/", 1)[-1]
                    cur = nm or "[anon]"
                elif ln.startswith("Swap:"):
                    agg[cur] = agg.get(cur, 0.0) + int(ln.split()[1]) / 1024.0 / 1024.0
            top = sorted(agg.items(), key=lambda x: -x[1])[:10]
            sys.stderr.write("[memprobe.swapmap] " + " ".join("%s=%.1fG" % (k, v) for k, v in top if v > 0.1) + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _cuda_record() -> None:
        try:
            import torch

            torch.cuda.memory._record_memory_history(max_entries=200000)
            sys.stderr.write("[memprobe] cuda memory-history recording ON\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[memprobe] cuda record failed: %s\n" % e)
        sys.stderr.flush()

    def _cuda_dump() -> None:
        try:
            import torch

            p = "/tmp/dsv4_cuda_%d.pickle" % os.getpid()
            torch.cuda.memory._dump_snapshot(p)  # type: ignore[no-untyped-call]
            sys.stderr.write("[memprobe] cuda snapshot dumped: %s\n" % p)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("[memprobe] cuda dump failed: %s\n" % e)
        sys.stderr.flush()

    _ticking = threading.Event()

    def _ticker() -> None:
        # Detailed memory line + a stack sample every ~1.5s (the whole silent
        # NCCL/post-load phase trajectory lands in Loki), plus immediately on a
        # >0.5G cuda or >1G swap move. One-shot smaps swap-breakdown when the
        # process's own swap first exceeds 15G, and one-shot CUDA snapshot once
        # cuda_alloc stabilises high (post-KV). Runs only in the loading process.
        n = 0
        lca = lsw = -99.0
        swapdumped = cudadumped = False
        while True:
            d = _snap()
            n += 1
            ca = d.get("cuda_alloc", -1.0)
            sw = d.get("swapused", -1.0)
            vmsw = _self_vmswap_gb()
            if n % 3 == 0 or abs(ca - lca) > 0.5 or abs(sw - lsw) > 1.0:
                sys.stderr.write("[memprobe.tick] %s self_vmswap=%.2f\n" % (_fmt(d), vmsw))
                _stacks()
                lca, lsw = ca, sw
            if not swapdumped and vmsw > 15.0:
                _swap_smaps()
                swapdumped = True
            if not cudadumped and ca >= 90.0:
                _cuda_dump()
                cudadumped = True
            time.sleep(0.5)

    def _start_ticker_once() -> None:
        if not _ticking.is_set():
            _ticking.set()
            _cuda_record()
            threading.Thread(target=_ticker, name="memprobe", daemon=True).start()
            _emit("ticker-started")

    def _wrap_bracket(cls: Any, name: str, label: str, start_ticker: bool = False) -> None:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__memprobe__", False):
            return

        def w(*a: Any, **k: Any) -> Any:
            if not start_ticker:
                _emit("BEGIN " + label)
                try:
                    return orig(*a, **k)
                finally:
                    _emit("END   " + label)
            # load_model: start ticker + CUDA recording, and wrap the whole load
            # (incl. the post-load monitored_barrier) in a memray Tracker so the
            # HOST native+mmap high-water-mark is attributed to its call stack.
            _start_ticker_once()
            _emit("BEGIN " + label)
            tracker = None
            if _HAVE_MEMRAY:
                try:
                    dest = "/tmp/dsv4_memray_%d.bin" % os.getpid()
                    tracker = memray.Tracker(dest, native_traces=False)
                    tracker.__enter__()
                    sys.stderr.write("[memprobe] memray Tracker ON: %s\n" % dest)
                    sys.stderr.flush()
                except Exception as e:  # noqa: BLE001
                    tracker = None
                    sys.stderr.write("[memprobe] memray start failed: %s\n" % e)
                    sys.stderr.flush()
            try:
                return orig(*a, **k)
            finally:
                if tracker is not None:
                    try:
                        tracker.__exit__(None, None, None)
                        sys.stderr.write("[memprobe] memray Tracker written\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
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
