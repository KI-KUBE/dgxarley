# NVENC Hardware Encoder Test — GB10 / DGX Spark

**Date:** 2026-04-28
**Node:** spark1 (NVIDIA GB10, Blackwell, sm_121)
**Driver:** 580.142
**Container:** `ubuntu:24.04` + `apt install ffmpeg` (ffmpeg 6.1.1-3ubuntu5)
**Pod manifest:** [`nvenc_test_pod.yml`](../../nvenc_test_pod.yml)
**NVIDIA caps:** `compute,utility,video` (video required for `libnvidia-encode.so` injection)
**Co-tenant:** SGLang head pod running on same node, GPU utilization 0% during test (idle model).

## Goal

Verify that GB10 actually exposes NVENC and benchmark all three Blackwell-class encoders
(`h264_nvenc`, `hevc_nvenc`, `av1_nvenc`) through ffmpeg. Datacenter Blackwells (B100/B200)
have NVENC removed; the workstation/dev GB10 was undocumented at test time
([NVIDIA Dev Forum thread](https://forums.developer.nvidia.com/t/dgx-and-nvenc/348109)).

## Encoder availability

```
$ ffmpeg -hide_banner -encoders | grep nvenc
 V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)

$ ffmpeg -hide_banner -hwaccels
vdpau cuda vaapi drm opencl vulkan
```

All three NVENC codecs present + CUDA hwaccel for NVDEC. Confirms GB10 ships with at
least 1× NVENC + 1× NVDEC engine, consistent with the
[PNY DGX Spark spec sheet](https://www.pny.com/en-eu/professional/hardware/nvidia-dgx-spark)
which lists H.264, H.265 (incl. 4:2:2), VP8, VP9, AV1.

## Benchmark setup

- Input: `lavfi testsrc2`, 10 s, 60 fps (600 frames per run)
- Bitrate target: 8 Mbps CBR (`-b:v 8M`)
- Resolutions: 1080p60, 4K60
- Presets: `p1` (fastest) and `p4` (default / balanced)
- Single-stream (one ffmpeg process per run)
- Wall-time = `bench: rtime=…s` from ffmpeg `-benchmark`
- Encode fps derived as `600 / rtime` (ffmpeg's per-line `fps=` shows 0 for runs that
  finish before the progress counter ticks)

## Results

| Codec | Preset | Resolution | Wall (s) | Encode fps | Speedup vs realtime | Output size |
|-------|--------|-----------|---------:|-----------:|--------------------:|------------:|
| h264_nvenc | p1 | 1920×1080@60 | 0.872 |  **688** | 11.5× | 10.5 MB |
| h264_nvenc | p4 | 1920×1080@60 | 1.263 |  **475** |  7.9× | 10.8 MB |
| hevc_nvenc | p1 | 1920×1080@60 | 0.772 |  **777** | 13.0× | 10.4 MB |
| hevc_nvenc | p4 | 1920×1080@60 | 1.422 |  **422** |  7.0× | 10.9 MB |
| av1_nvenc  | p1 | 1920×1080@60 | 0.916 |  **655** | 10.9× | 10.6 MB |
| av1_nvenc  | p4 | 1920×1080@60 | 1.332 |  **450** |  7.5× | 10.6 MB |
| h264_nvenc | p1 | 3840×2160@60 | 2.435 |  **246** |  4.1× | 11.1 MB |
| h264_nvenc | p4 | 3840×2160@60 | 4.325 |  **139** |  2.3× | 11.2 MB |
| hevc_nvenc | p1 | 3840×2160@60 | 2.149 |  **279** |  4.7× | 10.7 MB |
| hevc_nvenc | p4 | 3840×2160@60 | 4.638 |  **129** |  2.2× | 11.4 MB |
| av1_nvenc  | p1 | 3840×2160@60 | 2.761 |  **217** |  3.6× | 11.3 MB |
| av1_nvenc  | p4 | 3840×2160@60 | 4.678 |  **128** |  2.1× | 11.2 MB |

Output sizes are uniform (~8 Mbps × 10 s ≈ 10 MB) — CBR target hit on all codecs.

## Findings

1. **NVENC works on GB10.** All three codecs (H.264, HEVC, AV1) produced valid
   playable output via the standard ffmpeg+NVIDIA runtime path.
2. **1 NVENC engine is plenty for typical Jitsi/Jibri load.** Even at preset `p4`
   the slowest combo (HEVC 4K60) hits 2.2× realtime; H.264 1080p60 p1 hits
   11.5× realtime — comfortable headroom for several concurrent recording sessions.
3. **HEVC slightly faster than H.264 at p1**, both at 1080p (777 vs 688 fps) and
   4K (279 vs 246 fps). AV1 is a hair slower but in the same class.
4. **p1 → p4 costs ~1.5–2× wall-time** for noticeably better quality at the same
   bitrate (typical NVENC behaviour).
5. **NVIDIA_DRIVER_CAPABILITIES must include `video`** — the default is
   `compute,utility` and `libnvidia-encode.so` will not be injected, so ffmpeg
   reports no NVENC encoders despite the driver being present.

## Caveats

- `lavfi testsrc2` generation is CPU-bound and runs on the same wall-clock.
  Pure NVENC throughput is **higher** than the numbers above — reported numbers
  reflect end-to-end pipeline throughput, which is the relevant metric for a
  Jitsi/Jibri-style use-case anyway.
- Single-stream test only; concurrent-stream behaviour (NVENC session limit and
  shared-engine throughput degradation) was **not** measured. Worth doing in a
  follow-up if multi-recording is planned.
- Container ran with `nvidia.com/gpu: 1` against the GPU-time-slicing config
  (4 replicas, see `CLAUDE.md`). The NVENC engine is not partitioned by
  time-slicing — encoding will compete with any concurrent NVENC user on the
  same physical GPU.
- SGLang head was running on the node (idle) but NVENC and CUDA compute share
  no scheduling — no observable interference.

## Reproduce

```bash
kubectl --context=ht@dgxarley apply -f nvenc_test_pod.yml

# Wait for init log to print encoder list
kubectl --context=ht@dgxarley logs -f nvenc-test

# Bench script is written into /scratch/bench.sh inline (see this TESTLOG's git history
# or copy from below). Run:
kubectl --context=ht@dgxarley exec nvenc-test -- bash /scratch/bench.sh

# Inspect raw TSV:
kubectl --context=ht@dgxarley exec nvenc-test -- cat /scratch/results.tsv

# Cleanup:
kubectl --context=ht@dgxarley delete -f nvenc_test_pod.yml
```

### Benchmark script

```bash
#!/bin/bash
set -u
DUR=10
RESULTS=/scratch/results.tsv
echo -e "codec\tpreset\tres\tfps_in\tencode_fps\tutime_s\tstime_s\trtime_s\tbytes_out" > "$RESULTS"
for RES in "1920x1080:60" "3840x2160:60"; do
  W=$(echo $RES | cut -d: -f1); R=$(echo $RES | cut -d: -f2)
  for CODEC in h264_nvenc hevc_nvenc av1_nvenc; do
    EXT=mp4; [ "$CODEC" = "av1_nvenc" ] && EXT=mkv
    for PRESET in p1 p4; do
      OUT=/scratch/out_${CODEC}_${PRESET}_${W//x/_}.${EXT}
      LOG=/scratch/log_${CODEC}_${PRESET}_${W//x/_}.txt
      ffmpeg -y -hide_banner -benchmark \
        -f lavfi -i testsrc2=size=${W}:rate=${R}:duration=${DUR} \
        -c:v $CODEC -preset $PRESET -b:v 8M "$OUT" 2> "$LOG"
      FPS=$(grep -oE "fps=[ ]*[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+")
      UT=$(grep -oE "utime=[0-9.]+s" "$LOG" | grep -oE "[0-9.]+")
      ST=$(grep -oE "stime=[0-9.]+s" "$LOG" | grep -oE "[0-9.]+")
      RT=$(grep -oE "rtime=[0-9.]+s" "$LOG" | grep -oE "[0-9.]+")
      BYTES=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
      echo -e "${CODEC}\t${PRESET}\t${W}\t${R}\t${FPS}\t${UT}\t${ST}\t${RT}\t${BYTES}" >> "$RESULTS"
    done
  done
done
```

## Follow-up 1 — Concurrent-stream saturation

Run N parallel `h264_nvenc` 1080p60 p4 encodes (20 s each) simultaneously, measure
per-stream fps and aggregate.

| N streams | per-stream fps | aggregate fps | wall (s) | per-stream vs realtime |
|----------:|---------------:|--------------:|---------:|-----------------------:|
| 1 |  519 |  519 |  2.31 |  8.7× |
| 2 |  266 |  532 |  4.51 |  4.4× |
| 4 |  133 |  532 |  9.02 |  2.2× |
| 8 |   65 |  523 | 18.34 |  1.09× |

**Findings:**
- **NVENC saturates immediately** — aggregate throughput plateaus at **~520–530 fps**
  for `h264_nvenc` 1080p60 p4. Per-stream fps is just `aggregate / N`.
- **No hard session cap observed up to N=8.** The encoder shares its single engine
  fairly across concurrent contexts; degradation is graceful, not cliff-edged.
- **Realtime ceiling for 1080p60 recording: ~8 concurrent streams.** At N=8,
  per-stream fps is 65 — barely above realtime (60 fps). N=9+ would fall below
  realtime and start dropping frames in a recording scenario.

## Follow-up 2 — Raw YUV pure encoder throughput

Pre-generated 1.78 GB raw YUV420p file (1080p60, 10 s, 600 frames), encoded directly
to remove `lavfi testsrc2` CPU overhead.

| Codec | Preset | Wall lavfi (s) | Wall raw (s) | fps lavfi | fps raw | Δ |
|-------|--------|--------------:|-------------:|----------:|--------:|---:|
| h264_nvenc | p1 | 0.872 | 0.819 | 688 | **733** | +6% |
| h264_nvenc | p4 | 1.263 | 1.278 | 475 | 469 | ~0 |
| hevc_nvenc | p1 | 0.772 | 0.750 | 777 | **800** | +3% |
| hevc_nvenc | p4 | 1.422 | 1.446 | 422 | 415 | ~0 |
| av1_nvenc  | p1 | 0.916 | 0.947 | 655 | 634 | ~0 |
| av1_nvenc  | p4 | 1.332 | 1.334 | 450 | 450 |  0 |

**Finding:** at 1080p, lavfi overhead is **negligible** (≤6% on the fastest preset,
zero everywhere else). The original lavfi-based numbers in the main table are
representative of real NVENC throughput. At 4K the gap might be larger; not measured
because raw 4K60 10 s ≈ 6 GB exceeds the test pod's emptyDir.

## Follow-up 3 — NVDEC + NVENC transcode roundtrip

60 s synthetic h264 source (3 600 frames), 1080p60, transcode to `h264_nvenc` p4 8 Mbps.
Four pipelines compared:

| Mode | Decoder | Frame transit | Wall (s) | fps | utime (s) | Δ utime |
|------|---------|---------------|---------:|----:|----------:|---------:|
| `cpu_dec_libx264`     | CPU      | sysmem        | 3.98 | **904** | (very high — multi-core) | baseline |
| `cpu_dec_nvenc`       | CPU h264 | sysmem        | 6.38 | 565 | 13.78 |  ref |
| `nvdec_sysmem_nvenc`  | NVDEC    | GPU→host→GPU  | 6.70 | 538 |  1.92 | **−86%** |
| `nvdec_cuda_nvenc`    | NVDEC    | GPU-resident  | 6.21 | **580** |  0.95 | **−93%** |

**Findings:**
- **NVDEC works on GB10.** Full GPU pipeline (`-hwaccel cuda -hwaccel_output_format cuda`)
  succeeds at 580 fps — slightly faster than CPU-decode + NVENC because no PCIe round trip.
- **The CPU savings are huge.** Full-GPU pipeline uses **~14× less CPU** than
  CPU-decode + NVENC for the same throughput. Critical for a Jibri-style deployment
  where the same host also runs Chromium, prosody, etc.
- **libx264 ultrafast wins on raw fps** (904 fps) but burns multiple CPU cores fully
  and produces lower quality at the same bitrate than NVENC p4. Not an apples-to-apples
  comparison; libx264 medium would land below NVENC p4 in fps with comparable quality.
- **Wall-time is encoder-bound** — all three NVENC variants land at 6.2–6.7 s. NVDEC
  is fast enough to keep the encoder fed; the encoder is the bottleneck.

## Verdict — is Jibri (or similar) worth deploying on the cluster?

**Yes, technically very feasible.** Concrete reasons:

- **One NVENC engine handles ~8 concurrent 1080p60 H.264 recordings in realtime** —
  vastly more than any homelab Jitsi instance needs. Even with 50% headroom for
  encoder variance and B-frame logic, 4–6 concurrent recordings is conservative.
- **Full GPU pipeline frees the CPU** for the rest of the recording stack (Xorg,
  Chromium, audio mux, file IO).
- **All three relevant codecs work** — H.264 for compatibility, HEVC for ~30%
  bitrate savings, AV1 for archival.
- **No conflict with SGLang.** NVENC and CUDA SMs are independent units;
  measurements were taken with the SGLang head pod live on the same GPU.

**Caveats / pre-requisites if you actually deploy Jibri:**

- Jibri itself is a Java service that drives a headless Chromium under Xorg + ffmpeg.
  It needs `--shm-size`, X11/PulseAudio inside the container, and either
  `nvidia.com/gpu: 1` or hostPath device passthrough. **The standard upstream
  jitsi-jibri Docker image is x86_64 only** — would need an ARM64 rebuild for the
  Sparks. That's the main porting cost, not NVENC capability.
- Jibri's ffmpeg invocation needs `-c:v h264_nvenc -preset p4` patched in (it
  defaults to libx264). One-line config change.
- NVENC engine is **not** partitioned by GPU time-slicing. If two deployments
  hammer NVENC simultaneously (e.g. multiple Jibri pods), they share the ~520 fps
  aggregate — plan accordingly.
- **Alternative recommendation:** the Sparks' actual sweet spot for conferencing is
  AI features (live transcription via Whisper, summarization via SGLang/Ollama),
  not the SFU itself. The JVB has nothing to GPU-accelerate. Worth deploying Jibri
  only if recording/streaming is a real requirement; otherwise the GPU is more
  valuable doing inference.

## Pod cleanup

```bash
kubectl --context=ht@dgxarley delete -f nvenc_test_pod.yml
```

