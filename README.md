# neural_audio: podcasts as Mimi, with transcripts and ad skipping

`AGENTS.md` is a short guide to the repository and to setting up a server (at home, on Modal, or both), for AI
coding agents and people.

A podcast pipeline. A server (**mimi-hub**) re-encodes episodes with Kyutai's
[Mimi](https://huggingface.co/kyutai/mimi) neural codec (32 codebooks, 4.4 kb/s, about 2 MB per hour), transcribes
them with speaker labels (Parakeet TDT 0.6B v2 + pyannote) and marks ads and self-promotion with an LLM. A patched
[Pocket Casts for Android](https://github.com/joshjob42/pocket-casts-android) (branch `mimi`) plays the `.mimi`
files, decoding on the phone's NPU / GPU, and shows the transcripts, highlights ad breaks on the seek bar and skips
them. Each part is optional per podcast: compression, and transcripts + ad marks.

Background and measurements: `reports/Neural audio codecs SOTA 2026.md`, `bakeoff_2026-10-30.md`. Timings in code
comments and `scripts/*_server.py` were measured on an AMD Ryzen AI Max+ 395 server (Radeon 8060S, gfx1151) and a
Galaxy Z Fold 8 (Snapdragon, Hexagon V81 NPU).

| where | what |
|---|---|
| `hub/` | **mimi-hub** (FastAPI, port 8765): progressive `.mimi` conversion and streaming, transcripts with speakers, LLM ad detection (`segments.py`); setup and service files |
| `FORMAT.md`, `scripts/mimi_file.py` | the `.mimi` container |
| `phone/libmimi/` | the C decoder on LiteRT (NPU transformer, GPU conv decoder) |
| `android/` | Media3 extension (`:mimi`), demo player, Mimi Bridge (see `android/README.md`) |
| `pocket-casts-android/` | the patched Pocket Casts, a separate repository cloned here (see below) |
| `scripts/`, `julia/` | the codec bake-off, benchmarks, Modal GPU tests |

## Set up a server

Linux or macOS (Apple silicon) with Python 3.12 ([uv](https://docs.astral.sh/uv/)) and `ffmpeg`. A GPU makes it
much faster but isn't required: a 44 min episode takes about 45 s end to end on the Radeon 8060S, about 5 minutes on
its CPU alone. With a backlog, the hub processes ~80x real time on the Radeon 8060S, ~67x on an M5 MacBook Pro and
~38x on an M4 iMac (pyannote's ResNet on the Neural Engine).

```sh
git clone https://github.com/joshjob42/Podcast-compress-transcribe neural_audio
cd neural_audio
uv sync                    # CPU environment (.venv)
hub/setup_rocm.sh          # or, AMD GPU: .venv-rocm (AMD's ROCm torch wheels; tested on gfx1151)
hub/setup_cuda.sh          # or, NVIDIA GPU: .venv-cuda (CUDA 12.8 torch wheels)
hub/setup_mac.sh           # or, Apple silicon: .venv-mac (Metal GPU + Core ML for the Neural Engine)
hub/fetch_models.sh        # CPU speaker diarization models (sherpa-onnx), into hub_models/
```

- **Models:** Mimi, Parakeet and Silero VAD download from Hugging Face on first use. The GPU diarizer, pyannote
  speaker-diarization-community-1, is gated: accept its conditions on huggingface.co, then run once
  `HF_TOKEN=... hf download pyannote/speaker-diarization-community-1`. Without it the hub uses the CPU diarizer.
- **Ad detection** needs either an OpenRouter API key in `OPENROUTER_API_KEY` (Claude Haiku 5.5, ~$0.004 per hour of
  audio) or an agent CLI signed in on the server (`claude` and/or `codex`, run in print mode with your own login);
  see `MIMI_HUB_SEGMENTER` below, or set `MIMI_HUB_SEGMENTER=off`.
- **Try it:** `hub/serve.sh --host 127.0.0.1 --port 8765`, then
  `curl -X POST localhost:8765/v1/convert -H 'Content-Type: application/json' -d '{"url": "<episode audio URL or pca.st link>"}'`
  and poll `GET /v1/episodes/<id>`. `hub/serve.sh` uses `.venv-rocm` or `.venv-cuda` if present, else the uv `.venv`.
- **Network:** bind the hub to a private address (a Tailscale IP works well) rather than the internet, or set
  `MIMI_HUB_TOKEN` and give the app the URL with the token in it, `https://<server>/t/<token>` (the app sends no
  header of its own).

### Or: your own server on Modal

No machine of your own: `hub/modal_app.py` runs the hub on a [Modal](https://modal.com) GPU that's only up (and
billed) while it works, with ad detection through an OpenRouter key.

```sh
pip install modal && modal setup
modal secret create mimi-hub MIMI_HUB_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') \
  [OPENROUTER_API_KEY=<your key>] [HF_TOKEN=<token>]   # both optional: below
modal deploy hub/modal_app.py       # prints https://<workspace>--mimi-hub-serve.<region>.modal.run
```

In the app, Settings > Advanced > Mimi server: `https://<workspace>--mimi-hub-serve.<region>.modal.run/t/<token>`
(`modal secret` doesn't show the token again: keep it). The app wakes the server when it comes to the foreground.

- **Ad detection:** either the server's own `OPENROUTER_API_KEY` (in the secret), or none there and the app's
  Settings > Advanced > Mimi server > OpenRouter key, sent with each conversion (`X-OpenRouter-Key`) and held only
  in the server's memory until that episode's ads are found. Episodes whose detection failed for want of a key are
  redone when one arrives.

- **First start:** converts the models and compiles their kernels, ~5-10 min, kept on the `mimi-hub-models` Volume.
  The next few cold starts (~1 min each) also make Modal's memory snapshots of the loaded container.
- **Cold start after that** (L40S, the default; an H100 when no L40S is free): restored from a snapshot, the server
  has its models ready ~8 s after the request. A 43 min episode then plays after ~2-3 s and is fully transcribed
  after ~13 s (+ ~10 s for ads). An H100 was no faster.
- **Cost:** Modal bills while the container is up: an L40S with 16 cores ~$3 an hour. A launch, restored and
  scaling down 20 s after its last request or job, ~$0.02 (~$0.05 when the app wakes it, which holds it 30 s for
  you to pick an episode; at most every 15 min), plus ~$0.02 per hour of audio. Modal's free plan includes $30 a
  month. `MIMI_MODAL_GPU` / `MIMI_MODAL_CPU` at deploy pick others (see `hub/modal_app.py`).
- **Speaker labels:** pyannote's community-1 model is gated: accept its conditions on huggingface.co and put
  `HF_TOKEN` in the secret (or keep its snapshot in `data/modal_pyannote`); otherwise the CPU diarizer.
- `scripts/remote_episode.py <url>/t/<token>` times a cold start and one episode against a running server.

### Or both: a home server with Modal as its spare GPU

The app talks only to your home server, which keeps the library, answers every status check (free, always on) and
runs the ad detection with its own LLM logins; jobs it can't start right away go to the Modal server and come back:

```sh
cat >> ~/.config/mimi-hub/hub.env <<'END'     # read by the hub at start; the environment wins
MIMI_HUB_REMOTE=https://<workspace>--mimi-hub-serve.<region>.modal.run
MIMI_HUB_REMOTE_TOKEN_FILE=~/.config/mimi-hub/modal-token   # the Modal server's MIMI_HUB_TOKEN
MIMI_HUB_REMOTE_WHEN=busy                     # busy (every local worker busy) | priority | always | off
END
```

A job on Modal is converted there without ad detection, copied back (~2 MB per hour of audio) and its ads found at
home; while it's being made the home server relays its `.mimi`, so the app can stream it. One Modal can't do is
redone at home. Uploads and episodes that keep their source stay home. On halo (2026-11-02): three episodes at once,
two converted at home and the third on Modal, all with Opus's ad marks.

A backlog: `scripts/modal_backlog.py` converts a list of episodes in bulk on a Modal H100 into the Modal library
(1,131 episodes, 912 h of Revolutions, The Weeds and The Ezra Klein Show in ~45 min at ~1,350x real time, ~$5),
`scripts/import_modal_library.py` copies them home, and `scripts/backlog_ads.py` finds their ads with the home
server's `claude` login (Haiku 5.5).

### Run it as a service

`hub/mimi-hub.service` is a systemd user unit:

```sh
cp hub/mimi-hub.service ~/.config/systemd/user/   # edit WorkingDirectory, --host and Environment
systemctl --user daemon-reload
systemctl --user enable --now mimi-hub
loginctl enable-linger "$USER"                     # keep running while you're logged out
journalctl --user -u mimi-hub -f                   # logs
```

On NixOS with home-manager, `hub/mimi-hub.service.nix` is the same unit as a module snippet. `hub/run.sh` starts a
one-off transient unit instead.

### Server settings

Environment variables, e.g. `Environment=` lines in the unit:

| variable | default | |
|---|---|---|
| `MIMI_HUB_COMPRESS` | `1` | `0`: no `.mimi` at all, only transcripts and ad marks (the app then plays originals via the hub) |
| `MIMI_HUB_SOURCES` | `fetch` | originals for transcript-only episodes: `fetch` = not stored, re-fetched from the publisher when asked (409 if the publisher's file changed); `keep` = stored on disk |
| `MIMI_HUB_DATA_MAX_GB` | `0` (no cap) | a cap on the data directory, checked once a minute: past it the originals go first (a `.mimi` episode keeps everything else; a transcript-only one is removed entirely: audio, transcript, ad marks, summary), then whole `.mimi` episodes, least recently served first in each; one asked for again is processed afresh. Uploads, unfinished backlogs and episodes still being worked on are left alone |
| `MIMI_HUB_SEGMENTER` | `claude:claude-haiku-5-5>claude:claude-opus-5-5,openrouter:anthropic/claude-haiku-5.5,codex:gpt-5.6-terra+high` | ad detection, tried in order (the subscription first; a CLI that isn't installed or an API without a key is skipped): APIs (`openrouter:` / `inception:`, key in `OPENROUTER_API_KEY` / `INCEPTION_API_KEY` or the app's `X-OpenRouter-Key`) or logged-in agent CLIs (`claude:` / `codex:` / `agy:`, `+effort` / `+fast` flags); `A>B`: B re-reads 60 s around each break A found (Haiku then Opus scored as Opus does, Opus reading 13.5% as much); `off` to disable. With only an OpenRouter key, Haiku 5.5 found all 98 judged breaks in 24 episodes for ~$0.004 per hour of audio; without one it's Opus through the `claude` CLI. Prompt `MIMI_HUB_SEGMENTS_PROMPT` (`v2`). A backlog: `python -m hub.segments --many 4 *.vtt`. `scripts/ads_bench.py` compares models against `scripts/ads_truth.py`'s judged breaks |
| `MIMI_HUB_SUMMARIZER` | `off` | executive summaries after the ad detection, a request of their own: `on` = `claude:claude-haiku-5-5,openrouter:anthropic/claude-haiku-5.5` (or a list as `MIMI_HUB_SEGMENTER`'s); `GET /v1/episodes/<id>.summary.json` (HTML of a few tags, sanitized), `POST /v1/episodes/<id>/summary` for older episodes |
| `MIMI_HUB_BACKLOG_MODAL_H` | `3` | backlogs from the app (`POST /v1/backlog`, `hub/backlog.py`) with at least this many hours of new audio run in bulk on Modal (if the `modal` CLI is set up here; `MIMI_HUB_BACKLOG_MODAL=0` never), smaller ones in the hub's queue; `MIMI_HUB_BACKLOG_CHUNK` (150), `MIMI_HUB_BACKLOG_THREADS` (8) |
| `MIMI_HUB_DEVICE` | `auto` | `cuda` / `cpu` |
| `MIMI_HUB_DIARIZER` | `auto` | `pyannote` (GPU; gated HF model, fetch once with a token) / `sherpa` (CPU) |
| `MIMI_HUB_ASR_ENCODER` | `native` | `onnx2torch` for the converted ONNX encoder |
| `MIMI_HUB_ASR_DECODER` | `native`; `onnx` on ROCm | `native`: TDT decoding batched on the GPU, same output as onnx_asr's CPU loop; frees the CPU |
| `MIMI_HUB_DIAR_DTYPE` | `fp16` on NVIDIA, `coreml` on macOS, `fp32` on ROCm | precision of pyannote's embedding ResNet (`fp32` / `tf32` / `fp16` / `bf16` / `coreml`: the Neural Engine, default on macOS; same diarization) |
| `MIMI_HUB_COMPILE` | `auto`: `aot` on CUDA and ROCm GPUs | Parakeet's encoder and pyannote's embedding ResNet compiled: `aot` = AOTInductor packages (`hub/aoti.py`, built on first use into `hub_models/aoti`: minutes; needs the CUDA toolkit or `rocm[devel]`), `jit` = `torch.compile` (slow on free-threaded Python), `0` = eager. Server 89 → 97× real time; H100 16 processes 1,646 → 1,942×, bulk mode 1,416 → 1,902× |
| `MIMI_HUB_MIMI_FP16` | `1` | `0`: Mimi encoder in fp32 on the GPU (fp16 is ~1.7x faster; the coarse codes nearly all match) |
| `MIMI_HUB_PROCS` | `1` | worker processes, each with its own models: a fast GPU needs several (Modal L40S: 1 -> 6 processes 170 -> 441x; M5 Mac: 1 -> 2, 67 -> 77x; the server: +3%) |
| `MIMI_HUB_WORKERS` | `2` | episodes at a time per process |
| `MIMI_HUB_BLAS_THREADS` | `1` | OpenBLAS threads (Parakeet's features are computed with NumPy) |
| `MIMI_HUB_STREAM_ASR` | `0` | `1`: transcribe during the download (same transcript; pays off only where the GPU has room to spare, not on the server) |
| `MIMI_HUB_TOKEN` | (none) | require `Authorization: Bearer <token>` |
| `MIMI_HUB_DATA` | `hub_data/` | where `.mimi` / `.vtt` / records live |

### Batch conversion

`hub/bulk.py` converts many files or URLs at once, outside the hub, into the hub's `.mimi` and `.vtt` files:

```sh
.venv-rocm/bin/python -m hub.bulk out/ episode1.mp3 episode2.mp3 ...
```

One process holds one copy of the models and batches each model's GPU work across all the episodes in flight
(Mimi in lockstep pieces, Parakeet's segments sorted by length, pyannote's windows through shared batchers); decoding,
VAD and features run in a pool of processes. Transcripts and speaker turns match the hub's (word similarity
0.998-1.0, DER 0); Mimi codes ~96% (batched fp16 rounds differently).

On NVIDIA, Mimi's streaming runs as CUDA graphs (`hub/mimi_static.py`: fixed-size state, one replay per 20 s chunk
of a batch). The process is still limited by Python launching GPU work, so it runs best on free-threaded Python
(3.14t, no GIL). On an H100: Python 3.12 ~440x real time; 3.14t ~790x, ~890x with `MIMI_BULK_IN_FLIGHT=96`
(for comparison, 16 hub-style processes under NVIDIA MPS: ~1,500x, but with 16 copies of the models). The
free-threaded environment is built in `scripts/modal_hub_bench.py`'s image: uv's `cpython-3.14.10+freethreaded`,
torch 2.14.1+cu130 (`cp314t`), protobuf's C extensions removed (they segfault without the GIL) and
`PYTHON_GIL=0`. On the Radeon 8060S it makes no difference (~90x vs ~80x for the hub; GPU-bound).

Since then, on an H100 (1,024 five-minute episodes, 5 s diarization step): AOTInductor for Parakeet's encoder and
the ResNet (`MIMI_HUB_COMPILE`), ~1,750–1,900x; the TDT search as CUDA graphs (`MIMI_BULK_ASR_GRAPH`, on by default
on NVIDIA; each graph set captured on a stream of its own, `hub/graph_stream.py`: two graph sets sharing a cuBLAS
workspace used to hang the GPU), 2,083x; Mimi's SEANet as a TensorRT engine (`MIMI_BULK_MIMI_TRT`,
`hub/trt_seanet.py`; needs `tensorrt-cu13` and `libnvinfer-headers-dev`, built on first use), 2,165x. Its Mimi codes
differ from cuDNN's in ~20% of entries but decode to audio as close to the original (log-mel L1 0.585 either way).
Settings: `MIMI_BULK_*` at the top of `hub/bulk.py`. `scripts/bulk_bench.py` measures it and checks it against the
hub.

## Build the app

The patched Pocket Casts includes this repository's `android/mimi` module (`../android/mimi`), so clone it inside
the checkout:

```sh
cd neural_audio
git clone -b mimi https://github.com/joshjob42/pocket-casts-android
android/fetch_deps.sh      # LiteRT, Qualcomm QNN runtime, Mimi decoder graphs + tables (not in git)
echo "mimiHubUrl=http://<your-hub>:8765" >> ~/.gradle/gradle.properties   # optional: the app's default hub URL
cd pocket-casts-android && ./gradlew :app:assembleDebugProd
```

- **Toolchain:** JDK 21, Android SDK 37 (36 for the standalone `android/` apps) and NDK 29.0.14206865 with CMake
  (the decoder is native).
  `android/pc_build.sh` runs the same build in a container.
- **Phone:** arm64 only. The NPU path needs the transformer precompiled for that phone's SoC
  (`mimi_dec_tx_npu.tflite`; see `phone/README.md`). Without it, decoding runs on the CPU and GPU, still well
  above real time.
- **In the app:** Settings > Advanced > Mimi server sets the hub URL and the defaults (compression, transcripts and
  ad marks, ad and self-promotion skipping, trimmed timeline, download when streaming). Each podcast's settings
  page overrides compression and transcripts.

`android/` also builds Mimi Bridge, which lets any player play `.mimi` files, and a demo player
(`android/README.md`).

## Bake-off pipeline

Python env: `uv sync` (CPU torch). Needs `opusenc` (libopus 1.6.1) and `ffmpeg` on PATH (on NixOS:
`nix shell nixpkgs#opus-tools nixpkgs#ffmpeg`, and nix-ld for the Python wheels).

1. `data/raw/` episode MP3 → `data/ref/ref_{16,24,48}k.wav` (mono, ffmpeg).
2. Opus: `opusenc --bitrate N` on `ref_48k.wav` → `outputs/opus/opus_Nk.{opus,wav}` (stock decoder,
   no 1.6 neural enhancement).
3. `scripts/run_codecs.py {mimi,qwen12hz}` — 30 s segments coded independently (cold start = seek case).
4. `scripts/eval.py` — per-segment dWER vs the reference's Whisper transcript, WavLM speaker cosine,
   PESQ-WB, log-mel L1 → `outputs/eval/seg_<variant>.csv` (one process per variant runs fine in parallel
   once `asr_ref.json` is cached).
5. `julia --project=julia julia/bb_summary.jl outputs/eval` — Bayesian-bootstrap table.
6. `scripts/bench_decode.py` — decoder GFLOP per audio second, 1/4-thread RTF, 2 s cold decode.
7. `scripts/mimi_file.py encode IN OUT.mimi` — production encoder (FORMAT.md); `phone/` — the Android decoder.
8. `scripts/transcribe_vtt.py` — sidecar WebVTT for the episode; `scripts/make_clips.py` — listening page.

`data/` and `outputs/` are gitignored (copyrighted audio, large files).

## mimi-hub internals

- **Progressive:** a job writes the header and preallocates the `.mimi` within about a second, then encodes 10 s
  and then 60 s chunks as one stream (the encoder's caches carried across), while ffmpeg is still decoding.
  `GET <id>.mimi` serves bytes as they land (Range requests work; a chunk a reader seeks to is encoded first).
- **Transcript:** Parakeet TDT 0.6B v2: transformers' native encoder in fp16 on the GPU (checked against NVIDIA's
  NeMo), onnx-asr's decoding, segments batched by length; sentence-sized cues with `<v Speaker N>` voices.
- **Speakers:** pyannote community-1 on the GPU with a batched embedding step (`hub/pyannote_gpu.py`), or
  sherpa-onnx on the CPU. Speakers with under 30 s in total are folded into their neighbours.
- **Ads:** `hub/segments.py` asks an LLM for ad and self-promotion cue ranges as soon as the ASR cues exist.
- **Originals:** with compression off for a podcast, the hub hands the client the exact audio it processed (`.src`),
  since dynamically inserted ads differ between downloads.
- **Tests:** `hub/test_hub.py` (upload), `hub/test_stream.py` (progressive streaming); in-process, ffmpeg on PATH.
  The API is documented in the `hub/mimi_hub.py` docstring.
