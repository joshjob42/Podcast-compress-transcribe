"""mimi-hub: on-demand podcast conversion on the server -> `.mimi` audio + WebVTT transcript.

Serve on the Tailscale address only, e.g.
    uv run uvicorn hub.mimi_hub:app --host <tailscale-ip> --port 8765
Optional auth: set MIMI_HUB_TOKEN; clients then send `Authorization: Bearer <token>`, or put it in the URL:
/t/<token>/v1/... is /v1/... with the token sent (for a client that's only given a server URL, as the app is).

API (JSON unless noted):
  Any request that queues work may carry the client's own OpenRouter key, `X-OpenRouter-Key: <key>`: used for that
  episode's ad detection instead of the server's OPENROUTER_API_KEY, held in memory only (not in the record or on
  disk), dropped once the detection has run. Sent for an episode whose detection failed (no key), it re-runs it.
  POST /v1/convert {"url": ...[, "id", "title", "priority"]}  enqueue (priority 0 = about to be streamed,
                                                jumps the queue). The URL may be a direct audio URL or a Pocket
                                                Casts share link (pca.st / pocketcasts.com), which is resolved to
                                                the episode's enclosure; the id is then the Pocket Casts episode UUID.
  POST /v1/upload?title=...  (multipart "file")  enqueue an uploaded audio file
  GET  /v1/episodes/{id}                        status: queued | starting | downloading | encoding | transcribing
                                                | ready | failed
  GET  /v1/episodes/{id}.mimi                   the audio, Range requests supported. Available from the moment the
                                                job is "encoding": bytes are served as their chunk is encoded, so a
                                                client can play (and seek in) a .mimi that is still being made.
  GET  /v1/episodes/{id}.vtt                    the transcript
  GET  /v1/episodes/{id}.src                    the original audio exactly as processed (only with keep_source):
                                                for clients that want the transcript and ad segments but not the
                                                .mimi. Dynamically inserted ads differ between downloads, so a
                                                fresh copy from the publisher may not match their timings.
                                                MIMI_HUB_SOURCES=fetch (default): not stored; fetched from the
                                                publisher again and passed through if its size still matches the
                                                processed file's, else 409. =keep: stored on disk and served.
Server settings: MIMI_HUB_COMPRESS=0 skips the .mimi entirely (transcripts and ad segments only; every episode
then keeps its source); records say "compressed": false.
  GET  /v1/episodes/{id}.segments.json          ads / self-promotion to skip, found by an LLM after transcription
                                                ({"model", "segments": [{"start", "end", "kind": "ad" | "promo",
                                                "label"}]}, seconds); the record's "segments_state" says
                                                pending | ready | failed | off. See hub/segments.py.
  POST /v1/episodes/{id}/segments               re-run that detection
  GET  /v1/episodes/{id}.summary.json           an executive summary (MIMI_HUB_SUMMARIZER; off by default), made after
                                                the ad detection by a request of its own (Haiku 5.5): {"model",
                                                "summary": HTML of <h4> <p> <ul> <ol> <li> <strong> <em> only}; the
                                                record's "summary_state" says pending | ready | failed (absent: none
                                                asked for). .summary.html: the HTML (no scripts: CSP); .summary.txt:
                                                its text.
  POST /v1/episodes/{id}/summary                make (or remake) it, e.g. for an episode from before summaries
  GET  /v1/features                             what this hub does: {"compress", "segments", "summaries", "remote",
                                                "backlog", "backlog_modal"}
  POST /v1/backlog {"podcast_uuid"[, "latest", "after", "before"] | "episodes": [{"id", "url", "title"[,
                   "duration_s"]}][, "title", "preview": true]}
                                                many episodes at once (hub/backlog.py): a podcast's (all, the latest
                                                N, published in [after, before)) or a list; preview: what it would
                                                do ({"episodes", "new", "hours", "mode": modal | local, "usd",
                                                "minutes"}); else a job (202): {"id", "state", "converted", "ads",
                                                "summaries", ...}. Big ones run in bulk on Modal.
  GET  /v1/backlogs[?podcast_uuid=]             the jobs, newest first; GET /v1/backlogs/{id}: one
  POST /v1/backlogs/{id}/cancel                 stop it (what's converted so far stays)
  GET  /v1/episodes                             all jobs, newest first
  POST /v1/warm                                 load the models now (in the background) and run each once, so the
                                                next episode starts at full speed: {"models": cold | loading | ready
                                                | failed}. A client calls it when it opens; MIMI_HUB_PRELOAD=1 does
                                                it at startup (hub/modal_app.py, a server that scales to zero),
                                                which then stays up MIMI_HUB_WARM_LINGER_S (30) for the client.
  GET  /v1/hold?s=60                            returns once nothing is converting or waiting (or after s seconds):
                                                a request in flight keeps a serverless container (Modal) alive
                                                while the work it started in the background runs.
Files live in $MIMI_HUB_DATA (default neural_audio/hub_data/): <id>.mimi, <id>.vtt, <id>.segments.json, <id>.json.

A remote GPU (MIMI_HUB_REMOTE: another hub's URL, e.g. hub/modal_app.py's, with /t/<token> or its token in
MIMI_HUB_REMOTE_TOKEN_FILE): jobs this hub can't start
right away (MIMI_HUB_REMOTE_WHEN=busy, the default: every local worker busy; priority: also every priority-0 job;
always) are converted there and copied back; this hub keeps the library, the clients and the ad detection (its own
LLM logins), and relays the remote's .mimi while it's being made. A job the remote fails is redone here.

GPU: run in .venv-rocm (hub/setup_rocm.sh; hub/serve.sh picks it) and the Mimi encoder and Parakeet's encoder run on
the Radeon 8060S (MIMI_HUB_DEVICE=auto | cuda | cpu). Diarization and Parakeet's decoding loop stay on the CPU.
"""

import atexit
import contextlib
import gzip
import hashlib
import hmac
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Settings from a file too (KEY=value a line, # comments; the environment wins): MIMI_HUB_ENV_FILE, default
# ~/.config/mimi-hub/hub.env. For a service whose unit is generated (home-manager) or settings kept out of it.
_env_file = Path(os.environ.get("MIMI_HUB_ENV_FILE", Path.home() / ".config/mimi-hub/hub.env"))
if _env_file.is_file():
    for _ln in _env_file.read_text().splitlines():
        _k, _eq, _v = _ln.strip().partition("=")
        if _eq and _k and not _k.startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip())

import mimi_file  # noqa: E402

from hub import segments  # noqa: E402

# uv's Python doesn't know NixOS's CA bundle; systemd-run services don't inherit SSL_CERT_FILE.
if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

# MIOpen (ROCm's conv library) otherwise tunes every new input shape exhaustively; Parakeet's batches all differ
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
# torch.compile's kernels (MIMI_HUB_COMPILE) kept across restarts: the first start compiles (Parakeet's encoder took
# 4.5 min on the server's CPU), later ones load from here
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(os.environ.get("MIMI_HUB_MODELS", ROOT / "hub_models"))
                                                     / "inductor-cache"))

DATA = Path(os.environ.get("MIMI_HUB_DATA", ROOT / "hub_data"))
DEVICE = os.environ.get("MIMI_HUB_DEVICE", "auto")
SOURCES = os.environ.get("MIMI_HUB_SOURCES", "fetch")  # fetch | keep (the originals of keep_source episodes)
MAX_BYTES = int(float(os.environ.get("MIMI_HUB_DATA_MAX_GB", "0")) * 1e9)  # 0: no cap (trim_data)
COMPRESS = os.environ.get("MIMI_HUB_COMPRESS", "1") != "0"  # 0: no .mimi, transcripts + ad segments only
TOKEN = os.environ.get("MIMI_HUB_TOKEN", "")
ASR_MODEL = os.environ.get("MIMI_HUB_ASR", "nemo-parakeet-tdt-0.6b-v2")  # -v3 for multilingual
THREADS = int(os.environ.get("MIMI_HUB_THREADS", "16"))
UA = {"User-Agent": "Mozilla/5.0 (mimi-hub)"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PC_EPISODE = re.compile(r"pocketcasts\.com/podcast/[^/]+/([0-9a-f-]{36})/[^/]+/([0-9a-f-]{36})")

DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="mimi-hub")
jobs: "queue.PriorityQueue" = queue.PriorityQueue()  # (priority, seq, id); 0 = streaming


class _RecordLock:
    """Read-modify-write lock for the job records, across threads and (with MIMI_HUB_PROCS > 1) processes."""

    def __init__(self):
        self.thread = threading.Lock()

    def __enter__(self):
        import fcntl
        self.thread.acquire()
        self.fd = os.open(DATA / ".records.lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        os.close(self.fd)  # releases the flock
        self.thread.release()


lock = _RecordLock()


# ---------- job records ----------

def rec_path(i):
    return DATA / f"{i}.json"


def record_paths():
    return [p for p in DATA.glob("*.json") if not p.name.endswith((".segments.json", ".summary.json"))]


def load(i):
    p = rec_path(i)
    return json.loads(p.read_text()) if p.exists() else None


def save(r):
    r["updated"] = time.time()
    tmp = rec_path(r["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(r, indent=1))
    tmp.replace(rec_path(r["id"]))
    return r


def update(i, **kw):
    with lock:
        r = load(i)
        r.update(kw)
        return save(r)


def enqueue(i, priority=1, keep_source=False, **fields):
    r = _enqueue(i, priority, keep_source, **fields)
    if priority == 0 and r.get("state") == "queued":
        queue_source(i)
    return r


def _enqueue(i, priority=1, keep_source=False, **fields):
    """Queue job i (priority 0 = someone is streaming it, 1 = download / background). [keep_source]: keep the
    original audio as processed and serve it (.src)."""
    keep_source = keep_source or not COMPRESS  # without a .mimi, clients play the original
    with lock:
        r = load(i)
        if r and r["state"] != "failed":
            if keep_source and not r.get("keep_source"):
                r = save(dict(r, keep_source=True))
                if r["state"] == "ready":  # processed without its source: redo it keeping one, so all match
                    r = save(dict(r, state="queued", priority=priority, progress=0.0, source_ready=False))
                    jobs.put((priority, next(_seq), i))
                    return r
                # in progress: run() queues it again when done
            if r["state"] == "queued" and priority < r.get("priority", 1):  # bump: re-queue at higher priority
                r = save(dict(r, priority=priority))
                jobs.put((priority, next(_seq), i))
            return r  # already queued, running or done
        r = save(dict(id=i, state="queued", priority=priority, progress=0.0, error=None, created=time.time(),
                      keep_source=keep_source, compressed=COMPRESS, **fields))
    jobs.put((priority, next(_seq), i))
    return r


def fetch_source(url, i):
    """Download [url] to <id>.src; its MIME type."""
    # a temporary file of its own: two fetches of one episode (processing, and an early fetch) never share one
    path, tmp = DATA / f"{i}.src", DATA / f"{i}.src.{os.getpid()}-{threading.get_ident()}.tmp"
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            ctype = resp.headers.get_content_type()
            shutil.copyfileobj(resp, f, 1 << 20)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return ctype if ctype.startswith("audio/") else "audio/mpeg"


def get_source(i, r):
    """Download job i's audio (the first readable of its candidate URLs) to <id>.src and mark it source_ready."""
    err = None
    for u in candidates(r["audio_url"]):  # skip tracking hops that are unreachable
        try:
            probe(u)
            break
        except Exception as e:
            err = e
    else:
        raise RuntimeError(f"cannot read source: {err}")
    ctype = fetch_source(u, i)
    path = DATA / f"{i}.src"
    update(i, source_ready=True, source_type=ctype, source_bytes=path.stat().st_size, source_url=u)
    return str(path)


# MIMI_HUB_DATA_MAX_GB caps the data directory, checked once a minute. Past it the originals go first, then whole
# episodes, the least recently served first in each. An episode with a .mimi only loses its .src: the .mimi is ours, and
# at ~2 MB per hour removing it frees little and costs a conversion, so those go last. A transcript-only episode goes
# entirely (audio, transcript, ad marks, summary, record): kept data without the audio it was made from is of little
# use, since dynamically inserted ads mean the publisher's file has usually changed by the time it's asked for again.
# Removed, an episode asked for again is simply processed afresh, from the file the client gets. Untouched: episodes
# queued, being processed, waiting for their ad marks or summary, or in an unfinished backlog, and uploads (no URL to
# get them from again).
TRIM_EVERY_S = 60
_trim_stuck = [False]  # over the cap with nothing left to remove (said once)


def _data_bytes():
    total = 0
    for f in DATA.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:  # deleted meanwhile
            pass
    return total


def _trimmable(r, backlogs):
    return (r.get("state") in ("ready", "failed") and not r.get("upload") and r["id"] not in backlogs
            and r.get("segments_state") != "pending" and r.get("summary_state") != "pending"
            and not r.get("source_fetching")
            # (converted in bulk by hand: scripts/backlog_ads.py is still to find its ads)
            and not (r.get("batch") and r.get("segments_state") not in ("ready", "off")))


def _mark_served(i, r):
    """For trim_data's order (least recently served first); at most once an hour per episode."""
    if MAX_BYTES and r and time.time() - (r.get("served") or 0) > 3600:
        with lock:
            if r := load(i):
                save(dict(r, served=time.time()))


def _remove(files):
    """The bytes freed."""
    freed = 0
    for f in files:
        try:
            size = f.stat().st_size
            f.unlink()
            freed += size
        except OSError:
            pass
    return freed


def trim_data():
    used = start_bytes = _data_bytes()
    if used <= MAX_BYTES:
        _trim_stuck[0] = False
        return
    from hub import backlog
    backlogs, todo = backlog.active_ids(), []  # todo: (0: its original | 1: the whole episode, last served, id)
    for f in record_paths():
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if _trimmable(r, backlogs):
            when = r.get("served") or r.get("created") or 0
            if (DATA / f"{r['id']}.src").exists():
                todo.append((0, when, r["id"]))
            if r["state"] == "ready":
                todo.append((1, when, r["id"]))
    originals = episodes = 0
    for whole, _, i in sorted(todo):
        if used <= MAX_BYTES:
            break
        with lock:  # still removable (not re-queued since)
            r = load(i)
            if not r or not _trimmable(r, backlogs):
                continue
            if not whole and (r["state"] != "ready" or (DATA / f"{i}.mimi").exists()):
                used -= _remove([DATA / f"{i}.src"])  # .src now re-fetches from source_url, as with SOURCES=fetch
                originals += 1
            else:  # the record first: a record means its files are there
                used -= _remove([rec_path(i), *DATA.glob(f"{i}.*")])
                episodes += 1
    stuck = used > MAX_BYTES
    if originals or episodes or (stuck and not _trim_stuck[0]):
        print(f"disk: data over the {MAX_BYTES / 1e9:g} GB cap; removed {originals} originals and {episodes} episodes "
              f"({(start_bytes - used) / 1e9:.1f} GB), now {used / 1e9:.1f} GB" + (
                  ": still over (the rest are uploads, in a backlog, queued or being processed)" if stuck else ""),
              flush=True)
    _trim_stuck[0] = stuck


def trim_worker():
    while True:
        try:
            trim_data()
        except Exception:
            traceback.print_exc()
        time.sleep(TRIM_EVERY_S)


# A client about to download an episode (priority 0) waits a few minutes for its source, and processing may be hours
# away behind a backlog. So the source is fetched as soon as the episode is queued; processing later uses that copy,
# and the client's download lines up with the transcript and ad segments whenever they arrive.
src_jobs: "queue.Queue" = queue.Queue()


def queue_source(i):
    src_jobs.put(i)


def source_worker():
    while True:
        i = src_jobs.get()
        with lock:
            r = load(i)
            if (not r or r["state"] != "queued" or not r.get("keep_source") or not r.get("audio_url")
                    or r.get("source_ready") or r.get("source_fetching")):
                continue
            save(dict(r, source_fetching=time.time()))
        t0 = time.time()
        try:
            get_source(i, r)
            print(f"source: fetched {i} ahead of processing in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            print(f"source: fetching {i} ahead of processing failed ({e}); processing will try again", flush=True)
        finally:
            update(i, source_fetching=None)


def await_source_fetch(i):
    """Wait for an in-flight early fetch of job i's source (so processing doesn't fetch a second, different copy:
    dynamically inserted ads differ between downloads). As long as it runs: its reads time out after 60 s, it clears
    source_fetching when it ends, and start() clears it after a restart."""
    while (r := load(i)) and r.get("source_fetching"):
        time.sleep(1)


# ---------- URL resolution ----------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def location(url):
    """The Location header of one GET to `url` (None if it does not redirect)."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(url, headers=UA), timeout=30)
        return None
    except urllib.error.HTTPError as e:
        return e.headers.get("Location") if 300 <= e.code < 400 else None


def resolve(url):
    """Returns (id, audio_url, title). Pocket Casts links -> episode UUID + enclosure."""
    if "pca.st/" in url or "pocketcasts.com/" in url:
        final = url
        for _ in range(5):  # follow short-link redirects by hand: pocketcasts.com 403s non-browsers
            if PC_EPISODE.search(final):
                break
            final = location(final) or final
        m = PC_EPISODE.search(final)
        if not m:
            raise ValueError(f"not a Pocket Casts episode link: {final}")
        pod, ep = m.groups()
        req = urllib.request.Request(f"https://podcast-api.pocketcasts.com/podcast/full/{pod}", headers=UA)
        raw = urllib.request.urlopen(req, timeout=60).read()
        data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        for e in data["podcast"]["episodes"]:
            if e["uuid"] == ep:
                return ep, e["url"], f'{data["podcast"].get("title", "")}: {e["title"]}'
        raise ValueError(f"episode {ep} not in podcast {pod}")
    return hashlib.sha1(url.encode()).hexdigest()[:16], url, ""


# ---------- worker ----------

_models = {}
_models_lock = threading.Lock()


def models(vad_workers=True):
    """The loaded models; [vad_workers]: also start MIMI_HUB_VAD_SHARDS' processes (not before a memory snapshot,
    hub/modal_app.py: started after its restore)."""
    with _models_lock:  # jobs starting together on a fresh hub must not see half-loaded models
        if not _models:
            _models.update(_load_models())
        if vad_workers and _vad_pool is None:
            t = time.time()
            _vad_workers()  # (MIMI_HUB_VAD_SHARDS) started with the models, not by the first episode
            _stage("VAD worker processes", t)
    return _models


def _stage(what, t0):
    """A startup stage's time, logged (where a cold start goes)."""
    print(f"startup: {what} {time.time() - t0:.1f} s", flush=True)
    return time.time()


_warm = {"models": "cold"}  # cold | loading | ready | failed (POST /v1/warm)
_warm_lock = threading.Lock()
PRELOAD = os.environ.get("MIMI_HUB_PRELOAD", "0") == "1"
# after POST /v1/warm, busy() for this long: a server that scales to zero (hub/modal_app.py) stays up while the client
# that woke it picks an episode, though it otherwise scales down seconds after its last job
WARM_LINGER_S = float(os.environ.get("MIMI_HUB_WARM_LINGER_S", "30"))
_linger_until = [0.0]


def warm(vad_workers=True):
    """Load the models and run the lazily loaded and first-call-slow parts once (pyannote, ASR) on noise."""
    with _warm_lock:
        if _warm["models"] in ("loading", "ready"):
            return
        _warm["models"] = "loading"
    t0 = time.time()
    try:
        import numpy as np
        m = models(vad_workers)
        t = _stage("models", t0)
        noise = (0.05 * np.random.default_rng(0).standard_normal(16000 * 30)).astype(np.float32)
        diarize(noise)
        t = _stage("diarizer loaded and run", t)
        list(m["asr"].recognize(noise, sample_rate=16000))
        _stage("ASR run", t)
        _warm["models"] = "ready"
        print(f"warm: models ready in {time.time() - t0:.1f} s", flush=True)
    except Exception:
        traceback.print_exc()
        _warm["models"] = "failed"


def _load_models():
    m = {}
    t = time.time()
    import onnx_asr
    import onnxruntime as ort
    import threadpoolctl
    import torch
    torch.set_num_threads(THREADS)
    # onnx_asr computes Parakeet's features with NumPy on the CPU, and OpenBLAS's default (a spin-waiting thread
    # per core) turned their small matrix products into most of the hub's CPU use: ASR 47 -> 19 CPU-ms per audio
    # second on the server with one BLAS thread, at the same speed
    threadpoolctl.threadpool_limits(int(os.environ.get("MIMI_HUB_BLAS_THREADS", "1")), user_api="blas")
    if DEVICE != "auto":
        dev = DEVICE
    elif torch.cuda.is_available():
        dev = "cuda"
    else:  # Apple silicon: the GPU through Metal
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m["device"] = dev
    t = _stage("imports", t)
    if COMPRESS:  # (MIMI_HUB_COMPRESS=0: transcripts and ad marks only, no .mimi: no Mimi either)
        m["mimi"] = mimi_file.load_model(dev)
    if "mimi" in m and MIMI_FP16 and dev != "cpu":  # the encoder's conv / linear weights as autocast uses them: the same codes
        from hub.autocast_weights import half_weights
        half_weights(m["mimi"].encoder)
    t = _stage("Mimi", t)
    # onnxruntime's default spin-waiting threads starve torch's: on a 15 min clip (scripts/bench_overlap.py)
    # ASR alone went 50 -> 31 s with spinning off, and encode || ASR 87 -> 39 s
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    asr = onnx_asr.load_model(ASR_MODEL, sess_options=so)
    t = _stage("onnx_asr Parakeet (ONNX sessions)", t)
    if dev != "cpu":
        try:
            # hub/parakeet_native.py. Checked against NVIDIA's NeMo encoder on real batches: native matches to
            # ~1e-5 relative error, the onnx2torch-converted ONNX export is off by ~4% (median); transcripts of
            # both are within 0.6% of NeMo's own. Native is also 18% more pipeline throughput on the server.
            if os.environ.get("MIMI_HUB_ASR_ENCODER", "native") == "native":
                from hub import parakeet_native
                model_dir = parakeet_native.convert(ASR_MODEL, MODELS_DIR / f"{ASR_MODEL}-hf")
                # its linear / conv weights in fp16, as autocast uses them: identical output, 1.2 GB less GPU memory
                encoder = parakeet_native.load(model_dir, dev, half=True)
                t = _stage("Parakeet encoder (native, to the GPU)", t)
                # compiled ahead of time (hub/aoti.py, fixed-length buckets): 1.7x on an H100; with the ResNet's
                # (hub/pyannote_gpu.py) torch.compile took the server's pipeline 86 -> 99x. MIMI_HUB_COMPILE.
                from hub import aoti
                asr.asr._encoder = parakeet_native.NativeEncoder(encoder, compile=aoti.mode(dev))
                t = _stage("Parakeet encoder AOT packages", t)
                # identical tokens and timestamps to onnx_asr's loop (548/548 segments of a 43.8 min episode;
                # scripts/parakeet_decoder_check.py). It frees the CPU: on Modal's L40S (CPU-bound) pipeline
                # throughput went 214 -> 268x, on an M5 Mac 43 -> 47x; the server (ROCm) is GPU-bound and ~3%
                # faster without it.
                if os.environ.get("MIMI_HUB_ASR_DECODER", "onnx" if torch.version.hip else "native") == "native":
                    path = parakeet_native.convert_decoder(ASR_MODEL, model_dir)
                    asr.asr._decoding = parakeet_native.NativeTdtDecoder(
                        path, dev, asr.asr._blank_idx, asr.asr._vocab_size,
                        max_symbols=asr.asr._max_tokens_per_step)
                    t = _stage("Parakeet decoder", t)
            elif dev == "cuda":
                asr.asr._encoder = TorchEncoder(parakeet_encoder())
        except Exception:
            traceback.print_exc()  # the ONNX encoder on the CPU still works
    # Silero VAD is one tiny ONNX call per 32 ms: onnxruntime's default (a spinning thread per core) used 5.1 CPU-s
    # per 10 min of audio, one thread 1.1 CPU-s in the same wall time
    vad_so = ort.SessionOptions()
    vad_so.intra_op_num_threads = vad_so.inter_op_num_threads = 1
    vad_so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    vad = onnx_asr.load_vad("silero", sess_options=vad_so)
    own_encode = vad._encode

    def vad_encode(waveforms, sample_rate, hop_size, context_size):
        pre = getattr(_vad_pre, "value", None)  # computed while decoding (STREAM_VAD)
        if pre is not None and waveforms.shape == (1, pre[0]):
            yield from pre[1][:, None]
        else:
            yield from own_encode(waveforms, sample_rate, hop_size, context_size)
    vad._encode = vad_encode
    if ASR_BUCKET > 1:
        vad.recognize_batch = types.MethodType(_bucketed_recognize_batch, vad)
    m["asr"] = asr.with_vad(vad).with_timestamps()
    _stage("VAD", t)
    return m


# onnx_asr sends VAD segments to the ASR model 8 at a time in time order, each batch padded to its longest segment:
# 61% of the encoder's input was padding on a 43.8 min episode. Batching all of an episode's segments by length
# (ASR_BUCKET at a time) cut the encoder from 14.3 to 8.3 s on the server (scripts/parakeet_encoder_server.py).
ASR_BUCKET = int(os.environ.get("MIMI_HUB_ASR_BUCKET", "16"))  # 0 / 1: onnx_asr's own batching
# The batches' three steps (features on the CPU, encoder on the GPU, decoding) as a pipeline: one batch's features
# are computed while the previous one is encoded and the one before decoded, instead of each waiting on the others.
ASR_PIPELINE = os.environ.get("MIMI_HUB_ASR_PIPELINE", "1") != "0"


def _pipelined(items, depth=2):
    """Iterate [items] (a generator) in a background thread, up to [depth] ahead."""
    q, end = queue.Queue(maxsize=depth), object()

    def produce():
        try:
            for x in items:
                q.put(x)
            q.put(end)
        except BaseException as e:
            q.put(e)
    threading.Thread(target=produce, daemon=True, name="mimi-asr-stage").start()
    while (x := q.get()) is not end:
        if isinstance(x, BaseException):
            raise x
        yield x


def _asr_batches(asr, batches, samples, asr_kwargs, sample_rate=16000):
    """Recognize [batches] (an iterable of lists of (start, end) sample ranges; may block for more) with [asr],
    taking the audio from samples(start, end): yields (segment, TimestampedSegmentResult) per segment."""
    from onnx_asr.utils import pad_list
    from onnx_asr.vad import TimestampedSegmentResult

    def features():
        for batch in batches:
            yield batch, asr._preprocessor(*pad_list([samples(s, e) for s, e in batch]))

    def encoded(items):
        for batch, feats in items:
            yield batch, asr._encode(*feats)

    if ASR_PIPELINE:
        outputs = (zip(batch, map(asr._decode_tokens, *zip(*asr._decoding(*enc, **asr_kwargs))))
                   for batch, enc in _pipelined(encoded(_pipelined(features()))))
    else:
        outputs = (zip(batch, asr.recognize_batch(*pad_list([samples(s, e) for s, e in batch]), **asr_kwargs))
                   for batch in batches)
    for batch in outputs:
        for (s, e), res in batch:
            yield (s, e), TimestampedSegmentResult(s / sample_rate, e / sample_rate, res.text, res.timestamps,
                                                   res.tokens, res.logprobs)


def _bucketed_recognize_batch(self, asr, waveforms, waveforms_len, sample_rate, asr_kwargs, batch_size=8,
                              **kwargs):
    """Drop-in for onnx_asr BaseVad.recognize_batch: segment each waveform fully, recognize the segments in
    length-sorted batches of ASR_BUCKET, and yield the results in time order."""

    def recognize(waveform, segments):
        segments = list(segments)
        order = sorted(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
        batches = [[segments[j] for j in order[i:i + ASR_BUCKET]] for i in range(0, len(order), ASR_BUCKET)]
        results = dict(_asr_batches(asr, batches, lambda s, e: waveform[s:e], asr_kwargs, sample_rate))
        return iter([results[seg] for seg in segments])

    return map(recognize, waveforms, self.segment_batch(waveforms, waveforms_len, sample_rate, **kwargs))


# ---------- Parakeet's encoder on the GPU ----------
#
# onnxruntime has no working GPU provider for gfx1151 here (no MIGraphX; WebGPU was slower than the CPU and changed
# the output), so the encoder ONNX graph is converted to torch once (onnx2torch) and run on ROCm in fp16, behind the
# InferenceSession interface onnx_asr calls. The encoder is ~85% of Parakeet's CPU time; preprocessing, VAD and the
# TDT decoding loop stay in onnxruntime on the CPU. On a 15 min clip: 33 s -> 10 s, 190/191 segments identical (one
# comma differs). bf16 changed more words; fp32 is identical but half the speed.

PARAKEET_ONNX = {"nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx"}


def parakeet_encoder():
    """Path of the torch-converted encoder of ASR_MODEL, converting it on first use (~2.4 GB, a few seconds)."""
    out = MODELS_DIR / f"{ASR_MODEL}-encoder.pt"
    if out.exists():
        return out
    import onnx2torch
    import torch
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(PARAKEET_ONNX[ASR_MODEL], allow_patterns=["encoder-model.onnx*"]))
    with tempfile.TemporaryDirectory(dir=MODELS_DIR) as tmp:
        for f in src.glob("encoder-model.onnx*"):  # onnx refuses the HF cache's symlinked weight file
            shutil.copyfile(f, Path(tmp) / f.name)
        module = onnx2torch.convert(str(Path(tmp) / "encoder-model.onnx"))
    torch.save(module, out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    return out


class TorchEncoder:
    """Stands in for onnx_asr's encoder InferenceSession: run(names, feeds) -> [outputs, encoded_lengths]."""

    def __init__(self, path):
        import torch
        self.torch = torch
        self.m = torch.load(path, weights_only=False).eval().to("cuda")

    def run(self, names, feeds):
        import numpy as np
        torch = self.torch
        x = torch.from_numpy(feeds["audio_signal"]).cuda()
        n = torch.from_numpy(np.asarray(feeds["length"])).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out, lens = self.m(x, n)
        return [out.float().cpu().numpy(), lens.cpu().numpy()]


def candidates(url):
    """Ways to fetch `url`'s file: for a tracking-prefix chain (podtrac/pdst/pscrb/... URLs end in
    `.../real.host/path`) the real host first (fast, and skips blocked tracker hops), then the outer
    hosts, then the original URL; any other URL is tried as is first."""
    parts = url.split("://", 1)[-1].split("/")
    inner = ["https://" + "/".join(parts[k:]) for k in range(len(parts) - 2, 0, -1)
             if "." in parts[k] and not re.search(r"\.(mp3|m4a|aac|ogg|opus|wav)$", parts[k], re.I)]
    yield from (inner + [url]) if inner else [url]


def cues(seg, max_s=7.0, max_chars=110):
    """Split one timestamped ASR segment into sentence-sized (start, end, text) cues.

    Token times are relative to the segment start; a token with a leading space starts a word. A cue
    ends after sentence punctuation, or (when it gets long) at a comma or word boundary.
    """
    toks, ts = seg.tokens, [seg.start + t for t in seg.timestamps]
    out, cur, t0 = [], "", None
    for k, (tok, t) in enumerate(zip(toks, ts)):
        if t0 is None:
            t0 = t
        cur += tok
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        at_word_end = nxt is None or nxt.startswith(" ")
        long = ts[k] - t0 >= max_s or len(cur) >= max_chars
        sentence = tok.rstrip().endswith((".", "?", "!")) and ts[k] - t0 >= 1.0
        if nxt is None or (at_word_end and (sentence or (long and (tok.endswith(",") or len(cur) >= max_chars)))):
            end = ts[k + 1] if nxt is not None else seg.end
            out.append((t0, max(end, t0 + 0.2), cur.strip()))
            cur, t0 = "", None
    return [c for c in out if c[2]]


# ---------- speaker diarization (sherpa-onnx: pyannote segmentation-3.0 + WeSpeaker ResNet34 + clustering) ----------
#
# Tuned on a 44-min two-host episode with ad reads. 0.7 looked right on one decode of it (hosts 18.6 / 15.9 min)
# but merged the two hosts on another (different dynamic ads; 24 kHz -> 16 kHz resampling), so clustering near 0.7
# is unstable. 0.6 kept the hosts apart on every version tried (~19 / ~16 min) and only adds small ad/news voices;
# over-splitting is the lesser evil. 0.9 merged hosts; TitaNet over-split. ~115 s per 44 min on 8 threads. Speakers
# are numbered by first appearance; ones with < 30 s in total are folded into their neighbours.

# pyannote's speaker-diarization-community-1 (VBx clustering) splits the hosts the same way (19.3 / 16.6 min on the
# same episode). Stock, it took 931 s on the CPU and 142 s on the GPU; with hub/pyannote_gpu.py's batched embedding
# step (identical output: DER 0.0000 against stock) it takes 33 s on the GPU, so it's the default there
# (MIMI_HUB_DIARIZER=auto; or pyannote / sherpa). Gated model: fetch it once into the HF cache with a token
# (accept its terms on huggingface.co, then e.g. `HF_TOKEN=... hf download pyannote/speaker-diarization-community-1`);
# the hub then loads it from the cache, offline, with no token.
MODELS_DIR = Path(os.environ.get("MIMI_HUB_MODELS", ROOT / "hub_models"))
DIARIZER = os.environ.get("MIMI_HUB_DIARIZER", "auto")
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
SHERPA_NAME = "pyannote-seg-3.0 + wespeaker-resnet34 @0.6 (sherpa-onnx)"
DIAR_THRESHOLD, DIAR_MIN_TALK_S = 0.6, 30.0
_diar_lock = threading.Lock()


def _pyannote_turns(wav16):
    import torch
    from hub import pyannote_gpu
    if "diar_pyannote" not in _models:  # from the HF cache (or MIMI_HUB_PYANNOTE_PATH, a copy); no token at runtime
        _models["diar_pyannote"] = pyannote_gpu.load(_models.get("device", "cpu"),
                                                     path=os.environ.get("MIMI_HUB_PYANNOTE_PATH") or None,
                                                     segmentation_device=os.environ.get("MIMI_HUB_SEG_DEVICE") or None)
    out = _models["diar_pyannote"]({"waveform": torch.from_numpy(wav16)[None], "sample_rate": 16000})
    ann = getattr(out, "speaker_diarization", out)
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


_diar_pool = None


def _sherpa_turns(wav16):
    """sherpa-onnx diarization in a child process (hub/diarize_worker.py), which keeps its models loaded."""
    global _diar_pool
    import multiprocessing
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from hub import diarize_worker
    if _diar_pool is None:
        _diar_pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=DATA) as f:
        np.save(f, wav16)
        f.flush()
        try:
            return _diar_pool.submit(diarize_worker.segments, f.name, str(MODELS_DIR), DIAR_THRESHOLD).result()
        except BrokenProcessPool:
            _diar_pool = None  # the child died (e.g. OOM); start a fresh one next time
            raise


def diarize(wav16):
    """([(start_s, end_s, speaker_number)], diarizer name), speakers numbered 1.. by first appearance."""
    with _diar_lock:  # one diarizer, not shared across threads
        segs = None
        if DIARIZER == "pyannote" or (DIARIZER == "auto" and _models.get("device", "cpu") != "cpu"):
            try:
                segs, name = _pyannote_turns(wav16), PYANNOTE_MODEL
            except Exception:
                if DIARIZER == "pyannote":
                    raise
                traceback.print_exc()  # e.g. pyannote missing or its model not cached: use sherpa
        if segs is None:
            segs, name = _sherpa_turns(wav16), SHERPA_NAME
    return number_turns(segs), name


def number_turns(segs):
    """A diarizer's [(start_s, end_s, label)] as the hub keeps them: speakers with under DIAR_MIN_TALK_S in all
    merged into the nearest kept turn's, then numbered 1.. by first appearance."""
    total = {}
    for a, b, k in segs:
        total[k] = total.get(k, 0.0) + b - a
    keep = [(a, b, k) for a, b, k in segs if total[k] >= DIAR_MIN_TALK_S]
    if not keep:
        return []
    out = []
    for a, b, k in segs:
        if total[k] < DIAR_MIN_TALK_S:  # fragment: take the speaker of the nearest kept turn
            k = min(keep, key=lambda t: max(t[0] - b, a - t[1], 0.0))[2]
        out.append((a, b, k))
    number = {}
    for _, _, k in out:
        number.setdefault(k, len(number) + 1)
    return [(a, b, number[k]) for a, b, k in out]


def write_vtt(path, all_cues, turns):
    """[all_cues] ([(start_s, end_s, text)]) as a WebVTT file, each cue voiced by its speaker in [turns]."""
    tmp = Path(path).with_suffix(".vtt.tmp")
    with open(tmp, "w") as f:
        f.write("WEBVTT\n\n")
        for a, b, text in all_cues:
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            who = speaker_for(a, b, turns)
            f.write(f"{vtt_time(a)} --> {vtt_time(b)}\n{f'<v Speaker {who}>' if who else ''}{text}\n\n")
    tmp.replace(path)


def speaker_for(a, b, turns):
    """Speaker number overlapping cue [a, b] most (nearest turn if none overlaps); None without turns."""
    if not turns:
        return None
    best, best_ov = None, 0.0
    for s, e, k in turns:
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = k, ov
    if best is None:
        best = min(turns, key=lambda t: max(t[0] - b, a - t[1], 0.0))[2]
    return best


def vtt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# ---------- progressive conversion ----------
#
# A job probes the source's duration, writes the .mimi header and preallocates the file at its final size,
# then encodes it in chunks (10 s first for a fast start, then 60 s; each with 20 s of left context) while
# ffmpeg is still decoding the source. Readers of <id>.mimi get bytes as their chunk lands; a chunk somebody
# is waiting for (e.g. after a seek) is encoded next. Every byte's position is fixed (44-byte frames), so
# Range requests work against the final file size from the start.

FPS = 12.5
FIRST_CHUNK, CHUNK, CTX = 125, 750, 250  # frames
FRAME_BYTES = 44  # 32 codebooks x 11 bits
WORKERS = int(os.environ.get("MIMI_HUB_WORKERS", "2"))
STREAMABLE = ("encoding", "transcribing", "ready")
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")


def chunk_bounds(n_frames):
    out, f, size = [], 0, FIRST_CHUNK
    while f < n_frames:
        out.append((f, min(n_frames, f + size)))
        f, size = f + size, CHUNK
    return out


class Live:
    """A conversion in progress; its .mimi on disk fills in chunk by chunk."""

    def __init__(self, i, n_frames, header):
        self.i, self.n_frames, self.header = i, n_frames, header
        self.data_offset = len(header)
        self.total = self.data_offset + n_frames * FRAME_BYTES
        self.bounds = chunk_bounds(n_frames)
        self.ready = [False] * len(self.bounds)
        self.wanted = []  # chunks readers are waiting for, most recent first
        self.cond = threading.Condition()
        self.failed = False

    def chunk_of(self, frame):
        c = 0 if frame < FIRST_CHUNK else 1 + (frame - FIRST_CHUNK) // CHUNK
        return min(c, len(self.bounds) - 1)

    def wait_ready(self, c, timeout=300):
        with self.cond:
            if not self.ready[c]:
                if c in self.wanted:
                    self.wanted.remove(c)
                self.wanted.insert(0, c)
                self.cond.notify_all()
            end = time.time() + timeout
            while not self.ready[c] and not self.failed and time.time() < end:
                self.cond.wait(1.0)
            return self.ready[c]


live: dict = {}


class Pcm:
    """24 kHz mono float32 decoded by ffmpeg in the background, usable as it arrives."""

    def __init__(self, src, n_samples, cond):
        import numpy as np
        self.buf = np.zeros(n_samples, np.float32)
        self.avail, self.done, self.error, self.cond = 0, False, None, cond
        http = ["-user_agent", UA["User-Agent"]] if src.startswith("http") else []
        self.proc = subprocess.Popen([FFMPEG, "-v", "error", *http, "-i", src, "-map", "0:a:0", "-ac", "1",
                                      "-ar", "24000", "-f", "f32le", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True, name="mimi-pcm").start()

    def _read(self):
        import numpy as np
        rest = b""
        try:
            while self.avail < len(self.buf):
                b = self.proc.stdout.read(1 << 18)
                if not b:
                    break
                b = rest + b
                k = len(b) // 4 * 4
                x = np.frombuffer(b[:k], np.float32)
                rest = b[k:]
                n = min(len(x), len(self.buf) - self.avail)
                self.buf[self.avail:self.avail + n] = x[:n]
                with self.cond:
                    self.avail += n
                    self.cond.notify_all()
        except Exception as e:
            self.error = e
        finally:
            self.proc.kill()
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if self.avail == 0 and not self.error:
                self.error = RuntimeError(f"ffmpeg produced no audio: {err[-300:]}")
            with self.cond:
                self.done = True
                self.cond.notify_all()


def probe(src):
    http = ["-user_agent", UA["User-Agent"]] if src.startswith("http") else []
    out = subprocess.run([FFPROBE, "-v", "error", *http, "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", src], capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())  # raises on failure / "N/A"


def pick(L, pcm):
    """Next chunk to encode: a waited-for one if its audio is in, else the first missing (or None: wait)."""
    def has_audio(c):
        return pcm.done or pcm.avail >= L.bounds[c][1] * mimi_file.SPF
    L.wanted = [c for c in L.wanted if not L.ready[c]]
    for c in L.wanted:
        if has_audio(c):
            return c
    for c, r in enumerate(L.ready):
        if not r:
            return c if has_audio(c) else None
    return None


# Chunks are encoded in order as one stream, carrying the encoder's KV and conv padding caches: the codes match a
# single uninterrupted encode (99.99% on 5 min, vs 98.7% for the old windows with 20 s of left context) and it's 27%
# less GPU work, since no context is re-encoded (43.8 min: 17.2 -> 12.6 s on the server; scripts/mimi_streaming_server.py).
# A chunk a reader is waiting for ahead of the stream is encoded at once with 20 s of left context, as before; the
# stream later passes over it to keep its state, without rewriting bytes that may have been served.
MIMI_STREAMING = os.environ.get("MIMI_HUB_MIMI_STREAMING", "1") != "0"
# On a GPU the encoder (SEANet, transformer, downsampling) runs in fp16 and the quantizer in fp32: 1.46x faster on the
# server, the coarse codes nearly all unchanged (first codebook 99.7%, first 8 98.5%, all 32 90.8% on 5 min) and the
# decoded audio as close to the original (log-mel L1 0.6552 vs 0.6549; scripts/mimi_fp16_server.py).
MIMI_FP16 = os.environ.get("MIMI_HUB_MIMI_FP16", "1") != "0"


def mimi_encode(model, x, **kw):
    """model.encode(x, num_quantizers=32, **kw), with the encoder in fp16 on a GPU (MIMI_FP16)."""
    import torch
    with torch.inference_mode():
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=MIMI_FP16 and x.device.type != "cpu"):
            o = model.encoder(x, past_key_values=kw.get("encoder_past_key_values"),
                              padding_cache=kw.get("padding_cache"), use_streaming=kw.get("use_streaming"))
        codes = model.quantizer.encode(o.latents.float(), 32).transpose(0, 1)
    return types.SimpleNamespace(audio_codes=codes, encoder_past_key_values=o.past_key_values,
                                 padding_cache=o.padding_cache)


def encode_live(model, L, pcm, fd, i):
    import numpy as np
    import torch
    spf = mimi_file.SPF
    dev = next(model.parameters()).device
    n = len(L.bounds)

    def has_audio(c):
        return pcm.done or pcm.avail >= L.bounds[c][1] * spf

    def next_job(seq):
        """("stream" | "window", chunk), or None to wait for audio."""
        if not MIMI_STREAMING:
            c = pick(L, pcm)
            return None if c is None else ("window", c)
        L.wanted = [c for c in L.wanted if not L.ready[c]]
        ahead = next((c for c in L.wanted if c != seq and has_audio(c)), None)
        if ahead is not None:
            return "window", ahead
        return ("stream", seq) if has_audio(seq) else None

    seq, kv, pad, written = 0, None, None, 0  # the stream's next chunk and caches
    while not all(L.ready):
        with L.cond:
            while (job := next_job(seq)) is None:
                if pcm.error:
                    raise pcm.error
                L.cond.wait(0.5)
        kind, c = job
        f0, f1 = L.bounds[c]
        if kind == "stream":
            x = torch.from_numpy(pcm.buf[f0 * spf:f1 * spf].copy())[None, None].to(dev)
            o = mimi_encode(model, x, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad, seq = o.encoder_past_key_values, o.padding_cache, seq + 1
            if L.ready[c]:
                continue  # encoded early for a reader: keep its bytes, the pass just advanced the caches
            seg = o.audio_codes[0].T.cpu().numpy()[: f1 - f0]
        else:
            lo = max(0, (f0 - CTX) * spf)
            x = torch.from_numpy(pcm.buf[lo:f1 * spf].copy())[None, None].to(dev)
            codes = mimi_encode(model, x).audio_codes[0].T.cpu().numpy()
            seg = codes[(f0 * spf - lo) // spf:][: f1 - f0]
        if len(seg) < f1 - f0:  # pad (only at the very end)
            seg = np.concatenate([seg, np.repeat(seg[-1:], f1 - f0 - len(seg), axis=0)])
        os.pwrite(fd, mimi_file.pack(seg, 32).tobytes(), L.data_offset + f0 * FRAME_BYTES)
        with L.cond:
            L.ready[c] = True
            L.cond.notify_all()
        _emit("ready", i, c)
        written += 1
        if written % 4 == 0 or written == n:
            update(i, progress=round(written / n, 3))


# Resampling to 16 kHz and Silero VAD (~6 s of serial CPU work on a 43.8 min episode) run as ffmpeg decodes, so
# ASR's batches start right after the download instead. Identical results: soxr's stream resampler gives the same
# samples as resampling the whole (librosa's default, soxr_hq), and the VAD sees the same frames in the same order.
STREAM_VAD = os.environ.get("MIMI_HUB_STREAM_VAD", "1") != "0"
_vad_pre = threading.local()  # (n_samples, probabilities) for the VAD call of this thread's next recognize()


# ...and with STREAM_ASR, ASR's batches start during the download too: VAD segments are final as soon as the audio
# after them is decoded, and every STREAM_WINDOW of them are batched (sorted by length) and recognized at once.
# Same transcripts. Off by default: on the server one episode is GPU-bound too, so ASR done sooner (a 43 min episode:
# 19.7 -> 18.2 s, ad marks 2.6 s sooner) only pushed the .mimi back (13 -> 35 s) and it was ready no sooner (~36 s).
# It would pay off where the GPU has room to spare during a download.
STREAM_ASR = os.environ.get("MIMI_HUB_STREAM_ASR", "0") == "1"
STREAM_WINDOW = int(os.environ.get("MIMI_HUB_STREAM_WINDOW", "64"))

# For one episode on a GPU with room to spare (a cloud H100: the hub's order left it waiting on one CPU thread for
# ~10 of a 43 min episode's 15 s): diarization starts as soon as the audio is resampled (~1 s), on its own CUDA
# stream, alongside the VAD and ASR (streamed, as STREAM_ASR), instead of after them. Same speakers; streamed ASR
# batches segments differently, which moved a few close calls (2 43 min episodes: 3 / 5 of ~900 cues, 99.97% of words
# the same). On the server (GPU-bound) a 43 min upload was ready 26.5 -> 24.5 s.
LATENCY = os.environ.get("MIMI_HUB_LATENCY", "0") == "1"
# Silero VAD in this many worker processes (hub/vad_shard.py; 0: one thread, frame by frame, identical to onnx_asr):
# 4.6 s -> ~0.4 s for a 43 min episode on the server; the transcript of the reference was word for word the same.
# Latency mode's default 16.
VAD_SHARDS = int(os.environ.get("MIMI_HUB_VAD_SHARDS", "16" if LATENCY else "0"))
_vad_pool = None


def _vad_workers():
    global _vad_pool
    if _vad_pool is None and VAD_SHARDS > 1:
        from hub import vad_shard
        _vad_pool = vad_shard.pool(VAD_SHARDS)
        atexit.register(_vad_pool.shutdown, wait=False, cancel_futures=True)
        list(_vad_pool.map(abs, range(VAD_SHARDS)))  # started (and their sessions loaded) now, not by an episode
    return _vad_pool


class _StreamPrep:
    """[pcm] at 16 kHz, and SileroVad._encode's per-frame probabilities of it, as ffmpeg decodes it: one thread
    resamples, another runs the VAD behind it (wav16() is ready as soon as the resampling is, ~1 s for a 43 min
    episode; the VAD, one ONNX call per 32 ms, takes ~5 s more)."""

    END = object()

    def __init__(self, pcm, cond, vad, sr=16000, hop=512, ctx=64):
        import numpy as np
        self.pcm, self.cond, self.vad, self.sr, self.hop, self.ctx = pcm, cond, vad, sr, hop, ctx
        self.w = np.empty(int(len(pcm.buf) * sr / mimi_file.SR) + 1024, np.float32)
        self.n = 0  # samples written
        self.length = None  # the final number of samples, once resampled
        self.probs = []
        self.q = queue.Queue()
        self.grown = threading.Condition()  # w / n / length changed
        self.resampled, self.finished = threading.Event(), threading.Event()
        self.failed = False
        threading.Thread(target=self._resample, daemon=True, name="mimi-resample").start()
        threading.Thread(target=self._vad, daemon=True, name="mimi-vad").start()

    def _resample(self):
        import numpy as np
        import soxr
        try:
            pcm, cond = self.pcm, self.cond
            rs = soxr.ResampleStream(mimi_file.SR, self.sr, 1, dtype="float32", quality="HQ")
            pos = 0
            while True:
                with cond:
                    while pcm.avail - pos < mimi_file.SR and not pcm.done:
                        cond.wait(1.0)
                    done, avail = pcm.done, pcm.avail
                if done and pcm.error:
                    self.failed = True
                    return
                y = rs.resample_chunk(pcm.buf[pos:avail], last=done)
                pos = avail
                with self.grown:
                    if self.n + len(y) > len(self.w):
                        self.w = np.concatenate([self.w, np.empty(self.n + len(y) - len(self.w) + self.sr,
                                                                  np.float32)])
                    self.w[self.n:self.n + len(y)] = y
                    self.n += len(y)
                    if done:
                        self.length = self.n
                    self.grown.notify_all()
                if done:
                    return
        finally:
            with self.grown:
                self.failed = self.failed or self.length is None
                self.grown.notify_all()
            self.resampled.set()

    def _vad(self):
        if _vad_workers() is not None:
            return self._vad_sharded()
        import numpy as np
        try:
            hop, ctx = self.hop, self.ctx
            state = np.zeros((2, 1, 128), dtype=np.float32)
            srs = np.array([self.sr], dtype=np.int64)
            next_frame = -1  # -1: the first (left-padded) frame; k >= 0: w[448 + 512 k:][:576]

            def vad_run(frame):
                nonlocal state
                o, state = self.vad._model.run(["output", "stateN"], {"input": frame[None], "state": state, "sr": srs})
                self.probs.append(o[0, 0])
                self.q.put(o[0, 0])

            while True:
                with self.grown:
                    # frame -1 needs the first hop samples, frame k >= 0 hop x (k + 2)
                    while not self.failed and self.length is None and self.n < hop * (next_frame + 2):
                        self.grown.wait(1.0)
                    if self.failed:
                        return
                    w, n, length = self.w, self.n, self.length
                if next_frame == -1 and n >= hop:
                    vad_run(np.pad(w[:hop], (ctx, 0)))
                    next_frame = 0
                while next_frame >= 0 and hop - ctx + hop * next_frame + ctx + hop <= n:
                    a = hop - ctx + hop * next_frame
                    vad_run(w[a:a + ctx + hop])
                    next_frame += 1
                if length is not None and n == length:
                    break
            if next_frame >= 0 and (last := n % hop):
                vad_run(np.pad(w[n - last - ctx:n], (0, hop - last)))
            # as SileroVad._encode's output; under one frame: let the VAD do it all
            self.probs = np.array(self.probs, dtype=np.float32) if next_frame >= 0 else None
        finally:
            self.q.put(self.END)
            self.finished.set()

    def _vad_sharded(self):
        """As _vad, in hub/vad_shard.py's shards, each sent to a worker as soon as its audio is here."""
        import numpy as np
        from hub import vad_shard as vs
        pending = queue.Queue()  # the shards' futures, in order

        def collect():
            try:
                parts = []
                while (f := pending.get()) is not None:
                    probs = f.result()
                    parts.append(probs)
                    for p in probs:
                        self.q.put(p)
                self.probs = np.concatenate(parts) if parts else None
            except Exception:
                traceback.print_exc()
                self.probs = None
            finally:
                self.q.put(self.END)
                self.finished.set()
        threading.Thread(target=collect, daemon=True, name="mimi-vad-collect").start()
        try:
            i0 = 0
            while True:
                i1 = i0 + vs.SHARD_FRAMES
                with self.grown:  # until this shard's audio is here, or all of it is
                    while not self.failed and self.length is None and self.n < vs.samples_for(i1):
                        self.grown.wait(1.0)
                    if self.failed:
                        return
                    w, n, length = self.w, self.n, self.length
                if length is not None:
                    i1 = min(i1, vs.n_frames(length))
                if i0 >= i1:
                    return
                a = max(0, i0 - vs.WARM_FRAMES)
                pending.put(_vad_pool.submit(vs._run, vs.frames(w, n, a, i1), i0 - a))
                i0 = i1
        finally:
            pending.put(None)

    def wav16(self):
        """The resampled audio, once all of it is (the VAD may still be running)."""
        self.resampled.wait()
        return self.w[:self.n]

    def vad_probs(self):
        self.finished.wait()
        return self.probs

    def stream(self):
        """The probabilities as they come."""
        while (p := self.q.get()) is not self.END:
            yield p


def _merge_stream(vad, segments, length, sample_rate, *, min_speech_duration_ms=250, max_speech_duration_s=20,
                  min_silence_duration_ms=100, speech_pad_ms=30, **kwargs):
    """onnx_asr's BaseVad._merge_segments (0.12) for a waveform still being decoded: length() is its final length,
    only asked for once [segments] has ended (until then no segment can reach the end)."""
    from itertools import chain
    INF = vad.INF
    speech_pad = int(speech_pad_ms * sample_rate // 1000)
    min_speech_duration = int(min_speech_duration_ms * sample_rate // 1000) - 2 * speech_pad
    max_speech_duration = int(max_speech_duration_s * sample_rate) - 2 * speech_pad
    min_silence_duration = int(min_silence_duration_ms * sample_rate // 1000) + 2 * speech_pad

    final = [INF]  # the waveform's length: no segment before the end markers reaches it

    def tail():
        final[0] = length()
        yield final[0], final[0]
        yield INF, INF

    cur_start, cur_end = -INF, -INF
    for start, end in chain(segments, tail()):
        n = final[0]
        if start - cur_end < min_silence_duration and end - cur_start < max_speech_duration:
            cur_end = end
        else:
            if cur_start < n and cur_end > cur_start and cur_end - cur_start > min_speech_duration:
                yield max(cur_start - speech_pad, 0), min(cur_end + speech_pad, n)
            while end - start > max_speech_duration:
                yield max(start - speech_pad, 0), start + max_speech_duration + speech_pad
                start += max_speech_duration
            cur_start, cur_end = start, end


def _stream_recognize(adapter, prep):
    """ASR of [prep]'s audio as it decodes: TimestampedSegmentResults in time order (as adapter.recognize)."""
    vad, asr, kw = adapter.vad, adapter.asr, adapter._vadargs
    segments = _merge_stream(vad, vad._find_segments(prep.stream(), prep.hop, **kw), lambda: prep.length, prep.sr,
                             **kw)

    def batches():
        pending = []
        for seg in segments:
            pending.append(seg)
            if len(pending) >= STREAM_WINDOW:
                pending.sort(key=lambda x: x[1] - x[0])
                yield from (pending[k:k + ASR_BUCKET] for k in range(0, len(pending), ASR_BUCKET))
                pending = []
        pending.sort(key=lambda x: x[1] - x[0])
        yield from (pending[k:k + ASR_BUCKET] for k in range(0, len(pending), ASR_BUCKET))

    results = dict(_asr_batches(asr, batches(), lambda s, e: prep.w[s:e], {"need_logprobs": True}, prep.sr))
    return [results[seg] for seg in sorted(results)]


def transcribe(m, pcm, cond, out, on_cues=None):
    """ASR and diarization of the full decoded episode, concurrently; results (or "error") go into [out].
    [on_cues] gets the ASR cues as soon as they exist, before diarization is done."""
    try:
        import librosa
        mark = out.get("mark", lambda k: None)
        prep = _StreamPrep(pcm, cond, m["asr"].vad) if STREAM_VAD else None
        streamed = None
        diar_on_gpu = m.get("device", "cpu") != "cpu" and DIARIZER in ("auto", "pyannote")
        if prep is not None and LATENCY:
            wav16 = prep.wav16()
            if pcm.error or prep.failed:
                return  # the encoder reports it
            mark("decoded")  # resampled (the VAD still running)

            def speakers_early():
                import torch
                try:
                    with torch.cuda.stream(torch.cuda.Stream()) if diar_on_gpu else contextlib.nullcontext():
                        out["turns"], out["diarization"] = diarize(wav16)
                except Exception:
                    traceback.print_exc()
                    out["turns"], out["diarization"] = [], None
            d = threading.Thread(target=speakers_early, daemon=True, name="mimi-diarize")
            d.start()
            out["cues"] = [c for s in _stream_recognize(m["asr"], prep) for c in cues(s)]
            mark("asr")
            if on_cues:
                on_cues(out["cues"])
            d.join()
            mark("diar")
            return
        if prep is not None and STREAM_ASR:
            streamed = _stream_recognize(m["asr"], prep)  # done shortly after the download
        if prep is not None:
            wav16, probs = prep.wav16(), prep.vad_probs()
        else:
            with cond:
                while not pcm.done:
                    cond.wait(1.0)
        if pcm.error:
            return  # the encoder reports it
        if prep is None:
            wav16, probs = librosa.resample(pcm.buf[:pcm.avail], orig_sr=mimi_file.SR, target_sr=16000), None
        mark("decoded")  # and, with STREAM_VAD, resampled and VAD'd (and with STREAM_ASR, recognized)

        def speakers():
            try:
                out["turns"], out["diarization"] = diarize(wav16)
            except Exception:
                traceback.print_exc()  # a transcript without speakers is still useful
                out["turns"], out["diarization"] = [], None

        # pyannote on the GPU waits for ASR: the GPU is the bottleneck either way (the stages' times add up), and
        # this way the cues, and so ad detection, are ready sooner. Even for one episode alone: run alongside ASR,
        # a 43 min episode was ready 3 s sooner but its ad marks 14 s later and its .mimi 19 s later
        # (scripts/episode_timeline.py). sherpa (CPU) runs alongside.
        d = threading.Thread(target=speakers, daemon=True, name="mimi-diarize")
        if not diar_on_gpu:
            d.start()
        if streamed is not None and probs is not None:
            out["cues"] = [c for s in streamed for c in cues(s)]
        else:
            _vad_pre.value = (len(wav16), probs) if probs is not None else None
            try:
                out["cues"] = [c for s in m["asr"].recognize(wav16, sample_rate=16000) for c in cues(s)]
            finally:
                _vad_pre.value = None
        mark("asr")
        if on_cues:
            on_cues(out["cues"])
        if diar_on_gpu:
            d.start()
        d.join()
        mark("diar")
    except Exception as e:
        out["error"] = e


def run(i):
    r = load(i)
    upload = r.get("upload")
    L = None
    t0, tm = time.time(), {}

    def mark(stage):
        """Seconds from the job's start to the end of [stage], kept in the record's "timing"."""
        tm[stage] = round(time.time() - t0, 2)

    update(i, timing_start=t0, timing={})
    try:
        if not (FFMPEG and FFPROBE):
            raise RuntimeError("ffmpeg/ffprobe not on PATH (start the hub with hub/run.sh)")
        m = models()
        update(i, state="downloading", progress=0.0)
        if upload:
            src, dur = upload, probe(upload)
        elif r.get("keep_source"):  # process exactly the bytes the client will get
            await_source_fetch(i)
            r = load(i)
            kept = DATA / f"{i}.src"
            src = str(kept) if r.get("source_ready") and kept.exists() else get_source(i, r)
            dur = probe(src)
        else:
            src, dur, err = None, None, None
            for u in candidates(r["audio_url"]):  # skip tracking hops that are unreachable
                try:
                    src, dur = u, probe(u)
                    break
                except Exception as e:
                    err = e
            if src is None or dur is None:
                raise RuntimeError(f"cannot read source: {err}")
        mark("probed")
        n_frames = int(dur * FPS) + 13  # + ~1 s: ffprobe estimates can run short; the tail is padded
        # (a job queued while the hub compressed, resumed after a restart with MIMI_HUB_COMPRESS=0: not compressed now,
        # as there's no Mimi loaded)
        compress = COMPRESS and r.get("compressed", True)
        # transcription and diarization start once the whole episode is decoded (ffmpeg is far ahead of the
        # encoder) and run alongside the encode
        tx = {"mark": mark}
        # ad detection starts from the ASR cues, alongside diarization: speaker labels add little to it (episodes
        # without any were found as well), and diarization takes as long as everything else
        early = lambda cs: queue_segments(i, [(a, b, None, text) for a, b, text in cs])  # noqa: E731
        if compress:
            head, _ = mimi_file.header(n_frames, 32, dict(title=r.get("title", ""), source=r.get("url", ""), id=i,
                                                          encoder="transformers MimiModel", duration_s=dur))
            path = DATA / f"{i}.mimi"
            with open(path, "wb") as f:
                f.write(head)
                f.truncate(len(head) + n_frames * FRAME_BYTES)
            L = Live(i, n_frames, head)
            live[i] = L
            _emit("live", i, n_frames, head)
            update(i, state="encoding", progress=0.0, duration_s=dur, mimi_bytes=L.total)
            cond = L.cond
        else:
            cond = threading.Condition()
            update(i, state="transcribing", progress=0.0, duration_s=dur, compressed=False)
        pcm = Pcm(src, n_frames * mimi_file.SPF, cond)
        t = threading.Thread(target=transcribe, args=(m, pcm, cond, tx, early), daemon=True,
                             name=f"mimi-asr-{i}")
        t.start()
        if compress:
            fd = os.open(path, os.O_WRONLY)
            try:
                encode_live(m["mimi"], L, pcm, fd, i)
            finally:
                os.close(fd)
            live.pop(i, None)  # complete: served as a plain file from now on
            _emit("unlive", i, False)
            mark("mimi")
            update(i, state="transcribing", progress=0.0)
        t.join()
        if "error" in tx:
            raise tx["error"]
        if "cues" not in tx:
            raise pcm.error or RuntimeError("transcription did not run")
        all_cues, turns = tx["cues"], tx["turns"]
        write_vtt(DATA / f"{i}.vtt", all_cues, turns)
        mark("ready")
        with lock:  # merged: ad detection may have added its time already
            rec = load(i)
            save(dict(rec, state="ready", progress=1.0, asr=ASR_MODEL, device=m["device"],
                      speakers=len({t[2] for t in turns}), diarization=tx["diarization"] if turns else None,
                      timing={**rec.get("timing", {}), **tm}))
        with lock:
            rec = load(i)
            if rec.get("keep_source") and not rec.get("source_ready"):  # asked for while this run was going
                save(dict(rec, state="queued", progress=0.0))
                _requeue(i)
    except Exception as e:
        traceback.print_exc()
        if L is not None:
            with L.cond:
                L.failed = True
                L.cond.notify_all()
        live.pop(i, None)
        _emit("unlive", i, True)
        update(i, state="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if upload and os.path.exists(upload):
            os.unlink(upload)
        if SOURCES != "keep":  # fetch mode: the original was only needed to process; .src fetches it again
            (DATA / f"{i}.src").unlink(missing_ok=True)


# ---------- ad / self-promotion segments (an LLM over the transcript; one at a time, after "ready") ----------

seg_jobs: "queue.Queue" = queue.Queue()


_client_keys = {}  # episode id -> the OpenRouter key its client sent (X-OpenRouter-Key), until its ads are found


def _take_key(i, key):
    if key and key.strip():
        _client_keys[i] = key.strip()


def queue_segments(i, cues=None):
    """Detect [i]'s ads / self-promotion from [cues] ([(start_s, end_s, speaker or None, text)]), else its VTT."""
    if segments.SEGMENTER.strip().lower() in ("", "off") or (load(i) or {}).get("no_segments"):
        key = _client_keys.pop(i, None)
        update(i, segments_state="off")
        if not (load(i) or {}).get("no_segments"):  # (no ad detection here: a summary of the whole transcript)
            queue_summary(i, key, [], cues)
        return
    update(i, segments_state="pending")
    if _events is not None:  # in a worker process: the main process runs the (one) segment worker
        _emit("segments", i, cues)
        return
    seg_jobs.put((i, cues))


_active = [0]  # conversions and ad detections in progress (busy())
_active_lock = threading.Lock()


@contextlib.contextmanager
def _working():
    with _active_lock:
        _active[0] += 1
    try:
        yield
    finally:
        with _active_lock:
            _active[0] -= 1


# ---------- executive summaries (MIMI_HUB_SUMMARIZER: a request of their own after the ads, at low priority) ----------

sum_jobs: "queue.Queue" = queue.Queue()


def queue_summary(i, key=None, found=None, cues=None):
    """[i]'s summary, after its ads ([found]; else its .segments.json) are known. [key]: the client's OpenRouter key,
    kept in memory until it's made (for the fallback after the claude login)."""
    if not segments.SUMMARIES:
        return
    update(i, summary_state="pending", summary_error=None)
    sum_jobs.put((i, key, found, cues))


def summary_worker():
    while True:
        i, key, found, cues = sum_jobs.get()
        with _working():
            try:
                cues = cues or segments.read_cues((DATA / f"{i}.vtt").read_text())
                if found is None and (DATA / f"{i}.segments.json").exists():
                    found = json.loads((DATA / f"{i}.segments.json").read_text())["segments"]
                res = segments.summarize(cues, found, log=lambda m: print(f"[{i}] {m}", flush=True),
                                         keys={"OPENROUTER_API_KEY": key} if key else None)
                tmp = DATA / f"{i}.summary.json.tmp"
                tmp.write_text(json.dumps(res, indent=1))
                tmp.replace(DATA / f"{i}.summary.json")
                update(i, summary_state="ready", summary_model=res["model"], summary_error=None)
            except Exception as e:
                traceback.print_exc()
                update(i, summary_state="failed", summary_error=f"{type(e).__name__}: {e}"[:500])


def busy():
    """Anything converting, detecting ads, warming up or waiting to, or a client that has just woken the server
    (GET /v1/hold)."""
    return (_active[0] > 0 or not jobs.empty() or not seg_jobs.empty() or not sum_jobs.empty()
            or _warm["models"] == "loading"
            or time.time() < _linger_until[0])


def segment_worker():
    while True:
        i, cues = seg_jobs.get()
        with _working():
            _segment(i, cues)


def _segment(i, cues):
    try:
        if cues is None:
            cues = segments.read_cues((DATA / f"{i}.vtt").read_text())
        key = _client_keys.pop(i, None)
        res = segments.detect(cues, log=lambda m: print(f"[{i}] {m}", flush=True),
                              keys={"OPENROUTER_API_KEY": key} if key else None)
        queue_summary(i, key, res["segments"], cues)
        tmp = DATA / f"{i}.segments.json.tmp"
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(DATA / f"{i}.segments.json")
        with lock:
            r = load(i)
            timing = dict(r.get("timing") or {})
            if r.get("timing_start"):
                timing["segments"] = round(time.time() - r["timing_start"], 2)
            save(dict(r, segments_state="ready", segments=len(res["segments"]), segments_model=res["model"],
                      skippable_s=round(sum(s["end"] - s["start"] for s in res["segments"]), 1), timing=timing))
    except Exception as e:
        traceback.print_exc()
        update(i, segments_state="failed", segments_error=f"{type(e).__name__}: {e}"[:500])


_seq = itertools.count()


def _requeue(i):
    if _events is not None:
        _emit("requeue", i)
    else:
        jobs.put((1, next(_seq), i))


def _take_job():
    """The next queued job's id (highest priority first), marked starting."""
    while True:
        prio, _, i = jobs.get()
        with lock:
            r = load(i)
            if not r or r["state"] != "queued":
                continue  # duplicate entry (re-queued at a higher priority) or already handled
            if _for_remote(prio, r):  # the remote GPU's (MIMI_HUB_REMOTE_WHEN always / priority)
                jobs.put((prio, next(_seq), i))
                claimed = True
            else:
                save(dict(r, state="starting"))
                claimed = False
        if claimed:
            time.sleep(0.5)
            continue
        return i


def _for_remote(prio, r):
    return (bool(REMOTE) and (REMOTE_WHEN == "always" or (REMOTE_WHEN == "priority" and prio == 0))
            and bool(_remote_ok(r)))


_local_jobs = [0]  # local workers running a job (the remote takes jobs when they're all busy)


def worker():
    while True:
        i = _take_job()
        with _active_lock:
            _local_jobs[0] += 1
        try:
            with _working():
                run(i)
        finally:
            with _active_lock:
                _local_jobs[0] -= 1


# ---------- a remote GPU (MIMI_HUB_REMOTE: another hub, e.g. hub/modal_app.py on Modal) ----------

REMOTE = os.environ.get("MIMI_HUB_REMOTE", "").strip().rstrip("/")  # with /t/<token> if it has one, or:
if REMOTE and os.environ.get("MIMI_HUB_REMOTE_TOKEN_FILE"):  # its token from a file (not in the settings)
    REMOTE += "/t/" + Path(os.environ["MIMI_HUB_REMOTE_TOKEN_FILE"]).expanduser().read_text().strip()
REMOTE_WHEN = os.environ.get("MIMI_HUB_REMOTE_WHEN", "busy")  # busy | priority | always | off
REMOTE_JOBS = int(os.environ.get("MIMI_HUB_REMOTE_JOBS", "2"))  # at once (Modal's hub: one container, 2 workers)
# POST /v1/warm passed on to the remote: by default only when it takes jobs a client may be waiting for
REMOTE_WARM = os.environ.get("MIMI_HUB_REMOTE_WARM", "1" if REMOTE_WHEN in ("priority", "always") else "0") != "0"
_remote_jobs = [0]


def _remote_ok(r):
    """Jobs the remote can do: an episode by URL, compressed (uploads and kept sources stay here)."""
    return (r and not r.get("upload") and not r.get("keep_source") and COMPRESS and r.get("compressed", True)
            and not r.get("remote_failed"))


def _remote_wants():
    if not REMOTE or REMOTE_WHEN == "off" or _remote_jobs[0] >= REMOTE_JOBS:
        return False
    with jobs.mutex:
        head = jobs.queue[0] if jobs.queue else None
    if head is None:
        return False
    return (REMOTE_WHEN == "always" or (REMOTE_WHEN == "priority" and head[0] == 0)
            or _local_jobs[0] >= WORKERS)


def _remote_open(path, body=None, headers=None, timeout=60):
    """urllib (not httpx: its versions' APIs differ between the hub's environments)."""
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json", **UA, **(headers or {})}
    return urllib.request.urlopen(urllib.request.Request(REMOTE + path, data=data, headers=h), timeout=timeout)


def _remote(method, path, body=None, timeout=60):
    try:
        with _remote_open(path, {} if method == "POST" and body is None else body, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def remote_worker():
    while True:
        if not _remote_wants():
            time.sleep(0.5)
            continue
        try:
            prio, _, i = jobs.get(timeout=1)
        except queue.Empty:
            continue
        with lock:
            r = load(i)
            if not r or r["state"] != "queued":
                continue
            if not _remote_ok(r):  # back for a local worker
                jobs.put((prio, next(_seq), i))
                time.sleep(1)
                continue
            save(dict(r, state="starting", remote=REMOTE.split("/t/")[0]))
        with _active_lock:
            _remote_jobs[0] += 1
        try:
            with _working():
                run_remote(i)
        finally:
            with _active_lock:
                _remote_jobs[0] -= 1


def run_remote(i):
    """Job [i] converted by the remote and copied here, then its ads found here; redone locally if that fails."""
    r = load(i)
    t0 = time.time()
    update(i, timing_start=t0, timing={})
    try:
        _remote("POST", "/v1/convert", {"url": r.get("audio_url") or r["url"], "id": i, "title": r.get("title"),
                                        "priority": r.get("priority", 1), "segments": False})
        last = None
        while True:
            s = _remote("GET", f"/v1/episodes/{i}")
            if s is None:
                raise RuntimeError("the remote lost the job")
            if s["state"] == "failed":
                raise RuntimeError(f"remote: {s.get('error')}")
            if s["state"] == "ready":
                break
            mirror = (s["state"], s.get("progress"))
            if mirror != last:  # its progress, for clients polling this hub
                fields = {k: s[k] for k in ("duration_s", "mimi_bytes") if s.get(k)}
                update(i, state=s["state"] if s["state"] != "queued" else "starting", progress=s.get("progress", 0.0),
                       remote_state=s["state"], **fields)
                last = mirror
            time.sleep(1)
        for ext in ("vtt", "mimi"):  # the .mimi last: once it's a plain file here, it's served from here
            tmp = DATA / f"{i}.{ext}.remote"
            with _remote_open(f"/v1/episodes/{i}.{ext}", timeout=300) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            tmp.replace(DATA / f"{i}.{ext}")
        with lock:
            rec = load(i)
            save(dict(rec, state="ready", progress=1.0, remote_state="ready", asr=s.get("asr"),
                      device=f"remote: {s.get('device')}", duration_s=s.get("duration_s"),
                      mimi_bytes=(DATA / f"{i}.mimi").stat().st_size, speakers=s.get("speakers"),
                      diarization=s.get("diarization"), compressed=True,
                      timing={**(s.get("timing") or {}), "copied": round(time.time() - t0, 2)}))
        queue_segments(i)
    except Exception as e:
        traceback.print_exc()
        for ext in ("vtt", "mimi"):
            (DATA / f"{i}.{ext}.remote").unlink(missing_ok=True)
        print(f"[{i}] remote failed ({type(e).__name__}: {e}); converting here", flush=True)
        with lock:
            rec = load(i)
            save(dict(rec, state="queued", progress=0.0, remote_failed=True, remote_error=f"{type(e).__name__}: {e}"[:300]))
        jobs.put((rec.get("priority", 1), next(_seq), i))


def proxy_remote(i, rng):
    """The remote's .mimi for job [i] while it's being made there, passed through (Range and all)."""
    try:
        resp = _remote_open(f"/v1/episodes/{i}.mimi", headers={"Range": rng} if rng else None, timeout=120)
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, "remote: not ready") from None
    keep = ("content-range", "content-length", "accept-ranges", "content-type")
    headers = {k: v for k, v in resp.headers.items() if k.lower() in keep}

    def body():
        with resp:
            while chunk := resp.read1(1 << 16):
                yield chunk
    return StreamingResponse(body(), status_code=resp.status, headers=headers)


# ---------- worker processes (MIMI_HUB_PROCS > 1) ----------
#
# One Python process can't keep a fast GPU busy: Python and kernel launches are serial. With PROCS > 1, jobs run in
# that many worker processes, each with its own models and WORKERS jobs at a time; the main process keeps HTTP, the
# job queue and ad detection. A conversion's Live state is mirrored in the main process from the worker's events
# (header, chunk ready, done), and readers' wanted chunks are sent to the worker. Modal L40S, 8 x 5 min backlog:
# 1 process 170x, 6 processes 441x; an M5 MacBook Pro 67 -> 77x with 2 (scripts/throughput_server.py --procs).
PROCS = int(os.environ.get("MIMI_HUB_PROCS", "1"))
_events = None  # in a worker process: queue of events for the main process


def _emit(*event):
    if _events is not None:
        _events.put(event)


def _proc_main(commands, events):
    """A worker process: runs the jobs it's sent, reports Live progress, takes readers' wanted chunks."""
    global _events
    _events = events
    while True:
        cmd = commands.get()
        if cmd[0] == "run":
            def go(i=cmd[1]):
                try:
                    run(i)
                finally:
                    _emit("done", i)
            threading.Thread(target=go, daemon=True, name=f"mimi-job-{cmd[1]}").start()
        elif cmd[0] == "want":
            _, i, c = cmd
            L = live.get(i)
            if L is not None:
                with L.cond:
                    if not L.ready[c]:
                        if c in L.wanted:
                            L.wanted.remove(c)
                        L.wanted.insert(0, c)
                        L.cond.notify_all()


class _RemoteLive(Live):
    """The main process's mirror of a conversion running in a worker process."""

    def __init__(self, i, n_frames, header, commands):
        super().__init__(i, n_frames, header)
        self.commands = commands

    def wait_ready(self, c, timeout=300):
        if not self.ready[c]:
            self.commands.put(("want", self.i, c))
        return super().wait_ready(c, timeout)


class _Procs:
    def __init__(self, n):
        import multiprocessing
        self.ctx = multiprocessing.get_context("spawn")
        self.events = self.ctx.Queue()
        self.done = {}  # job id -> threading.Event
        self.owner = {}  # job id -> worker process index
        self.slots = []
        for k in range(n):
            self.slots.append(None)
            self._spawn(k)
        threading.Thread(target=self._listen, daemon=True, name="mimi-proc-events").start()

    def _spawn(self, k):
        commands = self.ctx.Queue()
        p = self.ctx.Process(target=_proc_main, args=(commands, self.events), daemon=True, name=f"mimi-proc-{k}")
        p.start()
        self.slots[k] = (p, commands)

    def _listen(self):
        while True:
            kind, i, *rest = self.events.get()
            if kind == "live":
                k = self.owner.get(i)
                live[i] = _RemoteLive(i, rest[0], rest[1], self.slots[k][1] if k is not None else self.ctx.Queue())
            elif kind == "ready":
                L = live.get(i)
                if L is not None:
                    with L.cond:
                        L.ready[rest[0]] = True
                        L.cond.notify_all()
            elif kind == "unlive":
                L = live.pop(i, None)
                if L is not None and rest[0]:
                    with L.cond:
                        L.failed = True
                        L.cond.notify_all()
            elif kind == "segments":
                queue_segments(i, rest[0])
            elif kind == "requeue":
                jobs.put((1, next(_seq), i))
            elif kind == "done" and i in self.done:
                self.done[i].set()

    def dispatcher(self, k):
        while True:
            i = _take_job()
            p, commands = self.slots[k]
            if not p.is_alive():
                self._spawn(k)
                p, commands = self.slots[k]
            self.owner[i], self.done[i] = k, threading.Event()
            commands.put(("run", i))
            while not self.done[i].wait(5):
                if not p.is_alive():  # the process died with the job (e.g. out of memory): fail it, restart
                    L = live.pop(i, None)
                    if L is not None:
                        with L.cond:
                            L.failed = True
                            L.cond.notify_all()
                    update(i, state="failed", error=f"worker process exited ({p.exitcode})")
                    self._spawn(k)
                    break
            self.done.pop(i, None)
            self.owner.pop(i, None)


_procs = None


@app.on_event("startup")
def start():
    for p in sorted(record_paths(), key=lambda p: p.stat().st_mtime):
        r = json.loads(p.read_text())
        if r["state"] not in ("ready", "failed"):  # resume interrupted jobs from scratch
            (DATA / f"{r['id']}.mimi").unlink(missing_ok=True)  # partial, preallocated: never serve it
            save(dict(r, state="queued", progress=0.0, source_fetching=None))
            jobs.put((r.get("priority", 1), next(_seq), r["id"]))
            if r.get("priority", 1) == 0:
                queue_source(r["id"])
        elif r["state"] == "ready" and r.get("segments_state") not in ("ready", "failed", "off"):
            queue_segments(r["id"])  # converted before segment detection existed, or interrupted
        elif r["state"] == "ready" and r.get("summary_state") == "pending":
            queue_summary(r["id"])  # interrupted (episodes from before summaries: POST /v1/episodes/{id}/summary)
    global _procs
    if PROCS > 1:
        _procs = _Procs(PROCS)
        for k in range(PROCS):
            for w in range(WORKERS):
                threading.Thread(target=_procs.dispatcher, args=(k,), daemon=True,
                                 name=f"mimi-dispatch-{k}-{w}").start()
    else:
        for k in range(WORKERS):
            threading.Thread(target=worker, daemon=True, name=f"mimi-worker-{k}").start()
    if REMOTE and REMOTE_WHEN != "off":
        for k in range(REMOTE_JOBS):
            threading.Thread(target=remote_worker, daemon=True, name=f"mimi-remote-{k}").start()
    threading.Thread(target=segment_worker, daemon=True, name="mimi-segments").start()
    threading.Thread(target=summary_worker, daemon=True, name="mimi-summaries").start()
    from hub import backlog
    backlog.start()
    for k in range(2):
        threading.Thread(target=source_worker, daemon=True, name=f"mimi-source-{k}").start()
    if MAX_BYTES:
        threading.Thread(target=trim_worker, daemon=True, name="mimi-trim").start()
    if PRELOAD:
        threading.Thread(target=warm, daemon=True, name="mimi-warm").start()


@app.on_event("shutdown")
def stop():
    if _diar_pool is not None:
        _diar_pool.shutdown(wait=False, cancel_futures=True)


# ---------- HTTP ----------

class _TokenPath:
    """/t/<MIMI_HUB_TOKEN>/... served as /... with the token's Authorization header."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if TOKEN and scope["type"] == "http" and scope["path"].startswith("/t/"):
            given, _, rest = scope["path"][3:].partition("/")
            if hmac.compare_digest(given, TOKEN):
                path = "/" + rest
                headers = [h for h in scope["headers"] if h[0] != b"authorization"]
                scope = dict(scope, path=path, raw_path=path.encode(),
                             headers=[*headers, (b"authorization", f"Bearer {TOKEN}".encode())])
        await self.inner(scope, receive, send)


app.add_middleware(_TokenPath)


def auth(authorization):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


def check_id(i):
    if not ID_RE.match(i):
        raise HTTPException(400, "bad id")


class ConvertRequest(BaseModel):
    url: str
    id: str | None = None
    title: str | None = None
    priority: int = 1  # 0: a client is about to stream it
    keep_source: bool = False  # also keep and serve the original audio (GET .src)
    segments: bool = True  # False: no ad detection here (a hub using this one as its remote GPU does its own)


@app.post("/v1/convert", status_code=202)
def convert(req: ConvertRequest, authorization: str | None = Header(None),
            x_openrouter_key: str | None = Header(None)):
    auth(authorization)
    try:
        i, audio, title = resolve(req.url.strip())
    except Exception as e:
        raise HTTPException(422, f"cannot resolve URL: {e}")
    if req.id:
        check_id(req.id)
        i = req.id
    _take_key(i, x_openrouter_key)
    extra = {} if req.segments else {"no_segments": True}
    r = enqueue(i, priority=min(1, max(0, req.priority)), keep_source=req.keep_source, url=req.url,
                audio_url=audio, title=req.title or title, **extra)
    return _retry_segments(r, x_openrouter_key)


def _retry_segments(r, key):
    """An episode whose ad detection failed, now with a key: run it again."""
    if key and r and r.get("state") == "ready" and r.get("segments_state") == "failed":
        _take_key(r["id"], key)
        queue_segments(r["id"])
        return load(r["id"])
    if r and (r.get("state") == "ready" and r.get("segments_state") != "pending"):
        _client_keys.pop(r["id"], None)  # nothing will use it
    return r


@app.post("/v1/upload", status_code=202)
def upload(file: UploadFile = File(...), title: str = "", authorization: str | None = Header(None),
           x_openrouter_key: str | None = Header(None)):
    auth(authorization)
    h = hashlib.sha1()
    fd, tmp = tempfile.mkstemp(prefix="mimi-upload-", dir=DATA)
    with os.fdopen(fd, "wb") as f:
        while chunk := file.file.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    i = "u" + h.hexdigest()[:15]
    _take_key(i, x_openrouter_key)
    r = enqueue(i, url="upload:" + (file.filename or ""), upload=tmp, title=title or file.filename or i)
    if r.get("upload") != tmp:
        os.unlink(tmp)  # duplicate of an existing job
    return _retry_segments(r, x_openrouter_key)


@app.post("/v1/warm", status_code=202)
def warm_up(authorization: str | None = Header(None)):
    auth(authorization)
    _linger_until[0] = max(_linger_until[0], time.time() + WARM_LINGER_S)
    if _warm["models"] in ("cold", "failed"):
        threading.Thread(target=warm, daemon=True, name="mimi-warm").start()
    if REMOTE and REMOTE_WARM:  # (fire and forget: it may be cold-starting)
        threading.Thread(target=lambda: _remote("POST", "/v1/warm", {}, timeout=90), daemon=True).start()
    return {"models": "loading" if _warm["models"] == "cold" else _warm["models"]}


@app.get("/v1/hold")
async def hold(s: float = 60, authorization: str | None = Header(None)):
    # async, so it can be cancelled: Modal cancels a request whose client is gone and kills the container if the
    # handler doesn't stop within 30 s (a thread sleeping in a sync handler doesn't)
    import asyncio
    auth(authorization)
    end = time.time() + min(max(s, 0), 3600)
    while busy() and time.time() < end:
        await asyncio.sleep(1)
    return {"busy": busy()}


@app.post("/v1/episodes/{i}/summary", status_code=202)
def make_summary(i: str, authorization: str | None = Header(None), x_openrouter_key: str | None = Header(None)):
    auth(authorization)
    check_id(i)
    if not segments.SUMMARIES:
        raise HTTPException(409, "summaries are off on this hub (MIMI_HUB_SUMMARIZER)")
    r = load(i)
    if not r or r["state"] != "ready" or r.get("segments_state") == "pending":
        raise HTTPException(409, "episode not ready")
    queue_summary(i, (x_openrouter_key or "").strip() or None)
    return load(i)


@app.get("/v1/features")
def features(authorization: str | None = Header(None)):
    auth(authorization)
    from hub import backlog
    return {"compress": COMPRESS, "segments": segments.SEGMENTER.strip().lower() not in ("", "off"),
            "summaries": segments.SUMMARIES, "remote": bool(REMOTE) and REMOTE_WHEN != "off",
            "backlog": True, "backlog_modal": backlog.modal_available()}


class BacklogEpisode(BaseModel):
    id: str
    url: str
    title: str | None = None
    duration_s: float | None = None


class BacklogRequest(BaseModel):
    podcast_uuid: str | None = None
    episodes: list[BacklogEpisode] | None = None  # (instead of a podcast's)
    latest: int | None = None
    after: str | None = None  # YYYY-MM-DD, published on or after
    before: str | None = None  # published before
    title: str | None = None
    preview: bool = False


@app.post("/v1/backlog")
def start_backlog(req: BacklogRequest, authorization: str | None = Header(None)):
    from hub import backlog
    auth(authorization)
    if req.episodes:
        for e in req.episodes:
            check_id(e.id)
        n = len(req.episodes)
        eps, title = [e.model_dump() for e in req.episodes], req.title or f"{n} episode{'' if n == 1 else 's'}"
    elif req.podcast_uuid:
        if not re.fullmatch(r"[0-9a-f-]{36}", req.podcast_uuid):
            raise HTTPException(400, "bad podcast_uuid")
        try:
            show, eps = backlog.podcast_episodes(req.podcast_uuid)
        except Exception as e:
            raise HTTPException(502, f"cannot list the podcast's episodes: {e}")
        for d in (req.after, req.before):
            if d and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                raise HTTPException(400, "dates are YYYY-MM-DD")
        eps, title = backlog.select(eps, req.latest, req.after, req.before), req.title or show
    else:
        raise HTTPException(400, "give podcast_uuid or episodes")
    if not eps:
        raise HTTPException(422, "no episodes in that range")
    if req.preview:
        return {"title": title, **backlog.plan(eps)}
    return JSONResponse(backlog.public(backlog.create(title, eps, req.podcast_uuid)), status_code=202)


@app.get("/v1/backlogs")
def list_backlogs(podcast_uuid: str | None = None, authorization: str | None = Header(None)):
    from hub import backlog
    auth(authorization)
    return [backlog.public(j) for j in backlog.jobs(podcast_uuid)]


@app.get("/v1/backlogs/{job_id}")
def backlog_status(job_id: str, authorization: str | None = Header(None)):
    from hub import backlog
    auth(authorization)
    job = backlog.get(job_id)
    if not job:
        raise HTTPException(404, "no such backlog")
    return backlog.public(job)


@app.post("/v1/backlogs/{job_id}/cancel")
def cancel_backlog(job_id: str, authorization: str | None = Header(None)):
    from hub import backlog
    auth(authorization)
    job = backlog.cancel(job_id)
    if not job:
        raise HTTPException(404, "no such backlog")
    return backlog.public(job)


@app.post("/v1/episodes/{i}/segments", status_code=202)
def redo_segments(i: str, authorization: str | None = Header(None), x_openrouter_key: str | None = Header(None)):
    auth(authorization)
    check_id(i)
    r = load(i)
    if not r or r["state"] != "ready":
        raise HTTPException(409, "episode not ready")
    _take_key(i, x_openrouter_key)
    queue_segments(i)
    return load(i)


@app.get("/v1/episodes")
def list_jobs(authorization: str | None = Header(None)):
    auth(authorization)
    rs = [json.loads(p.read_text()) for p in record_paths()]
    return sorted(rs, key=lambda r: -r.get("created", 0))


def proxy_source(r, rng):
    """The original from its publisher (MIMI_HUB_SOURCES=fetch), passed through, if it's still the file that was
    processed: same size (dynamically inserted ads change it). Otherwise 409, and the client uses the publisher's
    own copy without the hub's ad marks."""
    req = urllib.request.Request(r["source_url"], headers={**UA, **({"Range": rng} if rng else {})})
    try:
        up = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        raise HTTPException(502 if e.code != 416 else 416, f"publisher: HTTP {e.code}")
    except OSError as e:
        raise HTTPException(502, f"publisher unreachable: {e}")
    if up.status == 206:
        total = int((up.headers.get("Content-Range") or "/-1").rsplit("/", 1)[1])
    else:
        total = int(up.headers.get("Content-Length") or -1)
    if total != r.get("source_bytes"):
        up.close()
        raise HTTPException(409, f"the publisher now serves a different file ({total} vs {r.get('source_bytes')} "
                                 "bytes): its timings may not match this transcript")
    headers = {"Accept-Ranges": "bytes"}
    for h in ("Content-Length", "Content-Range"):
        if up.headers.get(h):
            headers[h] = up.headers[h]

    def body():
        with up:
            while chunk := up.read(1 << 16):
                yield chunk

    return StreamingResponse(body(), status_code=up.status, media_type=r.get("source_type") or "audio/mpeg",
                             headers=headers)


def stream_live(L, rng):
    """Serve (a Range of) a .mimi that is still being encoded, waiting for each chunk as needed."""
    start, end, status = 0, L.total - 1, 200
    m = re.match(r"bytes=(\d*)-(\d*)$", (rng or "").strip())
    if m and (m.group(1) or m.group(2)):
        a, b = m.groups()
        if a == "":
            start = max(0, L.total - int(b))
        else:
            start, end = int(a), min(int(b), end) if b else end
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{L.total}"})
        status = 206
    path = DATA / f"{L.i}.mimi"

    def body():
        # os.pread, not a buffered file: read-ahead would cache bytes of chunks that aren't written yet
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = start
            while pos <= end:
                if pos < L.data_offset:
                    n = min(end + 1, L.data_offset) - pos
                    yield L.header[pos:pos + n]
                    pos += n
                    continue
                c = L.chunk_of((pos - L.data_offset) // FRAME_BYTES)
                if not L.wait_ready(c):
                    return  # job failed or stalled; the client sees a short read
                stop = min(end, L.data_offset + L.bounds[c][1] * FRAME_BYTES - 1)
                data = os.pread(fd, stop - pos + 1, pos)
                if not data:
                    return
                yield data
                pos += len(data)
        finally:
            os.close(fd)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{L.total}"
    return StreamingResponse(body(), status_code=status, media_type="audio/x-mimi", headers=headers)


@app.get("/v1/episodes/{name}")
def episode(name: str, request: Request, authorization: str | None = Header(None)):
    auth(authorization)
    stem, _, ext = name.partition(".")
    check_id(stem)
    if ext == "segments.json":
        p = DATA / f"{stem}.segments.json"
        if not p.exists():
            raise HTTPException(404, "no segments yet")
        return FileResponse(p, media_type="application/json")
    if ext in ("summary.json", "summary.html", "summary.txt"):
        p = DATA / f"{stem}.summary.json"
        if not p.exists():
            raise HTTPException(404, "no summary yet")
        if ext == "summary.json":
            return FileResponse(p, media_type="application/json")
        html = segments.sanitize_html(json.loads(p.read_text())["summary"])  # (again: whatever wrote the file)
        if ext == "summary.txt":
            import html as h
            text = h.unescape(re.sub(r"<(/(p|h4|li|ul|ol)|br)>", "\n", html))
            return Response(re.sub(r"<[^>]+>", "", text).strip() + "\n", media_type="text/plain")
        return Response(html, media_type="text/html", headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'", "X-Content-Type-Options": "nosniff"})
    if ext == "mimi":
        # a just-requested job writes its header within a few seconds; one on the remote GPU after its start (a
        # cold one on Modal: up to a minute or so)
        t0 = time.time()
        while (L := live.get(stem)) is None:
            r = load(stem)
            if not r or r["state"] not in ("queued", "starting", "downloading"):
                break
            if time.time() - t0 > (120 if r.get("remote") and not r.get("remote_failed") else 30):
                break
            time.sleep(0.25)
        if L is not None:
            return stream_live(L, request.headers.get("range"))
        r = load(stem)
        if (REMOTE and r and r.get("remote") and not r.get("remote_failed") and r["state"] not in ("ready", "failed")
                and r.get("remote_state") not in (None, "queued", "starting", "ready")):
            return proxy_remote(stem, request.headers.get("range"))  # being made on the remote GPU
    if ext == "src":
        r = load(stem)
        p = DATA / f"{stem}.src"
        _mark_served(stem, r)
        if r and p.exists():  # kept, or still being processed
            return FileResponse(p, media_type=r.get("source_type") or "audio/mpeg", filename=f"{stem}.src")
        if not r or not r.get("source_ready") or not r.get("source_url"):
            raise HTTPException(404, "no source")
        return proxy_source(r, request.headers.get("range"))
    if ext in ("mimi", "vtt"):
        p = DATA / f"{stem}.{ext}"
        if not p.exists():
            raise HTTPException(404, "not ready")
        _mark_served(stem, load(stem))
        return FileResponse(p, media_type="audio/x-mimi" if ext == "mimi" else "text/vtt",
                            filename=f"{stem}.{ext}")
    r = load(stem)
    if r is None:
        raise HTTPException(404, "unknown id")
    return r
