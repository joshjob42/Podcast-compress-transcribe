"""Backlogs: many episodes of a podcast converted at once, asked for from the app (hub/mimi_hub.py's /v1/backlog).

A backlog is a podcast's episodes (all, the latest N, or those published in a date range: Pocket Casts' public
podcast/full API, ids = the app's episode UUIDs) or an explicit list (the app's multi-select). Small ones (under
MIMI_HUB_BACKLOG_MODAL_H hours of audio not yet in the library, default 3) go through the hub's own queue. Bigger
ones, where the modal CLI is set up here (~/.modal.toml), go to Modal's bulk mode as one run
(scripts/modal_backlog.py, ~1,350x real time on an H100), each chunk of MIMI_HUB_BACKLOG_CHUNK (150) episodes
copied into the library as it lands (scripts/import_modal_library.py) and its ads and summaries found here,
MIMI_HUB_BACKLOG_THREADS (8) at a time, with this hub's own LLM logins. One Modal backlog runs at a time; jobs are
kept in DATA/backlogs/ and resume after a restart (the Modal run skips what it already made).
"""
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid as uuidlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODAL_CMD = os.environ.get("MIMI_HUB_MODAL_CMD", "uvx --python 3.12 --from modal modal")
MODAL_PYTHON = os.environ.get("MIMI_HUB_MODAL_PYTHON", "uvx --python 3.12 --with modal python")  # (the import)
MODAL = os.environ.get("MIMI_HUB_BACKLOG_MODAL", "auto")  # auto: if the modal CLI is set up here | 0: never
MODAL_MIN_H = float(os.environ.get("MIMI_HUB_BACKLOG_MODAL_H", "3"))
CHUNK = int(os.environ.get("MIMI_HUB_BACKLOG_CHUNK", "150"))
THREADS = int(os.environ.get("MIMI_HUB_BACKLOG_THREADS", "8"))
# a Modal bulk run's cost: ~6 min of H100 + 32 cores + 96 GiB starting and loading (per chunk, in part), then
# ~1,350x real time (2026-11-02: 1,131 episodes, 912 h, ~45 min, ~$5)
USD_START, USD_PER_AUDIO_H = 0.65, 0.0046
UA = {"User-Agent": "Mozilla/5.0 (mimi-hub)"}
FINAL = ("done", "failed", "cancelled")

_jobs = {}
_lock = threading.Lock()
_wake = threading.Event()
_procs = {}  # job id -> its Modal run (for cancelling)


def _hub():
    from hub import mimi_hub
    return mimi_hub


def _dir():
    d = _hub().DATA / "backlogs"
    d.mkdir(exist_ok=True)
    return d


# ---------- which episodes ----------

def podcast_episodes(podcast_uuid):
    """(the podcast's title, [{"id", "url", "title", "published", "duration_s"}]), oldest first."""
    import gzip
    raw = urllib.request.urlopen(urllib.request.Request(
        f"https://podcast-api.pocketcasts.com/podcast/full/{podcast_uuid}", headers=UA), timeout=60).read()
    pod = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)["podcast"]
    show = pod.get("title", "")
    eps = [{"id": e["uuid"], "url": e["url"], "title": f"{show}: {e['title']}", "published": e.get("published", ""),
            "duration_s": e.get("duration") or 0} for e in pod.get("episodes", []) if e.get("url")]
    return show, sorted(eps, key=lambda e: e["published"])


def select(eps, latest=None, after=None, before=None):
    """[eps] published on or after [after] and before [before] (YYYY-MM-DD), then the [latest] N of them."""
    eps = [e for e in eps if (not after or e.get("published", "")[:10] >= after)
           and (not before or e.get("published", "")[:10] < before)]
    return eps[-latest:] if latest else eps


def modal_available():
    if MODAL == "0":
        return False
    exe = shlex.split(MODAL_CMD)[0]
    return (Path.home() / ".modal.toml").exists() and shutil.which(exe) is not None


def plan(episodes):
    """What a backlog of [episodes] would do: how many aren't in the library yet, their hours, where they'd be
    converted (modal / local) and, on Modal, a rough cost."""
    hub = _hub()
    new = [e for e in episodes if (hub.load(e["id"]) or {}).get("state") != "ready"]
    hours = sum(e.get("duration_s") or 3600 for e in new) / 3600  # (no duration listed: an hour)
    modal = modal_available() and hours >= MODAL_MIN_H
    return {"episodes": len(episodes), "new": len(new), "hours": round(hours, 1),
            "mode": "modal" if modal else "local",
            "usd": round(USD_START + hours * USD_PER_AUDIO_H, 2) if modal else 0.0,
            "minutes": round(6 + hours / 1350 * 60 + len(new) * 0.02) if modal else None}


# ---------- jobs ----------

def _save(job):
    job["updated"] = time.time()
    tmp = _dir() / f"{job['id']}.json.tmp"
    tmp.write_text(json.dumps(job, indent=1))
    tmp.replace(_dir() / f"{job['id']}.json")


def _progress(job):
    """Counts from the episodes' records: converted, with ads, with a summary."""
    hub = _hub()
    recs = [hub.load(e["id"]) or {} for e in job["episodes"]]
    return {"converted": sum(r.get("state") == "ready" for r in recs),
            "failed": sum(r.get("state") == "failed" for r in recs),
            "ads": sum(r.get("segments_state") in ("ready", "off") for r in recs),
            "summaries": sum(r.get("summary_state") == "ready" for r in recs)}


def public(job):
    """A job as the API shows it: without its episode list, with its progress."""
    out = {k: v for k, v in job.items() if k != "episodes"}
    out.update(_progress(job))
    return out


def create(title, episodes, podcast_uuid=None):
    p = plan(episodes)
    job = {"id": uuidlib.uuid4().hex[:12], "podcast_uuid": podcast_uuid, "title": title, "created": time.time(),
           "state": "queued", "error": None, "total": len(episodes), **p, "episodes": episodes}
    with _lock:
        _jobs[job["id"]] = job
        _save(job)
    _wake.set()
    return job


def jobs(podcast_uuid=None):
    with _lock:
        js = list(_jobs.values())
    return sorted([j for j in js if not podcast_uuid or j.get("podcast_uuid") == podcast_uuid],
                  key=lambda j: -j["created"])


def active_ids():
    """The episodes of unfinished jobs (the hub's disk cap leaves them alone: they're counted, and their ads are due)."""
    return {e["id"] for j in jobs() if j["state"] not in FINAL for e in j["episodes"]}


def get(job_id):
    return _jobs.get(job_id)


def cancel(job_id):
    job = _jobs.get(job_id)
    if not job or job["state"] in FINAL:
        return job
    job["state"] = "cancelled"
    _save(job)
    proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)  # (scripts/modal_backlog.py cancels its Modal call on SIGTERM)
    return job


def start():
    """Load the saved jobs (unfinished ones queued again) and start the runner."""
    for p in sorted(_dir().glob("*.json")):
        if p.name.endswith(".manifest.json") or p.name.endswith(".results.json"):
            continue
        job = json.loads(p.read_text())
        if job["state"] not in FINAL:
            job["state"] = "queued"
        _jobs[job["id"]] = job
    threading.Thread(target=_runner, daemon=True, name="mimi-backlog").start()
    _wake.set()


def _runner():
    while True:
        _wake.wait(30)
        _wake.clear()
        for job in sorted(jobs(), key=lambda j: j["created"]):
            if job["state"] != "queued":
                continue
            try:
                (_run_modal if job["mode"] == "modal" else _run_local)(job)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                if job["state"] != "cancelled":
                    job.update(state="failed", error=f"{type(e).__name__}: {e}"[:500])
                    _save(job)


def _run_local(job):
    """Into the hub's own queue (normal priority); the hub converts them and finds their ads and summaries."""
    hub = _hub()
    job["state"] = "converting"
    _save(job)
    for e in job["episodes"]:
        if job["state"] == "cancelled":
            return
        r = hub.load(e["id"])
        if not r or r.get("state") == "failed":
            hub.enqueue(e["id"], priority=1, url=e["url"], audio_url=e["url"], title=e.get("title") or e["id"])
    while job["state"] != "cancelled":  # done once every episode is converted (or failed) and its ads are found
        p = _progress(job)
        if p["converted"] + p["failed"] >= job["total"] and p["ads"] >= p["converted"]:
            job["state"] = "done"
            _save(job)
            return
        time.sleep(15)


def _run_modal(job):
    """One Modal bulk run of the episodes not in the library yet; each chunk imported as it lands, then its ads and
    summaries found here."""
    hub = _hub()
    todo = [e for e in job["episodes"] if (hub.load(e["id"]) or {}).get("state") != "ready"]
    manifest = _dir() / f"{job['id']}.manifest.json"
    manifest.write_text(json.dumps([{"uuid": e["id"], "url": e["url"], "title": e.get("title") or e["id"],
                                     "duration_s": e.get("duration_s") or 0} for e in todo]))
    job["state"] = "converting"
    _save(job)
    env = dict(os.environ)
    if not env.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
        env["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    hours = sum(e.get("duration_s") or 3600 for e in todo) / 3600
    wait_min = max(30, int(hours / 1350 * 60 * 2 + 20 + len(todo) / CHUNK * 3))
    llm = ThreadPoolExecutor(THREADS, thread_name_prefix="mimi-backlog-llm")
    queued = set()

    def bring_in():  # what has landed on Modal: copied here, then its ads and summary
        if todo:
            subprocess.run(shlex.split(MODAL_PYTHON) + ["scripts/import_modal_library.py", str(manifest), "--data",
                                                        str(hub.DATA)],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        for e in job["episodes"]:  # (also ones brought in before a restart, or in the library without ads / a summary)
            if e["id"] not in queued and _needs_llm(hub.load(e["id"]) or {}):
                queued.add(e["id"])
                llm.submit(_llm, e["id"])

    if todo:
        cmd = shlex.split(MODAL_CMD) + ["run", "scripts/modal_backlog.py", "--manifest", str(manifest),
                                        "--chunk", str(CHUNK), "--wait-min", str(wait_min)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _procs[job["id"]] = proc
        try:
            for line in proc.stdout:
                if line.startswith("CHUNK {"):
                    print(f"backlog {job['id']}: {line.strip()[:200]}", flush=True)
                    bring_in()
                elif "Error" in line or "Traceback" in line:
                    print(f"backlog {job['id']}: {line.strip()[:300]}", flush=True)
            proc.wait()
        finally:
            _procs.pop(job["id"], None)
        if job["state"] == "cancelled":
            llm.shutdown(wait=False, cancel_futures=True)
            return
        if proc.returncode:
            job["error"] = f"modal run exited {proc.returncode}"
    bring_in()
    job["state"] = "finishing"  # (ads and summaries)
    _save(job)
    llm.shutdown(wait=True)
    p = _progress(job)
    job["state"] = "done" if p["converted"] else "failed"
    _save(job)


def _needs_llm(r):
    from hub import segments
    return r.get("state") == "ready" and (r.get("segments_state") not in ("ready", "off") or (
        segments.SUMMARIES and r.get("summary_state") != "ready"))


def _llm(i):
    """[i]'s ads, then its summary, as the hub's own workers would (written under its record lock)."""
    hub = _hub()
    from hub import segments
    try:
        cues = segments.read_cues((hub.DATA / f"{i}.vtt").read_text())
        r = hub.load(i) or {}
        if r.get("segments_state") != "ready":
            res = segments.detect(cues, log=lambda m: print(f"[{i}] {m}", flush=True))
            tmp = hub.DATA / f"{i}.segments.json.tmp"
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(hub.DATA / f"{i}.segments.json")
            hub.update(i, segments_state="ready", segments=len(res["segments"]), segments_model=res["model"],
                       segments_error=None, skippable_s=round(sum(s["end"] - s["start"] for s in res["segments"]), 1))
            found = res["segments"]
        else:
            found = json.loads((hub.DATA / f"{i}.segments.json").read_text())["segments"]
        if segments.SUMMARIES and r.get("summary_state") != "ready":
            res = segments.summarize(cues, found, log=lambda m: None)
            tmp = hub.DATA / f"{i}.summary.json.tmp"
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(hub.DATA / f"{i}.summary.json")
            hub.update(i, summary_state="ready", summary_model=res["model"], summary_error=None)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        hub.update(i, segments_error=f"backlog: {type(e).__name__}: {e}"[:300])
