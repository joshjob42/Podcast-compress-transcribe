"""In-process test of the disk cap (MIMI_HUB_DATA_MAX_GB, mimi_hub.trim_data), on a made-up library: no models, no GPU.

usage: uv run python hub/test_trim.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

DATA = Path(tempfile.mkdtemp(prefix="mimi-trim-"))
os.environ.update(MIMI_HUB_DATA=str(DATA), MIMI_HUB_DATA_MAX_GB="0.00001", MIMI_HUB_ENV_FILE="/nonexistent")  # 10 kB
sys.path.insert(0, ".")
from hub import backlog, mimi_hub as hub  # noqa: E402

now = time.time()


def episode(i, age, src=0, mimi=0, **fields):
    for ext, size in (("src", src), ("mimi", mimi)):
        if size:
            (DATA / f"{i}.{ext}").write_bytes(b"x" * size)
    (DATA / f"{i}.vtt").write_text("WEBVTT\n")
    (DATA / f"{i}.json").write_text(json.dumps(dict({"id": i, "state": "ready", "created": now - age}, **fields)))


def files():
    return sorted(p.name for p in DATA.iterdir() if p.name != ".records.lock")


def reset():
    for p in DATA.iterdir():
        if p.is_file() and p.name != ".records.lock":
            p.unlink()


# originals go first, oldest first; the episode keeps the rest (marks still right for our copy on phones)
episode("old_mimi", 90, mimi=2000)
episode("both", 50, src=8000, mimi=2000, keep_source=True, compressed=True, source_ready=True, source_url="http://x/a")
episode("only_src", 40, src=8000, keep_source=True, compressed=False)
episode("new_src", 30, src=3000, keep_source=True, compressed=False)
hub.trim_data()
assert files() == ["both.json", "both.mimi", "both.vtt", "new_src.json", "new_src.src", "new_src.vtt", "old_mimi.json",
                   "old_mimi.mimi", "old_mimi.vtt", "only_src.json", "only_src.vtt"], files()

# .mimi episodes only once no original is left, least recently served first
reset()
episode("a", 90, mimi=4000, served=now - 10)
episode("b", 80, mimi=4000)
episode("c", 70, src=6000, keep_source=True, compressed=False)
hub.trim_data()
assert not (DATA / "c.src").exists() and (DATA / "c.json").exists() and (DATA / "a.mimi").exists(), files()
assert (DATA / "b.mimi").exists(), files()
episode("d", 60, mimi=4000)
hub.trim_data()
assert not (DATA / "b.json").exists() and (DATA / "a.mimi").exists() and (DATA / "c.json").exists(), files()

# left alone: still being worked on, uploads, what a backlog or scripts/backlog_ads.py hasn't finished
reset()
episode("queued", 90, src=9000, state="queued")
episode("ads", 90, mimi=9000, segments_state="pending")
episode("summary", 90, mimi=9000, summary_state="pending")
episode("upload", 90, mimi=9000, upload="/gone", url="upload:talk.mp3")
episode("imported", 90, mimi=9000, batch=True, segments_state="failed")
episode("in_backlog", 90, mimi=9000, segments_state="ready")
backlog._jobs["job"] = {"id": "job", "state": "converting", "created": now, "episodes": [{"id": "in_backlog"}]}
before = files()
hub.trim_data()
assert files() == before, files()
assert hub._trim_stuck[0]

# a finished backlog's episodes are ordinary ones again
backlog._jobs["job"]["state"] = "done"
hub.trim_data()
assert not (DATA / "in_backlog.json").exists(), files()

# a failed episode's leftover original goes, its record stays (the app retries it)
reset()
episode("failed", 90, src=20000, state="failed", error="x")
hub.trim_data()
assert not (DATA / "failed.src").exists() and (DATA / "failed.json").exists(), files()

# a crash's temp files don't count against the cap, and go at startup
reset()
episode("keep", 90, mimi=9000)
(DATA / "tmpabc123.npy").write_bytes(b"x" * 50000)
(DATA / "keep.segments.json.tmp").write_bytes(b"x" * 5000)
hub.trim_data()
assert (DATA / "keep.mimi").exists(), files()
hub._trim_stuck[0] = False
for f in DATA.iterdir():
    if f.is_file() and hub._scratch(f):
        f.unlink()
assert files() == ["keep.json", "keep.mimi", "keep.vtt"], files()

# served: any read of the audio or transcript counts, once an hour; a removed episode is no error
reset()
episode("read", 90, mimi=10)
hub._mark_served("read", hub.load("read"))
served = hub.load("read")["served"]
assert served > now
hub._mark_served("read", hub.load("read"))
assert hub.load("read")["served"] == served
hub._mark_served("gone", {"id": "gone"})
print("ok")
