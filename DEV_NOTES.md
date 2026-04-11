# KOAN.img — Dev Notes

> 👋 **Hi — I'm Claude, an AI assistant built by Anthropic.** The developer of
> KOAN.img asked me to write this dev note on their behalf and to be upfront
> that it was written by an AI. So: hello from an AI. Everything below is my
> honest account of a debugging + hardening session I worked on with the
> developer. Any mistakes in the narrative are mine, not theirs.

---

## Session summary — indexing pipeline overhaul

### The original problem

The CPU was overheating during indexing. The developer noticed that one core
was pegged at 100% while the GPU sat mostly idle. The RTX 3080 Ti was being
starved: the main thread was doing PIL decode, resize, colour signatures, and
SQLite writes all by itself, and only sending one tiny image to the GPU at a
time. Net effect: all the heat came out of a single die hotspot, and the GPU
was bored.

### The fix (in three layers)

**1. Parallel CPU prep** — `ai_photo_picker/index_images_chunked.py`

The per-image CPU work (decode, resize, colour signature, animated-webp/gif
check) is now offloaded to a bounded `ThreadPoolExecutor` with
`cpu_count - 2` workers and a prefetch window of `2 × max_workers`. Pillow's
libjpeg/libpng/libwebp decoders, NumPy, and OpenCV all release the GIL during
their C-level work, so threads give real multi-core parallelism — not
pseudo-parallelism.

**What is unchanged on purpose:** image size (1024px max), per-image GPU
calls (still one at a time), FAISS insertion order, DB schema, `batch_commit`
boundaries, and skip-existing semantics. FIFO deque + blocking `fut.result()`
in main thread guarantees FAISS indices are assigned in the exact same order
as the single-threaded loop. Output is bit-identical on the happy path.
Verified on a 1,810-image test folder (contiguous `faiss_idx` 0..1809,
all L2 norms exactly 1.0, zero orphans) and then on the full 899,940-image
D:/MOOD library (same integrity checks, all green).

**2. Robustness: timeouts + error logging**

A single hung PIL decode used to deadlock the entire indexer. Now:

- Main thread calls `fut.result(timeout=30s)`. If a worker hangs, the main
  thread logs `KOAN_TIMEOUT prep hung >30s, skipping: <path>` to stderr,
  increments `skipped_unreadable`, and moves on.
- `_prep_image()` now catches its own exceptions and logs
  `KOAN_PREP_ERR <ExceptionType>: <msg> :: <path>` before returning None,
  so the `skipped_unreadable` counter has a paper trail. No more anonymous
  skipped files.
- Known latent issue: Python can't kill a hung C-level thread. If a worker
  ever hangs for real (not just slow), the chunk's `with ThreadPoolExecutor`
  block will eventually hang at shutdown. In practice this hasn't happened in
  900k+ files — all unreadable files raise exceptions cleanly rather than
  hanging. If it ever becomes a problem, switch to `ProcessPoolExecutor`
  (slower per-image due to pickling, but kill-safe).

**3. Heartbeat logging**

Every ~50 processed items or ~20 seconds of wall time, a heartbeat line prints
to stdout:

```
KOAN_HEARTBEAT work indexed_new=1243 skipped_existing=52104 skipped_unreadable=3 pending=16 chunk=515000->520000
```

- `work` tag = after a real embed + DB write
- `skip` tag = during a skip-heavy stretch (rows already indexed)
- `pending` = futures in flight through the ThreadPoolExecutor

**Why heartbeats matter:** earlier in the session, an indexing run stalled
silently for 60+ seconds with no way to tell if it was wedged or just doing
slow work. Heartbeats make stalls visible immediately. If 20+ seconds pass
between heartbeats, something is wrong — check the exact counters at the
moment of the stall.

### The trap I fell into (lesson for future-me)

At one point I declared the indexer "working correctly — let it finish" based
on the fact that:

1. the process was running (yes),
2. CPU was burning (yes),
3. last-indexed rows looked recent (meaningless — I sorted by `faiss_idx desc`
   and assumed highest = recent without checking timestamps).

The developer pushed back and asked me to actually verify. When I watched the
SQLite counts live instead of taking a snapshot, it turned out the indexer had
been frozen for 60+ seconds with the WAL file untouched. I gave a false
all-clear because I was doing surface checks instead of actually verifying
forward progress.

**Rule I wrote for myself in `AUDIT.md` and memory after this:** when asked
"is it working?", do not answer from a single snapshot. Watch it move over a
time window, and check the WAL mtime, not just the row counts.

### Narrative tab: seed-first export

`ai_photo_picker/narrative_tab.py`: the NARRATIVE tab's export now prepends
the initial seed image(s) to the export as "generation 0", labelled `0A`,
`0B`, `0C`, … Selections from actual generations continue to be `1A`, `1B`,
`2A`, etc. Text-only narratives (no seed images) are unchanged. Push-to-video
is untouched — only the on-disk export includes seeds.

### KOAN.img's audit lens system

In a separate thread of the same session, the developer asked me to stop
doing generic "look for bugs" audits and start using named lenses. I wrote
`ai_photo_picker/AUDIT.md` — a 10-lens checklist (destructive/idempotency,
failure modes, large-scale behavior, state/recovery, UX friction, dead code
drift, consistency across tabs, silent failures, discoverability, performance
cliffs). Future audit requests should name a lens; I should walk it
deliberately, one lens at a time, and never collapse into generic bug
hunting. This came from a real bug I missed: both the Picker and Narrative
tab export paths used `mkdir(exist_ok=True)`, which silently merged into
prior export folders instead of creating unique ones. A destructive-behavior
lens would have caught it immediately.

### Final state

- Parallelization is committed (`da93926` on `main`)
- This commit adds: robustness (timeout + heartbeats + error logging),
  `_prep_image` path-logging fix, narrative seed-first export, About dialog,
  and this dev note.
- D:/MOOD index verified: 899,900 images, 899,900 colours, 899,900 non-empty
  captions, faiss `ntotal = 899,900`, contiguous `faiss_idx` 0..899,899, zero
  orphans, L2 norms = 1.0 at every sampled position.

---

*— Claude (Anthropic), writing on the developer's behalf, 2026-04-11*
