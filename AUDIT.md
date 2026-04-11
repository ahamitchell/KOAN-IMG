# KOAN.img Audit Lenses

This file defines the **lenses** Claude must walk through when the user asks for an audit of KOAN.img. Each lens is a separate pass. Do not collapse multiple lenses into one generic "look for bugs" sweep — that's what causes real issues to be missed.

When the user says something like "audit KOAN," ask which lens (or lenses) they want, or offer to run them in sequence. Report findings **per lens**, not as one flat list.

---

## 1. Destructive / Idempotency
What happens on run 2, 3, 10 of the same operation?
- Can it overwrite user files, prior exports, or cached work?
- `exist_ok=True` on `mkdir` — is the folder meant to be merged into, or is it a silent clobber?
- Does any write path assume "first run" semantics?
- If the user re-runs with the same inputs, do they get a new output or a corrupted mix of old + new?
- Look especially at: exports, saves, deploys, cache writes, temp folders, log rotations.

## 2. Failure modes at boundaries
What happens when the outside world misbehaves?
- Disk full mid-write
- File locked / permission denied (e.g. image open in another app)
- Network dies mid-download / mid-API call
- User closes the app mid-operation
- Source file deleted between index and copy
- Corrupt image, zero-byte file, unexpected format
- Partial writes — is there a temp-file + atomic-rename pattern, or can half-written output survive a crash?

## 3. Large-scale behavior
What breaks at 10k items that works fine at 100?
- UI freezes during long ops (missing thread / QThread / progress dialog)
- Memory growth (images held in memory instead of paths, thumbnails not released)
- Linear scans where an index would do
- Progress feedback: does the user know the app is still alive?
- Cancel button: does one exist, and does it actually stop work?

## 4. State & recovery
If the app crashes mid-task, what's lost?
- Is there a resume path for long operations (scan, download, export)?
- Is state persisted incrementally or only at the end?
- On restart, does the app know what was already done?
- Settings persistence: what's saved, what's silently dropped?

## 5. UX friction
Things that technically work but feel bad.
- Too many clicks for common operations
- Unclear labels, missing tooltips, cryptic error messages
- Missing visual feedback (spinner, toast, disabled button during work)
- Modal vs. non-modal — does a dialog block work that should stay running?
- Default values that are almost never right
- Keyboard shortcuts that exist but aren't documented, or are missing for common actions
- Results of an action that are hard to find after it completes

## 6. Dead code / drift
- Functions no longer called from anywhere
- Settings keys written but never read (or read but never written)
- UI widgets wired to handlers that do nothing
- Commented-out blocks, "TODO / FIXME" older than a month
- Imports no longer used
- Duplicate helpers that diverged over time

## 7. Consistency across tabs/modules
Same operation done two different ways in two places.
- Picker, Narrative, and Video tabs each have their own export / selection / progress / settings code — are they consistent?
- If a bug is fixed in one tab, does the same bug still exist in another? (The unique-folder bug was in BOTH ui_app.py and narrative_tab.py.)
- Common logic that should be a shared helper but isn't
- Style/labeling inconsistencies that make the app feel like three apps stitched together

## 8. Silent failures
Errors that happen but no one notices.
- `try: ... except Exception: pass` blocks
- Errors appended to a list but never surfaced to the user
- `exist_ok=True`, `errors='ignore'`, `.get(key, default)` with suspicious defaults
- Subprocess calls whose return code is ignored
- Network calls without status code checks
- Log lines that are the only evidence something went wrong

## 9. Discoverability
Features that exist but the user probably forgot about.
- Keyboard shortcuts not shown in UI
- Hidden toggles / advanced settings
- Right-click menus, drag-and-drop targets, etc.
- Features built but never wired into a visible button/menu
- This lens is a *reminder pass* — surface things for the user to rediscover.

## 10. Performance cliffs
Places where cost grows nonlinearly.
- O(n²) loops on user-scale data
- Repeated filesystem stats in a tight loop
- Model/embedding reloads per item instead of per batch
- Image decoded multiple times
- Repeated JSON parse of the same file

---

## How to run a lens

1. Pick ONE lens.
2. Read the relevant code with that lens as the *only* filter. Ignore other kinds of issues during this pass (note them separately if they're glaring, but don't let them distract).
3. Report findings as a list: file:line, what you found, why it matters under this lens, suggested fix.
4. Do NOT fix anything without confirmation — audit is read-only until the user approves changes.
5. When done, ask which lens to run next.

## How NOT to run an audit

- Don't grep for "error" and call it an audit.
- Don't mix lenses — each one finds different things.
- Don't skip files because they "look fine." The unique-folder bug looked fine.
- Don't assume "no symptom reported" = "no issue." Silent bugs are the whole point of this file.
