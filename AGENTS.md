# Standing rules for this repository (Wild-ID, tiger re-identification)

## Honesty
- Cosine similarity is a ranking signal, not a probability, "confidence", or verified identity. Never label it so.
- Never write "match found", "new tiger detected", or any accuracy claim unless a committed script produced it from a documented dataset. Show sample sizes and method next to every metric.
- Never fabricate results, test passes, dataset facts, species lists, or detections. If something cannot be run or verified, say so plainly.
- No Tadoba deployment or validation claims. Never relabel any animal as a tiger.
- Do not guess the dataset name, source, or licence. Ask me.

## Safety
- No destructive commands (recursive deletes, force pushes) without asking me first. Prefer moving to a `trash/` folder.
- Never read or write outside the project folder. Validate and sanitize any name used in a file path.
- Do not use pickle for new files. Do not add dependencies without a stated reason and pinned version.
- Do not upload images or data to third-party services.

## Environment
- Windows, PowerShell, Python 3.13, CPU only. Read and write all text files as UTF-8. Avoid non-ASCII characters in print statements.
- Run scripts from the repository root.

## Process
- Inspect before editing. Keep working features. Make small commits per task.
- After each task, report files changed, tests run with real output, and anything not run and why.
