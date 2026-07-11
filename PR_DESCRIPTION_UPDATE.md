PR Description Update (Latest Fixes)

Summary
- Fixes terminal repaint/background flash and garbled bottom rows during startup/heavy operations.
- Ensures logging stays centralized so UI and plain-log output are consistent.
- Adds defensive handling for non-UI terminal output while TUI is active.

What changed
1. Unified logger path for CUDA plugin logs
- Removed plugin-local env logger init from plugins/cuda/src/lib.rs.
- Result: CUDA worker info lines now flow through the app logger path instead of writing directly to stderr.

2. TUI-safe stderr handling
- Added stderr redirection in TUI mode in src/main.rs.
- Default target: $HOME/.keryx/stderr.log (override with KERYX_STDERR_LOG_FILE).
- Result: stray dependency/thread stderr output no longer corrupts alternate-screen rendering.

3. UI log sanitization
- Added log message sanitization in src/ui.rs before rendering.
- Strips control/escape characters and normalizes multiline log records.
- Result: prevents control-sequence artifacts and row corruption in the log panel.

4. Minor PoM startup message normalization
- Changed long status punctuation to ASCII in src/pom_gpu.rs for safer terminal rendering.

Behavior impact
- TUI: CUDA startup/worker logs should appear in the in-app log panel via the main logger path.
- Plain file logging: unaffected; when enabled, those lines are included as well.
- stderr capture: raw stderr is still preserved in $HOME/.keryx/stderr.log for diagnostics.

Notes
- Very early output emitted before main logger initialization may still only appear on stderr capture.

Validation
- Diagnostics report no Rust errors in the touched files:
  - src/main.rs
  - src/ui.rs
  - src/pom_gpu.rs
  - plugins/cuda/src/lib.rs
