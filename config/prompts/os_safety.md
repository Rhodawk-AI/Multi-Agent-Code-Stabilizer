# OS Safety Audit Specification — Section VII
# For autonomous agents operating directly on an OS

## OS-Level Safety Requirements

### S1 — Subprocess Isolation (CRITICAL)
All subprocess calls must:
- Never use `shell=True` with any non-constant string
- Explicitly set `timeout` parameter
- Capture stderr separately from stdout
- Handle `TimeoutExpired` and `CalledProcessError`

### S2 — Filesystem Boundaries (CRITICAL)
All file operations must:
- Validate paths are within allowed directories (no path traversal)
- Check permissions before writing
- Use atomic writes (write to temp, then rename) for critical files
- Never delete without explicit confirmation

### S3 — Signal Handling (MAJOR)
Autonomous agents must register handlers for:
- SIGTERM — graceful shutdown, flush all state
- SIGINT — same as SIGTERM
- SIGKILL cannot be caught; ensure state is persisted before any blocking operation

### S4 — Privilege Minimization (CRITICAL)
- No operations requiring root that can be done as user
- Drop privileges as soon as elevated access is done
- Audit every `os.setuid`, `os.setgid`, `subprocess` with elevated env

### S5 — IPC Safety (MAJOR)
- Shared memory access must use locks
- No race conditions in multiprocessing scenarios
- Socket listeners must validate connection sources
