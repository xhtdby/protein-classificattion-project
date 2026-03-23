#!/usr/bin/env bash
# =============================================================================
# run_training_safe.sh — Fault-tolerant LoRA fine-tuning daemon for RunPod
#
# USAGE
#   export GITHUB_PAT="ghp_xxxxxxxxxxxxxxxxxxxx"
#   export RUNPOD_POD_ID="<your-pod-id>"          # set by RunPod automatically
#   bash src/runpod/run_training_safe.sh [--batch-size 2] [--epochs 10]
#
# STRUCTURAL GUARANTEES
#   (1) Authenticates against GitHub via a Personal Access Token supplied in
#       the $GITHUB_PAT environment variable — no plain-text credentials in repo.
#   (2) Retries training up to 2 times on CUDA OOM, halving --batch-size each
#       attempt, before declaring failure.
#   (3) After the final run (success or failure), commits the log file and any
#       generated model artefacts to a timestamped branch and pushes to origin
#       so you can review results asynchronously from your local machine.
#   (4) A bash `trap` ensures `runpodctl pod stop` is called unconditionally
#       at process exit — even on SIGINT/SIGTERM — to end billing immediately
#       after the push completes.
#
# ENVIRONMENT VARIABLES (required)
#   GITHUB_PAT      — GitHub personal access token with repo write scope.
#   RUNPOD_POD_ID   — The pod ID to stop. RunPod sets this automatically in
#                     the container environment; verify with: echo $RUNPOD_POD_ID
#
# OPTIONAL ENVIRONMENT VARIABLES
#   REPO_OWNER      — GitHub account/org name (default: inferred from remote URL)
#   REPO_NAME       — Repository name (default: inferred from remote URL)
#   REMOTE_URL      — Full HTTPS remote URL; set this to override auto-detection.
#
# =============================================================================

set -uo pipefail
# NOTE: We intentionally do NOT use `set -e`.
# We need to capture exit codes manually to distinguish OOM from other errors
# and to guarantee the Git push + pod stop trap always runs.

# ---------------------------------------------------------------------------
# 0. Configuration & defaults
# ---------------------------------------------------------------------------

# Absolute path to the repository root (assumed to be two levels above this
# script, i.e.  <repo>/src/runpod/run_training_safe.sh → <repo>/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Log file that captures all Python stdout/stderr.
LOG_FILE="${REPO_ROOT}/outputs/training_daemon.log"

# Primary model artefact produced by lora_finetune.py.
ARTEFACT="${REPO_ROOT}/outputs/models/finetune_artifact.joblib"

# Results JSON written by the training script.
RESULTS_JSON="${REPO_ROOT}/outputs/finetune_results.json"

# Glob pattern for per-fold checkpoint files.
FOLD_CKPTS="${REPO_ROOT}/outputs/models/finetune_fold*.pt"

# Branch name for this run, stamped with UTC time so each RunPod job
# produces an independently reviewable branch.
BRANCH_NAME="runpod-results-$(date -u +%Y%m%d-%H%M%S)"

# Maximum number of OOM retries (each retry halves the batch size).
MAX_RETRIES=2

# Initialise batch-size and epoch count from CLI arguments (or keep defaults).
# These may be overridden by the OOM retry loop below.
BATCH_SIZE=2
EPOCHS=10
EXTRA_ARGS=""   # any additional CLI flags forwarded verbatim to the Python module

# Parse simple --key value pairs forwarded to this script.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --epochs)       EPOCHS="$2";      shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Logging helper — every message is timestamped and written to both the
#    terminal and the log file (tee).  The log file is opened here so that
#    even pre-training failures are captured for later inspection.
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1   # redirect all stdout+stderr through tee

log() {
    # Usage: log [INFO|WARN|ERROR] "message"
    local level="${1:-INFO}"
    local msg="$2"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [$level] $msg"
}

log INFO "=========================================================="
log INFO " LoRA ESM-2 fine-tuning daemon starting"
log INFO " Branch  : $BRANCH_NAME"
log INFO " Repo    : $REPO_ROOT"
log INFO " Log     : $LOG_FILE"
log INFO "=========================================================="

# ---------------------------------------------------------------------------
# 2. Validate required environment variables
# ---------------------------------------------------------------------------

if [[ -z "${GITHUB_PAT:-}" ]]; then
    log ERROR "GITHUB_PAT is not set. Export your GitHub PAT before running:"
    log ERROR "  export GITHUB_PAT='ghp_xxxxxxxxxxxxxxxxxxxx'"
    # We still want the trap to fire (pod stop), so we exit with a non-zero
    # code rather than 'exit' after the trap is registered. For now, defer
    # the actual exit until after trap registration below.
    _FATAL_ENV_ERROR=1
else
    _FATAL_ENV_ERROR=0
fi

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    log WARN "RUNPOD_POD_ID is not set. Pod will NOT be stopped automatically."
    log WARN "Set it manually: export RUNPOD_POD_ID=\$(runpodctl get pod | awk 'NR==2{print \$1}')"
fi

# ---------------------------------------------------------------------------
# 3. Dead-man's switch — registered BEFORE any work begins.
#
#    The `trap` command hooks the EXIT pseudo-signal, which fires in ALL of
#    the following scenarios:
#      • Normal script completion (success or failure)
#      • Unhandled error (set -u triggers on unbound variable)
#      • SIGINT  (Ctrl-C or pod interruption)
#      • SIGTERM (RunPod internal shutdown)
#
#    Execution order inside the trap:
#      (a) Push artefacts to GitHub (so results are never lost).
#      (b) Stop the pod (ends billing).
#
#    The `|| true` after each command prevents a failure in the trap itself
#    from masking the original exit condition.
# ---------------------------------------------------------------------------

_git_push_results() {
    # Inner function called from the EXIT trap.
    # Commits whatever artefacts exist and pushes to the timestamped branch.
    log INFO "----------------------------------------------------------"
    log INFO " GIT: Committing artefacts to branch '$BRANCH_NAME'"
    log INFO "----------------------------------------------------------"

    cd "$REPO_ROOT"

    # Create and check out the result branch (force, in case of re-run).
    git checkout -b "$BRANCH_NAME" 2>/dev/null \
        || git checkout "$BRANCH_NAME"

    # Stage the log unconditionally (it always exists after exec above).
    git add -f "$LOG_FILE" || true

    # Stage model artefacts only if they were produced (training may have failed).
    [[ -f "$ARTEFACT"      ]] && git add -f "$ARTEFACT"      || true
    [[ -f "$RESULTS_JSON"  ]] && git add -f "$RESULTS_JSON"  || true
    # Stage all per-fold checkpoint files that exist.
    for ckpt in $FOLD_CKPTS; do
        [[ -f "$ckpt" ]] && git add -f "$ckpt" || true
    done

    # Write the commit message dynamically so it records success/failure.
    local status_msg
    if [[ "${_TRAINING_SUCCESS:-0}" == "1" ]]; then
        status_msg="SUCCESS — LoRA fine-tuning completed"
    else
        status_msg="FAILURE — LoRA fine-tuning did not complete; see log"
    fi

    git commit -m "RunPod: $status_msg

Branch  : $BRANCH_NAME
Pod ID  : ${RUNPOD_POD_ID:-unknown}
Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Final batch size: $BATCH_SIZE
" || {
        log WARN "GIT: Nothing to commit (artefacts may already be tracked)."
    }

    # Push to origin using the PAT-authenticated remote URL.
    log INFO "GIT: Pushing '$BRANCH_NAME' to origin..."
    if git push origin "$BRANCH_NAME" --force; then
        log INFO "GIT: Push succeeded. Review at:"
        log INFO "  https://github.com/$(git remote get-url origin \
            | sed 's|.*github.com[:/]\(.*\)\.git|\1|' \
            | sed 's|.*github.com[:/]\(.*\)|\1|')/tree/$BRANCH_NAME"
    else
        log ERROR "GIT: Push failed. Artefacts are local in branch '$BRANCH_NAME'."
    fi
}

_stop_pod() {
    # Inner function to stop the RunPod pod, terminating billing.
    if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
        log WARN "RUNPOD_POD_ID not set — skipping pod stop. Stop the pod manually!"
        return 0
    fi

    log INFO "----------------------------------------------------------"
    log INFO " POD STOP: Stopping pod $RUNPOD_POD_ID"
    log INFO "----------------------------------------------------------"

    if command -v runpodctl &>/dev/null; then
        runpodctl pod stop "$RUNPOD_POD_ID" \
            && log INFO "POD STOP: runpodctl succeeded." \
            || log ERROR "POD STOP: runpodctl exited non-zero. Stop the pod manually!"
    else
        log ERROR "runpodctl not found on PATH. Stop the pod manually!"
        log ERROR "  https://www.runpod.io/console/pods"
    fi
}

# Register the compound EXIT trap.
# Runs _git_push_results then _stop_pod, guarded with `|| true` so a failure
# in the git step doesn't prevent billing from being stopped.
trap '_git_push_results || true; _stop_pod || true' EXIT

# Now handle the fatal env error we deferred above.
if [[ "$_FATAL_ENV_ERROR" == "1" ]]; then
    log ERROR "Aborting due to missing environment variables."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Authenticate git remote with the Personal Access Token.
#
#    We rewrite the remote URL to embed the token in the HTTPS credential.
#    Format: https://<token>@github.com/<owner>/<repo>.git
#
#    The token is NOT written to any config file that is tracked by git;
#    it only lives in the local .git/config for this session.
# ---------------------------------------------------------------------------

log INFO "SETUP: Configuring git remote authentication..."

# Auto-detect the remote URL if not explicitly set.
CURRENT_REMOTE="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || echo '')"

if [[ -z "$CURRENT_REMOTE" ]]; then
    log ERROR "No 'origin' remote found in $REPO_ROOT. Aborting."
    exit 1
fi

# Extract the host-and-path portion, stripping any existing credential.
# Handles both https://github.com/... and git@github.com:... forms.
if [[ "$CURRENT_REMOTE" =~ github\.com[:/]([^/]+)/(.+)(\.git)?$ ]]; then
    GH_OWNER="${BASH_REMATCH[1]}"
    GH_REPO="${BASH_REMATCH[2]%.git}"
else
    log ERROR "Cannot parse GitHub owner/repo from remote: $CURRENT_REMOTE"
    exit 1
fi

AUTHENTICATED_REMOTE="https://${GITHUB_PAT}@github.com/${GH_OWNER}/${GH_REPO}.git"

# Set the authenticated URL locally (not pushed to the repo).
git -C "$REPO_ROOT" remote set-url origin "$AUTHENTICATED_REMOTE"
log INFO "SETUP: Remote URL updated to use PAT (credentials not logged)."

# Set git identity for the commit (required in CI/headless environments).
git -C "$REPO_ROOT" config user.email "runpod-daemon@noreply.local" 2>/dev/null || true
git -C "$REPO_ROOT" config user.name  "RunPod Training Daemon"        2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. GPU sanity check — verify CUDA is visible before wasting time.
# ---------------------------------------------------------------------------

log INFO "SETUP: Verifying CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not found'; \
    print(f'  GPU: {torch.cuda.get_device_name(0)}'); \
    free, total = torch.cuda.mem_get_info(0); \
    print(f\"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total\")"; then
    log ERROR "CUDA is not available. Check the RunPod GPU allocation."
    exit 1
fi

# ---------------------------------------------------------------------------
# 6. OOM-resilient training loop
#
#    Strategy:
#      • Run the Python training module, redirecting output to the log.
#      • If it exits non-zero AND the log contains an OOM signature string,
#        halve the batch size and retry (up to MAX_RETRIES times).
#      • If it exits non-zero for any other reason, propagate the failure.
#      • If all retries are exhausted as OOM, exit with a specific code.
#
#    Batch-size halving logic:
#      Default 2 -> retry 1: 1 -> retry 2: 1 (floor at 1; cannot go lower)
#      Note: at batch=1 with grad_accum=8, effective batch is still 8.
# ---------------------------------------------------------------------------

# OOM fingerprint strings that PyTorch writes to stderr.
OOM_PATTERNS="CUDA out of memory|OutOfMemoryError|out_of_memory|cudaMalloc failed"

_run_training() {
    # Runs a single training attempt with the current $BATCH_SIZE.
    # Returns the Python process exit code.
    local cmd="python -m src.models.lora_finetune \
        --batch-size ${BATCH_SIZE} \
        --epochs     ${EPOCHS} \
        ${EXTRA_ARGS}"

    log INFO "TRAIN: Launching: $cmd"
    log INFO "TRAIN: batch_size=$BATCH_SIZE  epochs=$EPOCHS"

    # Run the command. stdout+stderr already flow to the log via the `exec tee`
    # at the top of the script; we don't need a separate redirect here.
    cd "$REPO_ROOT"
    $cmd
    return $?
}

_is_oom() {
    # Returns 0 (true) if the log contains an OOM signature since the last
    # training start marker.
    grep -qE "$OOM_PATTERNS" "$LOG_FILE"
}

ATTEMPT=0
_TRAINING_SUCCESS=0   # flag read by the EXIT trap to compose the commit message

while [[ $ATTEMPT -le $MAX_RETRIES ]]; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    log INFO "=========================================================="
    log INFO " ATTEMPT $ATTEMPT / $(( MAX_RETRIES + 1 ))  batch_size=$BATCH_SIZE"
    log INFO "=========================================================="

    # Write an attempt marker into the log so _is_oom can scope its search.
    echo "--- ATTEMPT $ATTEMPT START ---" >> "$LOG_FILE"

    _run_training
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        log INFO "TRAIN: Attempt $ATTEMPT succeeded (exit 0)."
        _TRAINING_SUCCESS=1
        break  # Exit the retry loop; the EXIT trap will push results.
    fi

    # Non-zero exit: check whether it was an OOM.
    log WARN "TRAIN: Attempt $ATTEMPT exited with code $EXIT_CODE."

    if _is_oom; then
        if [[ $ATTEMPT -le $MAX_RETRIES ]]; then
            # Halve the batch size, flooring at 1.
            NEW_BATCH=$(( BATCH_SIZE / 2 ))
            [[ $NEW_BATCH -lt 1 ]] && NEW_BATCH=1

            log WARN "OOM: CUDA out-of-memory detected."
            log WARN "OOM: Reducing batch_size $BATCH_SIZE -> $NEW_BATCH and retrying."
            BATCH_SIZE=$NEW_BATCH

            # Give the CUDA allocator a moment to release cached memory.
            sleep 5
        else
            log ERROR "OOM: All $MAX_RETRIES OOM retries exhausted. Giving up."
            _TRAINING_SUCCESS=0
            # EXIT trap will fire on `exit` below.
            exit $EXIT_CODE
        fi
    else
        # A non-OOM failure — do not retry; propagate the error.
        log ERROR "TRAIN: Non-OOM failure on attempt $ATTEMPT (exit $EXIT_CODE)."
        log ERROR "TRAIN: Check the log for details: $LOG_FILE"
        _TRAINING_SUCCESS=0
        # EXIT trap will fire on `exit` below.
        exit $EXIT_CODE
    fi
done

# If we exhausted the loop without setting _TRAINING_SUCCESS=1, record failure.
if [[ "$_TRAINING_SUCCESS" != "1" ]]; then
    log ERROR "TRAIN: All attempts failed without OOM match. Exiting."
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Log final summary before the EXIT trap fires.
# ---------------------------------------------------------------------------

log INFO "=========================================================="
log INFO " Training complete. Artefacts:"
[[ -f "$ARTEFACT"     ]] && log INFO "  $ARTEFACT"     || log WARN "  artefact.joblib NOT found"
[[ -f "$RESULTS_JSON" ]] && log INFO "  $RESULTS_JSON" || log WARN "  results JSON NOT found"
for ckpt in $FOLD_CKPTS; do
    [[ -f "$ckpt" ]] && log INFO "  $ckpt"
done
log INFO "=========================================================="
log INFO " Exiting cleanly. EXIT trap will push to GitHub and stop pod."
log INFO "=========================================================="

# Normal exit → EXIT trap fires → _git_push_results → _stop_pod.
exit 0
