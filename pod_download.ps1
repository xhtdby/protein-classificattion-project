# pod_download.ps1 — Download model artifact from a GPU pod after training
#
# Usage:
#   .\pod_download.ps1 -PodHost "ssh.runpod.io" -Port 12345
#
# Results JSONs and figures are pulled via git pull (much faster).
# Only the .joblib model needs scp because it's gitignored (binary, ~200MB).

param(
    [Parameter(Mandatory)][string]$PodHost,
    [Parameter(Mandatory)][int]$Port,
    [string]$Key = "~/.ssh/id_rsa",
    [string]$User = "root",
    [string]$RemoteDir = "/workspace/protein-classification"
)

Write-Host "========================================"
Write-Host "  POD DOWNLOAD"
Write-Host "  <- ${User}@${PodHost}:${Port}"
Write-Host "========================================"

# ── 1. Push results to GitHub from the pod ───────────────────────────────────
Write-Host "`n[1/2] Pushing results from pod to GitHub..."
$pushCmd = @"
cd $RemoteDir && \
git config user.email 'pod@results.local' && \
git config user.name 'pod' && \
git add outputs/*.json outputs/figures/*.png outputs/*.txt && \
git commit -m 'results: 650M ESM-2 training on pod' 2>/dev/null || echo 'Nothing new to commit' && \
git push origin main
"@
ssh -p $Port -i $Key "${User}@${PodHost}" $pushCmd

# ── 2. Pull the new results locally ──────────────────────────────────────────
Write-Host "`n[2/2] Pulling results to local machine..."
git pull origin main

# ── 3. scp the model artifact (gitignored, binary) ───────────────────────────
Write-Host "`nDownloading best_model.joblib (~200MB)..."
New-Item -ItemType Directory -Force -Path "outputs\models" | Out-Null
scp -P $Port -i $Key `
    "${User}@${PodHost}:${RemoteDir}/outputs/models/best_model.joblib" `
    "outputs\models\best_model.joblib"

Write-Host "`n========================================"
Write-Host "  Download complete."
Write-Host ""
Write-Host "  Verify with:"
Write-Host "    .venv\Scripts\python.exe -c `"import joblib; a=joblib.load('outputs/models/best_model.joblib'); print(a['feature_source'], a.get('cv_scores'))`""
Write-Host "========================================"
