# pod_upload.ps1 — Upload .npy feature files to a GPU pod
#
# Usage (replace with your pod's SSH details):
#   .\pod_upload.ps1 -Host "ssh.runpod.io" -Port 12345 -Key "~/.ssh/id_rsa" -User "root"
#
# For RunPod:   SSH details are on the pod's "Connect" page
# For Vast.ai:  SSH details are on the instance row
# For Lambda:   SSH details are in the instance dashboard

param(
    [Parameter(Mandatory)][string]$PodHost,
    [Parameter(Mandatory)][int]$Port,
    [string]$Key = "~/.ssh/id_rsa",
    [string]$User = "root",
    [string]$RemoteDir = "/workspace/protein-classification/outputs/features"
)

$LocalDir = "outputs\features"
$Files = @(
    "esm2_embeddings_esm2_t33_650M_UR50D.npy",   # 194 MB — 650M embeddings
    "handcrafted_features.npy",                    # 130 MB — composition + physicochemical
    "feature_names.npy"                            #   0 MB — feature name list
)

Write-Host "========================================"
Write-Host "  POD FILE UPLOAD"
Write-Host "  -> ${User}@${PodHost}:${Port}"
Write-Host "  -> $RemoteDir"
Write-Host "========================================"

# Create remote directory
Write-Host "`nCreating remote directory..."
ssh -p $Port -i $Key "${User}@${PodHost}" "mkdir -p $RemoteDir"

# Upload each file
$total = 0
foreach ($file in $Files) {
    $local = Join-Path $LocalDir $file
    if (Test-Path $local) {
        $mb = [math]::Round((Get-Item $local).Length / 1MB, 1)
        Write-Host "`nUploading $file ($mb MB)..."
        scp -P $Port -i $Key $local "${User}@${PodHost}:${RemoteDir}/${file}"
        $total += $mb
        Write-Host "  Done."
    } else {
        Write-Host "`nWARNING: $local not found — skipping."
    }
}

Write-Host "`n========================================"
Write-Host "  Upload complete. Total: ${total} MB"
Write-Host ""
Write-Host "  Next steps on the pod:"
Write-Host "    bash pod_setup.sh"
Write-Host "========================================"
