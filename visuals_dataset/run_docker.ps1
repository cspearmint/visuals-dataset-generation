# PowerShell script to run Docker container for HiKER augmentation pipeline
# Usage: .\run_docker.ps1 --count 10 --max-files 3 --output-dir output

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$IMAGE = "hiker-augment"

# Check if image exists, build if not
$imageExists = docker image inspect $IMAGE 2>$null
if (-not $imageExists) {
    Write-Host "Building Docker image $IMAGE ..."
    docker build -t $IMAGE .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        exit 1
    }
}

# Build docker run command
# Mount the parent directory (repo root) so both visuals_dataset and HiKER-SGG_Alterations are accessible
$repoRoot = Split-Path -Parent $PWD
$dockerCmd = @(
    "run",
    "--rm",
    "-v", "${repoRoot}:/workspace",
    "-w", "/workspace/visuals_dataset"
)

# Only add gcloud mount if credentials exist
# gcloud on Windows saves to AppData\Roaming\gcloud
$gcloudPath = "$ENV:USERPROFILE\AppData\Roaming\gcloud"
if (Test-Path $gcloudPath) {
    Write-Host "[info] Using gcloud credentials from $gcloudPath"
    $dockerCmd += "-v", "${gcloudPath}:/root/.config/gcloud"
    $dockerCmd += "-e", "GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json"
} else {
    Write-Host "[info] No gcloud credentials found at $gcloudPath (optional)"
}

# Add image and Python command
$dockerCmd += $IMAGE
$dockerCmd += "python"
$dockerCmd += "load_waymo_images.py"
$dockerCmd += $Args

Write-Host "[info] Running: docker $($dockerCmd -join ' ')"
& docker $dockerCmd
