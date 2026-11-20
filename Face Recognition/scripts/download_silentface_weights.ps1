$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$RepoDir = Join-Path $ProjectRoot "Face Recognition/third_party/Silent-Face-Anti-Spoofing"
$WeightsDir = Join-Path $ProjectRoot "Face Recognition/artifacts/antispoof/weights"
$WeightName = "2.7_80x80_MiniFASNetV2.pth"
$WeightUrl = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

New-Item -ItemType Directory -Force -Path (Split-Path $RepoDir) | Out-Null
New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

if (!(Test-Path (Join-Path $RepoDir "src/model_lib/MiniFASNet.py"))) {
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git $RepoDir
} else {
    Write-Host "Repo Silent-Face sudah ada: $RepoDir"
}

$Destination = Join-Path $WeightsDir $WeightName
if (!(Test-Path $Destination)) {
    Invoke-WebRequest -Uri $WeightUrl -OutFile $Destination
} else {
    Write-Host "Weight sudah ada: $Destination"
}

Write-Host "PAD repository: $RepoDir"
Write-Host "PAD weight: $Destination"
Get-FileHash -Algorithm SHA256 $Destination | Format-List
Write-Warning "Catat SHA256 dan commit repo. Jangan anggap PAD tervalidasi sebelum diuji pada data kamera dan serangan Anda."
