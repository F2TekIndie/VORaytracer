param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [switch]$UpdateBaselines
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot "bin\x64\$Configuration\VORaytracer.App.exe"
$baselineRoot = Join-Path $projectRoot 'assets\Regression'
$outputRoot = Join-Path $projectRoot 'artifacts\image-regression'
New-Item -ItemType Directory -Force -Path $baselineRoot, $outputRoot | Out-Null

$cases = @(
    @{ Name = 'vulkan-fbx-directional'; Backend = 'vulkan'; Frames = 8; Scene = 'assets\SampleObject.fbx'; Hdr = $null },
    @{ Name = 'vulkan-gltf-hdr'; Backend = 'vulkan'; Frames = 8; Scene = 'assets\FlightHelmet\FlightHelmet.gltf'; Hdr = 'assets\SampleHDR.hdr' },
    @{ Name = 'vulkan-obj-emissive'; Backend = 'vulkan'; Frames = 8; Scene = 'assets\TexturedTriangle.obj'; Hdr = $null },
    @{ Name = 'optix-obj-emissive'; Backend = 'optix'; Frames = 16; Scene = 'assets\TexturedTriangle.obj'; Hdr = $null }
)
$executedCases = 0

foreach ($case in $cases) {
    $scenePath = Join-Path $projectRoot $case.Scene
    $hdrPath = if ($case.Hdr) { Join-Path $projectRoot $case.Hdr } else { $null }
    if (-not (Test-Path -LiteralPath $scenePath) -or ($hdrPath -and -not (Test-Path -LiteralPath $hdrPath))) {
        Write-Warning "Skipping optional image regression case '$($case.Name)' because a local asset is missing."
        continue
    }
    $capture = Join-Path $outputRoot "$($case.Name).ppm"
    $baseline = Join-Path $baselineRoot "$($case.Name).ppm"
    if (-not $UpdateBaselines -and -not (Test-Path -LiteralPath $baseline)) {
        throw "Missing baseline '$baseline'. Run with -UpdateBaselines after visually approving captures."
    }

    $env:VOR_BACKEND = $case.Backend
    $env:VOR_SCENE = $scenePath
    $env:VOR_TEST_FRAMES = [string]$case.Frames
    $env:VOR_CAPTURE_FRAME = [string]$case.Frames
    $env:VOR_CAPTURE_PATH = $capture
    $env:VOR_TEST_WIDTH = '640'
    $env:VOR_TEST_HEIGHT = '360'
    $env:VOR_HEADLESS = '1'
    $env:VOR_HIDE_UI = '1'
    $env:VOR_IMAGE_MAX_RMSE = '0.02'
    Remove-Item Env:VOR_DENOISER -ErrorAction SilentlyContinue
    Remove-Item Env:VOR_DEFAULT_PLASTIC -ErrorAction SilentlyContinue
    if ($case.Hdr) {
        $env:VOR_GLOBAL_LIGHT = 'hdr'
        $env:VOR_HDR = $hdrPath
    } else {
        $env:VOR_GLOBAL_LIGHT = 'directional'
        Remove-Item Env:VOR_HDR -ErrorAction SilentlyContinue
    }
    if ($UpdateBaselines) {
        Remove-Item Env:VOR_REFERENCE_PATH -ErrorAction SilentlyContinue
    } else {
        $env:VOR_REFERENCE_PATH = $baseline
    }

    & $executable
    if ($LASTEXITCODE -ne 0) {
        throw "Image regression case '$($case.Name)' failed with exit code $LASTEXITCODE."
    }
    if ($UpdateBaselines) {
        Copy-Item -LiteralPath $capture -Destination $baseline -Force
    }
    ++$executedCases
}

Write-Host "Image regression suite passed ($executedCases cases)."
