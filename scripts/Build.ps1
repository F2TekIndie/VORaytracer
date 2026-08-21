param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',

    [string]$MSBuildPath
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not $MSBuildPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $MSBuildPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' |
            Select-Object -First 1
    }
}

if (-not $MSBuildPath) {
    $msbuildCommand = Get-Command 'MSBuild.exe' -ErrorAction SilentlyContinue
    if ($msbuildCommand) {
        $MSBuildPath = $msbuildCommand.Source
    }
}

if (-not $MSBuildPath -or -not (Test-Path -LiteralPath $MSBuildPath)) {
    throw 'MSBuild was not found. Install Visual Studio with Desktop development with C++, or pass -MSBuildPath.'
}

$solution = Join-Path $projectRoot 'VORaytracer.sln'
& $MSBuildPath $solution /m /p:Configuration=$Configuration /p:Platform=x64
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
