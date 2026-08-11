param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$msbuild = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'

& (Join-Path $PSScriptRoot 'Build-Assimp.ps1') -Configuration $Configuration
& $msbuild (Join-Path $projectRoot 'VORaytracer.sln') /m /p:Configuration=$Configuration /p:Platform=x64

