param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug'
)

$ErrorActionPreference = 'Stop'
$sourcePath = 'G:\CodingLibraries\assimp'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $projectRoot 'build\assimp'

cmake -S $sourcePath -B $buildPath -G 'Visual Studio 18 2026' -A x64 `
    -DASSIMP_BUILD_TESTS=OFF `
    -DASSIMP_BUILD_ASSIMP_TOOLS=OFF `
    -DASSIMP_INSTALL=OFF `
    -DASSIMP_WARNINGS_AS_ERRORS=OFF `
    -DBUILD_SHARED_LIBS=ON

cmake --build $buildPath --config $Configuration --target assimp --parallel

