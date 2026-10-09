# Installs what the Windows build links against into one folder, from pinned
# and hash-checked downloads, and prints the environment build-windows.ps1
# reads. Used by the release workflow; works the same on a developer machine.
#
#   scripts\install-windows-deps.ps1 C:\deps
#
#   DESTINATION\cuda           CUDA 12.6: nvcc, cudart, cuBLAS, CCCL   -> CUDA_HOME
#   DESTINATION\cudnn          cuDNN 8.9.7 for CUDA 12                 -> CUDNN_HOME
#   DESTINATION\libjpeg-turbo  libjpeg-turbo 3.2.0, static, /MT        -> JPEG_HOME
#
# CUDA comes from NVIDIA's per-component redistributable archives rather than
# the installer: no administrator, no display driver, and only the pieces the
# engine compiles and links against. libjpeg-turbo is built from source
# because it must use the static C runtime cpc links C+ programs with; its
# SIMD path is used when NASM is on PATH, and decodes the same pixels as the
# C path either way.
param([Parameter(Mandatory = $true)][string]$Destination)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$cudaRedist = "https://developer.download.nvidia.com/compute/cuda/redist"
$cuda = @(
    @{ path = "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.6.85-archive.zip";
       sha256 = "3fb9f76b87c37d02f947354be89b718ad5f2c76b6ab47995265bfa3a068a5e14" },
    @{ path = "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.6.77-archive.zip";
       sha256 = "7a313bc0c93b1a50bb03aa9783a199ae70c3b66e2d8084da65e8254a8577b925" },
    @{ path = "libcublas/windows-x86_64/libcublas-windows-x86_64-12.6.4.1-archive.zip";
       sha256 = "1a87ec80f8c0e5a39badc87010d479930c5b63abd788b3a05bd688a5980a3d07" },
    @{ path = "cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.6.77-archive.zip";
       sha256 = "469b783b6731964170b3d27ae214e97e2a63ec93c57ace3ed4052e6d01261f2e" }
)
$cudnn = @{
    url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip";
    sha256 = "94fc17af8e83a26cc5d231ed23981b28c29c3fc2e87b1844ea3f46486f481df5"
}
$jpeg = @{
    url = "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz";
    sha256 = "6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e"
}

New-Item -ItemType Directory -Force $Destination | Out-Null
$Destination = (Resolve-Path $Destination).Path
$work = Join-Path $Destination "downloads"
New-Item -ItemType Directory -Force $work | Out-Null

function Get-Verified([string]$url, [string]$sha256) {
    $file = Join-Path $work ([IO.Path]::GetFileName($url))
    if (-not (Test-Path $file) -or (Get-FileHash $file -Algorithm SHA256).Hash -ne $sha256.ToUpper()) {
        & curl.exe --fail --location --retry 3 --silent --show-error --output $file $url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    }
    $actual = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($actual -ne $sha256.ToUpper()) { throw "hash mismatch for $url`n  expected $sha256`n  got      $actual" }
    return $file
}

# Each archive holds one top-level folder whose bin/include/lib merge into the
# destination, the layout of an installed toolkit.
function Expand-Merged([string]$archive, [string]$target) {
    $unpack = Join-Path $work "unpack"
    if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
    Expand-Archive -Path $archive -DestinationPath $unpack
    $root = Get-ChildItem $unpack -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force $target | Out-Null
    Copy-Item -Path (Join-Path $root.FullName "*") -Destination $target -Recurse -Force
    Remove-Item -Recurse -Force $unpack
}

$cudaHome = Join-Path $Destination "cuda"
if (-not (Test-Path (Join-Path $cudaHome "bin\nvcc.exe"))) {
    foreach ($component in $cuda) {
        $archive = Get-Verified "$cudaRedist/$($component.path)" $component.sha256
        Expand-Merged $archive $cudaHome
    }
}
& (Join-Path $cudaHome "bin\nvcc.exe") --version | Select-Object -Last 1

$cudnnHome = Join-Path $Destination "cudnn"
if (-not (Test-Path (Join-Path $cudnnHome "include\cudnn.h"))) {
    Expand-Merged (Get-Verified $cudnn.url $cudnn.sha256) $cudnnHome
}

$jpegHome = Join-Path $Destination "libjpeg-turbo"
if (-not (Test-Path (Join-Path $jpegHome "lib\jpeg-static.lib"))) {
    $tarball = Get-Verified $jpeg.url $jpeg.sha256
    $source = Join-Path $work "libjpeg-turbo-3.2.0"
    if (Test-Path $source) { Remove-Item -Recurse -Force $source }
    & tar.exe -xzf $tarball -C $work
    if ($LASTEXITCODE -ne 0) { throw "could not unpack $tarball" }
    $simd = if (Get-Command nasm.exe -ErrorAction SilentlyContinue) { "ON" } else { "OFF" }
    $build = Join-Path $work "libjpeg-turbo-build"
    & cmake -S $source -B $build -G "Visual Studio 17 2022" -A x64 `
        -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_CRT_DLL=OFF -DWITH_TURBOJPEG=OFF `
        "-DWITH_SIMD=$simd" "-DCMAKE_INSTALL_PREFIX=$jpegHome"
    if ($LASTEXITCODE -ne 0) { throw "libjpeg-turbo configure failed" }
    & cmake --build $build --config Release --target install
    if ($LASTEXITCODE -ne 0) { throw "libjpeg-turbo build failed" }
    Write-Output "libjpeg-turbo built (SIMD $simd)"
}

Remove-Item -Recurse -Force $work
Write-Output "CUDA_HOME=$cudaHome"
Write-Output "CUDNN_HOME=$cudnnHome"
Write-Output "JPEG_HOME=$jpegHome"
