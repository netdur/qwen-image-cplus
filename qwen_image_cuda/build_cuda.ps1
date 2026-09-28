# Compiles the Windows engine's CUDA kernels and native helpers into the static
# library the qwen_image_cuda package links (see Cplus.toml [windows.link]).
# The Windows counterpart of build_cuda.sh.
#
#   $env:CUDA_ARCH = "sm_75"; qwen_image_cuda\build_cuda.ps1
#
# CUDA_ARCHS builds one fat library instead, for distribution: native code for
# every listed compute capability plus PTX for the last, which newer GPUs JIT.
#
#   $env:CUDA_ARCHS = "75 80 86 89"; qwen_image_cuda\build_cuda.ps1
#
# Environment:
#   CUDA_HOME    CUDA 12 toolkit
#   CUDNN_HOME   cuDNN 8 for CUDA 12 (holds include\cudnn.h)
#   JPEG_HOME    libjpeg-turbo (holds include\jpeglib.h)
#
# Everything is compiled against the static C runtime (/MT): cpc links C+
# programs with it, and one binary cannot mix the two runtimes.
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
if (-not $env:CUDA_HOME) { throw "set CUDA_HOME to the CUDA 12 toolkit" }
$cudaHome = $env:CUDA_HOME
if (-not $env:CUDNN_HOME) { throw "set CUDNN_HOME to cuDNN 8 for CUDA 12" }
if (-not $env:JPEG_HOME) { throw "set JPEG_HOME to libjpeg-turbo" }

# nvcc and cl need the Visual Studio x64 build environment; import it unless
# this shell already has it, and restore the caller's environment afterwards:
# vcvars puts Visual Studio's own (older) clang ahead of LLVM's on PATH, and
# cpc would pick that one up for everything built after this script.
$callerEnvironment = @{}
Get-ChildItem env: | ForEach-Object { $callerEnvironment[$_.Name] = $_.Value }
try {
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw "Visual Studio with the C++ x64 tools not found" }
        $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
        cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
        }
    }

    if ($env:CUDA_ARCHS) {
        $capabilities = @($env:CUDA_ARCHS -split "\s+" | Where-Object { $_ })
        $archFlags = @($capabilities | ForEach-Object { "-gencode=arch=compute_$_,code=sm_$_" })
        $last = $capabilities[-1]
        $archFlags += "-gencode=arch=compute_$last,code=compute_$last"
        $arch = "sm_" + ($capabilities -join ",sm_") + " + compute_$last"
    } else {
        $arch = if ($env:CUDA_ARCH) { $env:CUDA_ARCH } else { "sm_75" }
        $archFlags = @("-arch=$arch")
    }

    $sourceDir = Join-Path $here "cuda"
    $outputDir = Join-Path $sourceDir "target"
    New-Item -ItemType Directory -Force $outputDir | Out-Null
    $nvcc = Join-Path $cudaHome "bin\nvcc.exe"

    $objects = @()
    foreach ($source in Get-ChildItem (Join-Path $sourceDir "*.cu")) {
        $object = Join-Path $outputDir ($source.BaseName + ".obj")
        # Warning 221 is MSVC's math.h spelling INFINITY as a float overflow
        # (1e300 * 1e300); the value is still infinity.
        & $nvcc -O3 @archFlags -std=c++17 -diag-suppress=221 -Xcompiler "/MT /O2" `
            -I (Join-Path $env:CUDNN_HOME "include") -c $source.FullName -o $object
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed on $($source.Name)" }
        $objects += $object
    }
    # Host-side C helpers (image decoding and resampling).
    foreach ($source in Get-ChildItem (Join-Path $here "native\*.c")) {
        $object = Join-Path $outputDir ($source.BaseName + ".obj")
        & cl.exe /nologo /c /O2 /MT /D_USE_MATH_DEFINES /I (Join-Path $env:JPEG_HOME "include") `
            $source.FullName /Fo$object
        if ($LASTEXITCODE -ne 0) { throw "cl failed on $($source.Name)" }
        $objects += $object
    }
    $library = Join-Path $outputDir "qwen_image_cuda.lib"
    Remove-Item -Force -ErrorAction SilentlyContinue $library
    & lib.exe /nologo /OUT:$library @objects
    if ($LASTEXITCODE -ne 0) { throw "lib failed" }
    Write-Output "built $library ($arch)"
} finally {
    Get-ChildItem env: | Where-Object { -not $callerEnvironment.ContainsKey($_.Name) } | ForEach-Object { Remove-Item "env:$($_.Name)" }
    foreach ($name in $callerEnvironment.Keys) { Set-Item -Path "env:$name" -Value $callerEnvironment[$name] }
}
