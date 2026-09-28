# Builds the Windows distribution into dist\: the CLI, the GUI, the C library,
# and the CUDA runtime DLLs they load, laid out as an install prefix.
#
#   dist\bin\qwen-image-cplus.exe          CLI
#   dist\bin\qwen-image-gui.exe            desktop app (Win32)
#   dist\bin\qwen_image.dll                C API, shared
#   dist\bin\*.dll                         bundled CUDA 12 / cuDNN 8 runtime
#   dist\include\qwen_image.h              C API
#   dist\lib\qwen_image.lib                import library for qwen_image.dll
#   dist\lib\qwen_image_static.lib         C API, static (link the CUDA libs yourself)
#
# Windows loads a DLL from the executable's own directory first, so the
# runtime DLLs sit in bin\ beside the binaries and the prefix can be moved as
# a whole. The GPU driver (nvcuda.dll) comes from the system.
#
# Environment:
#   CPC          C+ compiler (default: cpc)
#   BUILD_MODE   release | debug (default: release)
#   CUDA_HOME    CUDA 12 toolkit
#   CUDNN_HOME   cuDNN 8 for CUDA 12
#   JPEG_HOME    libjpeg-turbo, built static with the static C runtime
#   CUDA_ARCHS   GPU architectures to compile for (default: "75 80 86 89")
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$cpc = if ($env:CPC) { $env:CPC } else { "cpc" }
$buildMode = if ($env:BUILD_MODE) { $env:BUILD_MODE } else { "release" }
if (-not $env:CUDA_ARCHS) { $env:CUDA_ARCHS = "75 80 86 89" }
foreach ($name in "CUDA_HOME", "CUDNN_HOME", "JPEG_HOME") {
    if (-not (Test-Path "env:$name")) { throw "set $name (see the header of this script)" }
}

$cpcFlags = @(switch ($buildMode) {
    "release" { "--release" }
    "debug" { }
    default { throw "BUILD_MODE must be release or debug" }
})

function Build-Package([string]$package) {
    Push-Location (Join-Path $projectRoot $package)
    try {
        & $cpc build @cpcFlags
        if ($LASTEXITCODE -ne 0) { throw "cpc build failed in $package" }
    } finally {
        Pop-Location
    }
}

& (Join-Path $projectRoot "qwen_image_cuda\build_cuda.ps1")

# The qwen_image* packages sit side by side at the repository root and
# resolve as sibling directories; vendor\ carries only third-party packages.
Build-Package "qwen_image"
Build-Package "cli"
Build-Package "ffi"
Build-Package "gui"

$dist = Join-Path $projectRoot "dist"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
$bin = New-Item -ItemType Directory -Force (Join-Path $dist "bin")
$include = New-Item -ItemType Directory -Force (Join-Path $dist "include")
$lib = New-Item -ItemType Directory -Force (Join-Path $dist "lib")
$ffiTarget = Join-Path $projectRoot "ffi\target\$buildMode"

Copy-Item (Join-Path $projectRoot "cli\target\$buildMode\qwen-image-cplus.exe") $bin
Copy-Item (Join-Path $projectRoot "gui\target\$buildMode\gui.exe") (Join-Path $bin "qwen-image-gui.exe")
Copy-Item (Join-Path $ffiTarget "qwen_image.dll") $bin
Copy-Item (Join-Path $ffiTarget "qwen_image.lib") $lib
Copy-Item (Join-Path $ffiTarget "qwen_image.h") $include

# The static library: the FFI objects plus every static archive the link line
# names. The CUDA and cuDNN import libraries stay out; a consumer links
# cudart.lib, cublas.lib and cudnn.lib, plus ws2_32, shell32, bcrypt, ntdll
# and advapi32 for stdlib.
Push-Location (Join-Path $projectRoot "ffi")
try {
    $linkArgs = & $cpc build @cpcFlags --print-link-args
    if ($LASTEXITCODE -ne 0) { throw "cpc --print-link-args failed" }
} finally {
    Pop-Location
}
$importRoots = @($env:CUDA_HOME, $env:CUDNN_HOME) | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd("\") + "\" }
$archives = @(Join-Path $ffiTarget "libqwen_image.a")
foreach ($arg in $linkArgs) {
    $path = [IO.Path]::GetFullPath(($arg -replace "^\\\\\?\\", ""))
    if ($path -notmatch "\.(a|lib)$") { continue }
    if ($importRoots | Where-Object { $path.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }) { continue }
    $archives += $path
}
$staticLib = Join-Path $lib "qwen_image_static.lib"
$mri = @("create $staticLib") + ($archives | ForEach-Object { "addlib $_" }) + @("save", "end")
$mri -join "`n" | & llvm-ar -M
if ($LASTEXITCODE -ne 0) { throw "llvm-ar failed to merge the static library" }

# The CUDA runtime, beside the binaries. cuDNN 8 is a small dispatcher that
# loads its sub-libraries by name from the same search path; the engine's
# convolutions need only the inference pair.
$runtime = @(
    (Join-Path $env:CUDA_HOME "bin\cudart64_12.dll"),
    (Join-Path $env:CUDA_HOME "bin\cublas64_12.dll"),
    (Join-Path $env:CUDA_HOME "bin\cublasLt64_12.dll"),
    (Join-Path $env:CUDNN_HOME "bin\cudnn64_8.dll"),
    (Join-Path $env:CUDNN_HOME "bin\cudnn_ops_infer64_8.dll"),
    (Join-Path $env:CUDNN_HOME "bin\cudnn_cnn_infer64_8.dll")
)
foreach ($dll in $runtime) {
    if (-not (Test-Path $dll)) { throw "cannot find $dll (check CUDA_HOME / CUDNN_HOME)" }
    Copy-Item $dll $bin
}

# Every binary resolves every DLL it imports from bin\ or from the system, the
# Windows form of the Linux build's ldd check. The import table is read rather
# than the binary run, so a missing DLL is named instead of reported as a
# failure to start.
$systemDirectory = Join-Path $env:SystemRoot "System32"
foreach ($binary in Get-ChildItem $bin -Include *.exe, *.dll -Recurse) {
    $imports = & llvm-readobj --coff-imports $binary.FullName |
        Select-String "^\s*Name: (.+\.dll)\s*$" | ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
    foreach ($dll in $imports) {
        if ($dll -match "^(api|ext)-ms-") { continue }
        if (-not (Test-Path (Join-Path $bin $dll)) -and -not (Test-Path (Join-Path $systemDirectory $dll))) {
            throw "$($binary.Name) imports $dll, which is neither in dist\bin nor in the system"
        }
    }
}

# The C API links and answers, shared and static, and the CLI starts, all with
# nothing on PATH but the system. No GPU needed: the smoke test only
# exercises validation.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$smoke = Join-Path ([IO.Path]::GetTempPath()) ("qwen-image-smoke-" + [Guid]::NewGuid())
New-Item -ItemType Directory $smoke | Out-Null
try {
    $source = Join-Path $projectRoot "tests\ffi_smoke.c"
    $consumerLibs = @(
        (Join-Path $env:CUDA_HOME "lib\x64\cudart.lib"),
        (Join-Path $env:CUDA_HOME "lib\x64\cublas.lib"),
        (Join-Path $env:CUDNN_HOME "lib\x64\cudnn.lib"),
        "ws2_32.lib", "shell32.lib", "bcrypt.lib", "ntdll.lib", "advapi32.lib"
    ) -join " "
    $compile = "cl /nologo /W4 /WX /MT `"$source`" /I `"$include`" /Fo`"$smoke\\`""
    cmd /c "`"$vcvars`" >nul && $compile `"$lib\qwen_image.lib`" /Fe`"$bin\ffi_smoke_shared.exe`" && $compile `"$staticLib`" $consumerLibs /Fe`"$bin\ffi_smoke_static.exe`""
    if ($LASTEXITCODE -ne 0) { throw "the C API smoke test does not compile" }
    $systemPath = "$env:SystemRoot\System32;$env:SystemRoot"
    $savedPath = $env:PATH
    $env:PATH = $systemPath
    try {
        foreach ($test in "ffi_smoke_shared", "ffi_smoke_static") {
            & (Join-Path $bin "$test.exe")
            if ($LASTEXITCODE -ne 0) { throw "$test failed" }
        }
        $usage = & (Join-Path $bin "qwen-image-cplus.exe") 2>&1 | Out-String
        if ($usage -notmatch "usage: qwen-image-cplus") { throw "the CLI does not start from dist\bin" }
    } finally {
        $env:PATH = $savedPath
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $bin "ffi_smoke_shared.exe"), (Join-Path $bin "ffi_smoke_static.exe")
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $smoke
}

Write-Output "distribution ready: $dist"
