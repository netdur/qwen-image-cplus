; The Windows installer: packs the distribution scripts\build-windows.ps1
; already built and verified in dist\; nothing is compiled here.
;
;   iscc /DAppVersion=0.2.2 windows\qwen-image-cplus.iss
;
; It installs the prefix as dist\ lays it out (bin\, include\, lib\) under
; Program Files, or under the user's profile when they choose to install for
; themselves only, adds a Start menu entry for the app, and offers to put bin\
; on PATH for the CLI. The bundled CUDA 12 / cuDNN 8 DLLs stay in bin\ beside
; the binaries; the GPU driver comes from the system.

#ifndef AppVersion
  #error Pass the version: iscc /DAppVersion=<version> windows\qwen-image-cplus.iss
#endif

#define AppName "Qwen Image"
#define AppExe "qwen-image-gui.exe"

[Setup]
; Never change AppId: it is how an upgrade finds the installation it replaces.
AppId={{8D3C2B7E-5F41-4A9C-9E26-1B7F0C6A4D53}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=qwen-image-cplus contributors
AppPublisherURL=https://github.com/netdur/qwen-image-cplus
AppSupportURL=https://github.com/netdur/qwen-image-cplus/issues
AppUpdatesURL=https://github.com/netdur/qwen-image-cplus/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
SetupIconFile=qwen-image-cplus.ico
UninstallDisplayIcon={app}\qwen-image-cplus.ico
UninstallDisplayName={#AppName}
; Program Files by default; the user may install for themselves only instead.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ChangesEnvironment=yes
CloseApplications=yes
WizardStyle=modern
OutputDir=..
OutputBaseFilename=qwen-image-cplus-x86_64-pc-windows-msvc-setup
; The CUDA libraries are most of the 1.3 GB; LZMA2 in one solid block packs
; them about in half.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

[Tasks]
Name: addtopath; Description: "Add the qwen-image-cplus command to PATH"
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade replaces bin\ whole, so no DLL from an older version lingers.
Type: filesandordirs; Name: "{app}\bin"

[Files]
Source: "..\dist\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs
Source: "..\dist\include\*"; DestDir: "{app}\include"; Flags: ignoreversion recursesubdirs
Source: "..\dist\lib\*"; DestDir: "{app}\lib"; Flags: ignoreversion recursesubdirs
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "qwen-image-cplus.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\bin\{#AppExe}"; WorkingDir: "{app}\bin"; IconFilename: "{app}\qwen-image-cplus.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\bin\{#AppExe}"; WorkingDir: "{app}\bin"; IconFilename: "{app}\qwen-image-cplus.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\bin\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  SystemEnvironment = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  UserEnvironment = 'Environment';

// PATH lives in the machine environment for an install for everyone, and in
// the user's for an install for themselves only.
procedure PathKey(var Root: Integer; var Key: String);
begin
  if IsAdminInstallMode then
  begin
    Root := HKEY_LOCAL_MACHINE;
    Key := SystemEnvironment;
  end
  else
  begin
    Root := HKEY_CURRENT_USER;
    Key := UserEnvironment;
  end;
end;

// The position of Dir in the ;-separated Path, ignoring case and a trailing
// backslash, or 0.
function PathEntry(const Path, Dir: String): Integer;
begin
  Result := Pos(';' + Uppercase(RemoveBackslashUnlessRoot(Dir)) + ';',
                ';' + Uppercase(Path) + ';');
end;

procedure AddToPath(const Dir: String);
var
  Root: Integer;
  Key, Path: String;
begin
  PathKey(Root, Key);
  if not RegQueryStringValue(Root, Key, 'Path', Path) then Path := '';
  if PathEntry(Path, Dir) > 0 then exit;
  // A Path that ends in ';' keeps ending in one, so removing Dir again gives
  // back exactly the Path it found.
  if Path = '' then
    Path := Dir
  else if Path[Length(Path)] = ';' then
    Path := Path + Dir + ';'
  else
    Path := Path + ';' + Dir;
  RegWriteExpandStringValue(Root, Key, 'Path', Path);
end;

procedure RemoveFromPath(const Dir: String);
var
  Root, Index: Integer;
  Key, Path: String;
begin
  PathKey(Root, Key);
  if not RegQueryStringValue(Root, Key, 'Path', Path) then exit;
  Index := PathEntry(Path, Dir);
  if Index = 0 then exit;
  // Index is the entry's position in ';' + Path + ';', which is where its
  // leading ';' sits in Path; the entry plus one separator goes.
  Path := ';' + Path + ';';
  Delete(Path, Index, Length(RemoveBackslashUnlessRoot(Dir)) + 1);
  Path := Copy(Path, 2, Length(Path) - 2);
  RegWriteExpandStringValue(Root, Key, 'Path', Path);
end;

// The engine needs the NVIDIA driver's nvcuda.dll; without it the app starts
// but cannot generate. Warn, but let the install go on (a driver can come
// later).
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not FileExists(ExpandConstant('{sys}\nvcuda.dll')) then
    Result := SuppressibleMsgBox(
      'No NVIDIA driver was found. Qwen Image runs on NVIDIA GPUs (Turing or ' +
      'newer) and needs the NVIDIA driver to generate images.' + #13#10#13#10 +
      'Install anyway?', mbConfirmation, MB_YESNO, IDYES) = IDYES;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}\bin'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RemoveFromPath(ExpandConstant('{app}\bin'));
end;
