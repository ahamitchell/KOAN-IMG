; KOAN.img Inno Setup Installer Script
; Builds a Windows installer that sets up the app directory structure
; and creates shortcuts.
;
; Prerequisites:
;   1. Install Inno Setup: https://jrsoftware.org/isdl.php
;   2. Build the launcher exe:
;      pip install pyinstaller
;      pyinstaller --onefile --name "KOAN.img" --icon=icon.ico launcher.py
;   3. Open this .iss file in Inno Setup Compiler and click Build

#define MyAppName "KOAN.img"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "KOAN"
#define MyAppURL "https://github.com/ahamitchell/KOAN-IMG"
#define MyAppExeName "KOAN.img.exe"

[Setup]
AppId={{A3D7F8E2-1B4C-4E9A-B5D6-8C2F0A1E3D7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=installer_output
OutputBaseFilename=KOAN.img_Setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; The launcher executable
Source: "dist\KOAN.img.exe"; DestDir: "{app}"; Flags: ignoreversion

; App source code
Source: "ui_app.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "common.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "embedder.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "features.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "query.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "captioner.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "index_images_chunked.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "video_api.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "video_llm.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "video_tab.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "version.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "updater.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}\app"; Flags: ignoreversion

; Collage module
Source: "collage\__init__.py"; DestDir: "{app}\app\collage"; Flags: ignoreversion
Source: "collage\build_shape_index.py"; DestDir: "{app}\app\collage"; Flags: ignoreversion
Source: "collage\mix_compose.py"; DestDir: "{app}\app\collage"; Flags: ignoreversion
Source: "collage\mix_query.py"; DestDir: "{app}\app\collage"; Flags: ignoreversion
Source: "collage\shape_features.py"; DestDir: "{app}\app\collage"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
