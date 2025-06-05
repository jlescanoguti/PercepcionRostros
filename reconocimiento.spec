# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['reconocimiento.py'],
    pathex=[],
    binaries=[],
    datas=[('shape_predictor_68_face_landmarks.dat', '.'), ('dlib_face_recognition_resnet_model_v1.dat', '.'), ('rostros', 'rostros')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='reconocimiento',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
