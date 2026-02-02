# app/api/gpu/cad/export_dwg.py
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Tuple, Optional


def convert_dxf_to_dwg(dxf_path: str, dwg_path: str) -> Tuple[bool, str]:
    """
    REAL DWG conversion (recommended):
      - Uses ODA File Converter (Teigha) if installed.

    You must install ODA File Converter on the PC (or later on POD):
      https://www.opendesign.com/guestfiles/oda_file_converter

    This function searches:
      - env var ODA_FILE_CONVERTER_EXE
      - common install paths

    Returns (ok, message).
    """

    if not os.path.isfile(dxf_path):
        return False, f"DXF not found: {dxf_path}"

    converter = _find_oda_converter()
    if not converter:
        return False, "ODA File Converter not found. Set ODA_FILE_CONVERTER_EXE or install ODA File Converter."

    in_dir = os.path.dirname(os.path.abspath(dxf_path))
    out_dir = os.path.dirname(os.path.abspath(dwg_path))
    os.makedirs(out_dir, exist_ok=True)

    # ODA converter works on directories; we convert the DXF in its folder
    # Output DWG will land in out_dir with same base filename typically.
    base = os.path.splitext(os.path.basename(dxf_path))[0]
    expected_out = os.path.join(out_dir, f"{base}.dwg")

    # Command format (typical):
    # ODAFileConverter.exe <InputFolder> <OutputFolder> <OutputVersion> <OutputType> <Recurse> <Audit> <InputFilter>
    # OutputVersion: "ACAD2018"
    # OutputType: "DWG"
    # Recurse: "0"
    # Audit: "1"
    # InputFilter: "*.dxf"
    cmd = [
        converter,
        in_dir,
        out_dir,
        "ACAD2018",
        "DWG",
        "0",
        "1",
        "*.dxf",
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            return False, f"ODA conversion failed rc={p.returncode}: {p.stderr[:1000]} {p.stdout[:1000]}"

        # ODA writes output in out_dir; ensure it exists then rename to requested dwg_path
        if os.path.isfile(expected_out):
            shutil.copyfile(expected_out, dwg_path)
            return True, "DWG written via ODA File Converter."
        # Some versions may output in input dir; fallback search
        alt = os.path.join(in_dir, f"{base}.dwg")
        if os.path.isfile(alt):
            shutil.copyfile(alt, dwg_path)
            return True, "DWG written via ODA File Converter (alt path)."

        return False, "ODA ran but DWG output not found."
    except Exception as exc:
        return False, f"ODA conversion error: {exc}"


def _find_oda_converter() -> Optional[str]:
    env = os.getenv("ODA_FILE_CONVERTER_EXE", "").strip()
    if env and os.path.isfile(env):
        return env

    candidates = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files\ODA File Converter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA File Converter\ODAFileConverter.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None
