"""
build_pdf.py — Converts documentation.html into ToxiGuard_Project_Documentation.pdf
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def generate_pdf():
    base_dir = Path(__file__).resolve().parent
    html_file = base_dir / "documentation.html"
    pdf_file = base_dir / "ToxiGuard_Project_Documentation.pdf"

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    edge_exe = None
    for p in edge_candidates:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("[ERROR] Microsoft Edge executable not found.")
        sys.exit(1)

    print(f"Reading HTML template from {html_file.resolve()}...")
    print(f"Rendering PDF via {edge_exe}...")

    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_file.resolve()}",
        str(html_file.resolve())
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_file.exists():
        print(f"[SUCCESS] PDF generated successfully: {pdf_file.resolve()} ({pdf_file.stat().st_size:,} bytes)")
    else:
        print(f"[ERROR] Failed to generate PDF: {res.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    generate_pdf()
