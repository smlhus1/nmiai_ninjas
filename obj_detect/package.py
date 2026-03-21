"""
Build submission.zip for NM i AI sandbox.

Validates:
  - run.py is at zip root (not in subfolder)
  - No forbidden imports (os, subprocess, socket)
  - Total size <= 420 MB
  - Required files present (run.py, model weights)
  - File types are safe (.py, .pt, .json, .txt, .npy)
"""

import argparse
import re
import zipfile
from pathlib import Path


FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil",
    "yaml",  # use json instead
    "requests", "urllib", "http",
    "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}
FORBIDDEN_CALLS = {"eval(", "exec(", "compile(", "__import__("}
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg", ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_SIZE_MB = 420


def parse_args():
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument("--output", type=str, default="submission.zip",
                        help="Output zip path")
    parser.add_argument("--run-py", type=str, default="run.py",
                        help="Path to run.py")
    parser.add_argument("--detect-model", type=str, default="detect_model.pt",
                        help="Path to detection model weights")
    parser.add_argument("--embeddings", type=str, default="weights/reference_embeddings.pt",
                        help="Path to reference embeddings")
    parser.add_argument("--metadata", type=str, default="weights/metadata.json",
                        help="Path to metadata.json")
    parser.add_argument("--classify-model", type=str, default=None,
                        help="Path to optional classifier model")
    parser.add_argument("--extra", type=str, nargs="*", default=[],
                        help="Extra files to include")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check only, don't create zip")
    return parser.parse_args()


def check_forbidden_imports(py_path: Path) -> list[str]:
    """Check Python file for forbidden imports."""
    errors = []
    content = py_path.read_text(encoding="utf-8")

    for line_num, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check "import os", "import subprocess", etc.
        match = re.match(r"^import\s+(\w+)", stripped)
        if match and match.group(1) in FORBIDDEN_IMPORTS:
            errors.append(f"  Line {line_num}: '{stripped}' — '{match.group(1)}' is FORBIDDEN")

        # Check "from os import ..."
        match = re.match(r"^from\s+(\w+)\s+import", stripped)
        if match and match.group(1) in FORBIDDEN_IMPORTS:
            errors.append(f"  Line {line_num}: '{stripped}' — '{match.group(1)}' is FORBIDDEN")

    return errors


def main():
    args = parse_args()

    errors = []
    warnings = []
    files_to_pack = []

    # --- Check run.py ---
    run_py = Path(args.run_py)
    if not run_py.exists():
        errors.append(f"run.py not found: {run_py}")
    else:
        import_errors = check_forbidden_imports(run_py)
        if import_errors:
            errors.append(f"Forbidden imports in {run_py}:")
            errors.extend(import_errors)
        files_to_pack.append(("run.py", run_py))  # Must be at zip root
        print(f"  [OK] run.py ({run_py.stat().st_size / 1024:.1f} KB)")

    # --- Check detection model ---
    detect_model = Path(args.detect_model)
    if not detect_model.exists():
        errors.append(f"Detection model not found: {detect_model}")
    else:
        size_mb = detect_model.stat().st_size / (1024 * 1024)
        files_to_pack.append((detect_model.name, detect_model))
        print(f"  [OK] {detect_model.name} ({size_mb:.1f} MB)")

    # --- Check embeddings ---
    embeddings = Path(args.embeddings)
    if not embeddings.exists():
        errors.append(f"Reference embeddings not found: {embeddings}")
    else:
        size_mb = embeddings.stat().st_size / (1024 * 1024)
        files_to_pack.append(("reference_embeddings.pt", embeddings))
        print(f"  [OK] reference_embeddings.pt ({size_mb:.1f} MB)")

    # --- Check metadata ---
    metadata = Path(args.metadata)
    if metadata.exists():
        files_to_pack.append(("metadata.json", metadata))
        print(f"  [OK] metadata.json")
    else:
        warnings.append(f"metadata.json not found: {metadata}")

    # --- Check optional classifier ---
    if args.classify_model:
        cls_model = Path(args.classify_model)
        if cls_model.exists():
            size_mb = cls_model.stat().st_size / (1024 * 1024)
            files_to_pack.append((cls_model.name, cls_model))
            print(f"  [OK] {cls_model.name} ({size_mb:.1f} MB)")
        else:
            warnings.append(f"Classifier model not found: {cls_model}")

    # --- Extra files ---
    for extra in args.extra:
        p = Path(extra)
        if p.exists():
            if p.suffix not in ALLOWED_EXTENSIONS:
                warnings.append(f"Unusual file type: {p} ({p.suffix})")
            files_to_pack.append((p.name, p))
            print(f"  [OK] {p.name}")
        else:
            warnings.append(f"Extra file not found: {p}")

    # --- Check file types ---
    for zip_name, local_path in files_to_pack:
        suffix = Path(zip_name).suffix
        if suffix not in ALLOWED_EXTENSIONS:
            warnings.append(f"Unusual file type in submission: {zip_name} ({suffix})")

    # --- Check total size ---
    total_size = sum(p.stat().st_size for _, p in files_to_pack if p.exists())
    total_mb = total_size / (1024 * 1024)

    if total_mb > MAX_SIZE_MB:
        errors.append(f"Total size {total_mb:.1f} MB exceeds {MAX_SIZE_MB} MB limit!")
    else:
        print(f"\n  Total size: {total_mb:.1f} MB / {MAX_SIZE_MB} MB")

    # --- Report ---
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print("\nSubmission NOT created. Fix errors first.")
        return

    if args.dry_run:
        print("\n  [DRY RUN] Checks passed. Would create zip with:")
        for zip_name, _ in files_to_pack:
            print(f"    {zip_name}")
        return

    # --- Create zip ---
    output = Path(args.output)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for zip_name, local_path in files_to_pack:
            zf.write(local_path, zip_name)
            print(f"  Added: {zip_name}")

    final_size = output.stat().st_size / (1024 * 1024)
    print(f"\nSubmission created: {output} ({final_size:.1f} MB)")

    # Verify
    with zipfile.ZipFile(output, "r") as zf:
        names = zf.namelist()
        if "run.py" not in names:
            print("  ERROR: run.py not at zip root!")
        else:
            print(f"  Verified: {len(names)} files, run.py at root")


if __name__ == "__main__":
    main()
