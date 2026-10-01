"""
GlucoTwin — HUPA-UCM Dataset Downloader (v2 — Browser-discovered URLs)
DOI: 10.17632/3hbcscwz44.1

Uses the Mendeley public-api ZIP endpoint discovered via browser inspection.
No login required (CC BY 4.0 public dataset).

Usage:
    python scripts/download_dataset.py
"""

import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

# ─── Discovered via browser inspection of Mendeley Data page ─────────────────
ZIP_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/3hbcscwz44/download/1"
OUTPUT_DIR = Path("data/raw/hupa_ucm")
ZIP_PATH = OUTPUT_DIR / "hupa_ucm_dataset.zip"

# Individual preprocessed file URLs (fallback if ZIP fails)
PREPROCESSED_FILES = {
    "HUPA0001P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/8c30f957-88b6-4ebb-905a-81bd77e851e8/file_downloaded",
    "HUPA0002P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/69121215-e306-489d-a3bb-748e6201ac73/file_downloaded",
    "HUPA0003P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/0bfc1159-4d81-4d3b-b87a-1cc78464f98b/file_downloaded",
    "HUPA0004P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/c5c3380a-4323-4ad1-bc5a-bf1830b886d3/file_downloaded",
    "HUPA0005P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/8965ec1d-9585-4917-89b9-e57bce07251b/file_downloaded",
    "HUPA0006P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/02e63f0a-e42c-4a92-8575-3f83ae17ad7d/file_downloaded",
    "HUPA0007P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/b8f21d3a-3d7f-4867-b9cf-b9d371c66cb7/file_downloaded",
    "HUPA0009P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/bf28c8b6-5b9b-482b-82cb-d3202e43bb6b/file_downloaded",
    "HUPA0010P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/684f9ed2-0cee-4f6c-b597-a55c49c99744/file_downloaded",
    "HUPA0011P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/28e87893-30fc-440d-935a-cbbfc6a37049/file_downloaded",
    "HUPA0014P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/8c5d3b43-b156-45d0-8556-43b7f9c061ee/file_downloaded",
    "HUPA0015P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/9face0a8-eb7d-47bc-80e9-cf452094d7f6/file_downloaded",
    "HUPA0016P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/31264d2d-353b-4ebd-8cd3-94fa7b4ad734/file_downloaded",
    "HUPA0017P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/d1e7623e-8132-49c0-84c1-9870482fd0b2/file_downloaded",
    "HUPA0018P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/cc3973ed-675c-40cb-8fa2-0cc805a11703/file_downloaded",
    "HUPA0019P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/05517475-5854-4bf3-8a1e-f04e4d281578/file_downloaded",
    "HUPA0020P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/ebd1a465-a338-4d3b-9bda-6fb04884b5cf/file_downloaded",
    "HUPA0021P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/465355e7-5ae2-4bfd-8769-1b87e5420d0c/file_downloaded",
    "HUPA0022P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/c6f6876a-9a1b-4321-9277-606be02d49ec/file_downloaded",
    "HUPA0023P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/f2083955-63e2-4dc1-84f7-569d4979c3ae/file_downloaded",
    "HUPA0024P.csv": "https://data.mendeley.com/public-files/datasets/3hbcscwz44/files/aa2e8d6c-2190-4286-9222-fd335874cd21/file_downloaded",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Referer": "https://data.mendeley.com/datasets/3hbcscwz44/1",
}


def download_file(url: str, dest: Path, label: str, expected_mb: float = None):
    """Download a file with progress reporting."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[SKIP] {label} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    
    print(f"[DOWN] {label}" + (f" (~{expected_mb:.0f} MB)" if expected_mb else ""))
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            last_print = time.time()
            
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 3:
                            mb = downloaded / 1e6
                            if total:
                                pct = downloaded / total * 100
                                print(f"       {mb:.1f} MB / {total/1e6:.1f} MB ({pct:.0f}%)", 
                                      end="\r", flush=True)
                            else:
                                print(f"       {mb:.1f} MB", end="\r", flush=True)
                            last_print = time.time()
            
            final_mb = dest.stat().st_size / 1e6
            print(f"\n[OK]  {label} — {final_mb:.1f} MB saved")
            return True
    
    except requests.HTTPError as e:
        print(f"\n[ERR] HTTP {e.response.status_code} for {label}")
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        print(f"\n[ERR] {label}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract the downloaded ZIP archive."""
    print(f"\n[UNZIP] Extracting {zip_path.name} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            print(f"        Found {len(names)} files in archive:")
            for n in names[:20]:
                print(f"          {n}")
            if len(names) > 20:
                print(f"          ... and {len(names) - 20} more")
            zf.extractall(extract_to)
        print(f"[OK]  Extraction complete")
        return True
    except zipfile.BadZipFile:
        print(f"[ERR] Downloaded file is not a valid ZIP (may be HTML redirect)")
        return False
    except Exception as e:
        print(f"[ERR] Extraction failed: {e}")
        return False


def download_individual_files(dest_dir: Path):
    """Fallback: download individual preprocessed CSV files."""
    print("\n[FALLBACK] Downloading individual preprocessed CSV files...")
    preprocessed_dir = dest_dir / "Preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    fail = 0
    
    for fname, url in PREPROCESSED_FILES.items():
        dest = preprocessed_dir / fname
        ok = download_file(url, dest, fname)
        if ok:
            success += 1
        else:
            fail += 1
        time.sleep(0.3)  # Rate limiting
    
    return success, fail


def save_manifest(method: str, files: list, output_dir: Path):
    """Save download provenance manifest."""
    manifest = {
        "dataset": "HUPA-UCM Diabetes Dataset",
        "doi": "10.17632/3hbcscwz44.1",
        "version": "1",
        "license": "CC BY 4.0",
        "source": "https://data.mendeley.com/datasets/3hbcscwz44/1",
        "download_method": method,
        "download_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_preprocessed_patients": 21,
        "note": "HUPA0008P and HUPA0012P, HUPA0013P excluded (not in preprocessed set)",
        "files": files,
    }
    path = output_dir / "download_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK]  Manifest: {path}")


def main():
    print("=" * 70)
    print("GlucoTwin — HUPA-UCM Dataset Downloader v2")
    print("DOI: 10.17632/3hbcscwz44.1  |  License: CC BY 4.0")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output: {OUTPUT_DIR.resolve()}")
    
    # ── Strategy 1: Download full ZIP ─────────────────────────────────────────
    print(f"\n[STRATEGY 1] Downloading full dataset ZIP (~77.7 MB)...")
    print(f"             URL: {ZIP_DOWNLOAD_URL}")
    
    zip_ok = download_file(ZIP_DOWNLOAD_URL, ZIP_PATH, "hupa_ucm_dataset.zip", 77.7)
    
    if zip_ok and ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000_000:
        # Try to extract
        extract_ok = extract_zip(ZIP_PATH, OUTPUT_DIR)
        
        if extract_ok:
            # List what was extracted
            extracted = list(OUTPUT_DIR.rglob("*.csv"))
            print(f"\n[OK]  Extracted {len(extracted)} CSV files")
            for f in sorted(extracted):
                rel = f.relative_to(OUTPUT_DIR)
                print(f"      {rel}  ({f.stat().st_size / 1024:.0f} KB)")
            
            save_manifest("zip", [str(f.relative_to(OUTPUT_DIR)) for f in extracted], OUTPUT_DIR)
            
            print("\n" + "=" * 70)
            print("DOWNLOAD COMPLETE — ZIP method")
            print(f"Location: {OUTPUT_DIR.resolve()}")
            print("=" * 70)
            print("\n[NEXT] Run: python scripts/dataset_audit.py")
            return
        else:
            print("[WARN] ZIP extraction failed — trying individual file download...")
    else:
        print("[WARN] ZIP download failed or file too small — trying individual files...")
    
    # ── Strategy 2: Individual file download ──────────────────────────────────
    success, fail = download_individual_files(OUTPUT_DIR)
    
    if success > 0:
        all_csvs = list((OUTPUT_DIR / "Preprocessed").glob("*.csv"))
        print(f"\n[OK]  Downloaded {success}/{len(PREPROCESSED_FILES)} preprocessed CSVs")
        
        save_manifest(
            "individual_files",
            [f.name for f in all_csvs],
            OUTPUT_DIR
        )
        
        print("\n" + "=" * 70)
        print(f"DOWNLOAD COMPLETE — Individual files method")
        print(f"Successful: {success}  |  Failed: {fail}")
        print(f"Location: {OUTPUT_DIR.resolve()}")
        print("=" * 70)
        print("\n[NEXT] Run: python scripts/dataset_audit.py")
    else:
        print("\n" + "=" * 70)
        print("[FAILED] Could not download dataset automatically.")
        print("\nManual download instructions:")
        print("  1. Open browser and go to:")
        print("     https://data.mendeley.com/datasets/3hbcscwz44/1")
        print("  2. Click 'Download All' button (~77.7 MB ZIP)")
        print(f"  3. Extract to: {OUTPUT_DIR.resolve()}")
        print("  4. Run: python scripts/dataset_audit.py")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
