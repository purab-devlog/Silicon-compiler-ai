"""Retrieve and preserve sensor source documents without summarizing or truncating them.

Public compatibility API: fetch_sensor_spec(part, output_dir) -> list[Path].
A manifest records provenance; discovery is best-effort, not proof of authenticity.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

HEADERS = {"User-Agent": "Silicon-Compiler-AI/2.0", "Accept": "application/vnd.github+json"}
DIRECT_SENSOR_HEADERS = {
    "OV2640": "https://raw.githubusercontent.com/espressif/esp32-camera/master/sensors/private_include/ov2640_regs.h",
    "OV7670": "https://raw.githubusercontent.com/espressif/esp32-camera/master/sensors/private_include/ov7670_regs.h",
    "OV5640": "https://raw.githubusercontent.com/espressif/esp32-camera/master/sensors/private_include/ov5640_regs.h",
    "GC0308": "https://raw.githubusercontent.com/espressif/esp32-camera/master/sensors/private_include/gc0308_regs.h",
}
MAX_FILE_BYTES = 30 * 1024 * 1024


def _part_name(part: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]", "", part.strip()).upper()
    if not name:
        raise ValueError("Sensor part number is empty")
    return name


def _valid_bytes(data: bytes, kind: str) -> bool:
    if not data or len(data) > MAX_FILE_BYTES:
        return False
    if kind == "pdf":
        return data[:1024].lstrip().startswith(b"%PDF-")
    if b"\x00" in data[:4096]:
        return False
    return bool(re.search(rb"(?m)^\s*#\s*(?:define|ifndef|include|pragma)\b", data))


def _download(session: requests.Session, url: str, kind: str) -> bytes | None:
    try:
        with session.get(url, timeout=(8, 30), stream=True) as response:
            if response.status_code != 200:
                return None
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    return None
                chunks.append(chunk)
            data = b"".join(chunks)
            return data if _valid_bytes(data, kind) else None
    except requests.RequestException as exc:
        print(f"  [NETWORK] {url}: {exc}")
        return None


def _github_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    try:
        response = session.get(url, headers=HEADERS, params=params, timeout=(8, 20))
        if response.status_code != 200:
            return {}
        data = response.json()
        return data if isinstance(data, dict) else {}
    except (requests.RequestException, ValueError):
        return {}


def _candidate_score(path: str, part: str) -> int:
    """Rank likely sensor source files; never mistake the rank for verification."""
    low, needle = path.lower(), part.lower()
    basename = Path(low).name
    if needle not in low:
        return -1
    if not re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", low):
        # Allow common filenames like bh1750_reg.h, but not bh17501.pdf.
        if not basename.startswith(tuple(needle + suffix for suffix in ("_", "-", "."))):
            return -1
    score = 10 if needle in basename else 0
    if any(x in low for x in ("datasheet", "data_sheet", "register", "_reg", "driver")):
        score += 5
    if any(x in low for x in ("test/", "tests/", "example/", "examples/")):
        score -= 4
    return score


def fetch_sensor_spec(sensor_part: str, output_dir: Path = Path("./sensor_specs")) -> list[Path]:
    """Fetch original headers/PDFs; cache each distinct source and record its origin.

    Existing naming convention is retained for the primary header/PDF. Additional
    sources use distinct filenames, preventing overwrites by later repositories.
    """
    part = _part_name(sensor_part)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{part}_sources.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except (ValueError, OSError):
        manifest = {}
    manifest = {"sensor": part, "sources": manifest.get("sources", [])}
    files: list[Path] = []
    known_hashes: set[str] = set()
    for entry in manifest["sources"]:
        path = output_dir / Path(entry.get("filename", "")).name
        if path.is_file() and _valid_bytes(path.read_bytes(), entry.get("kind", "header")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest == entry.get("sha256"):
                files.append(path)
                known_hashes.add(digest)
    if files:
        print(f"[CACHE] Found {len(files)} verified cached source(s) for {part}")

    def save(data: bytes | None, kind: str, url: str, repository: str) -> None:
        if data is None:
            return
        digest = hashlib.sha256(data).hexdigest()
        if digest in known_hashes:
            return
        suffix = ".pdf" if kind == "pdf" else ".h"
        conventional = f"{part}_datasheet.pdf" if kind == "pdf" else f"{part}_regs.h"
        path = output_dir / conventional
        if path.exists():
            # Preserve existing content; do not overwrite unrelated sources.
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                path = output_dir / f"{part}_{digest[:12]}{suffix}"
        path.write_bytes(data)
        known_hashes.add(digest)
        files.append(path)
        manifest["sources"].append({
            "filename": path.name, "kind": kind, "url": url,
            "repository": repository, "sha256": digest,
            "size_bytes": len(data), "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "verified_by_manufacturer": False,
        })
        print(f"  [SAVED] {path.name} ({len(data)} bytes) from {repository}")

    print(f"[SENSOR] Resolving {part}")
    with requests.Session() as session:
        if part in DIRECT_SENSOR_HEADERS:
            url = DIRECT_SENSOR_HEADERS[part]
            save(_download(session, url, "header"), "header", url, "espressif/esp32-camera")

        st_url = ("https://raw.githubusercontent.com/STMicroelectronics/"
                  f"STMems_Standard_C_drivers/master/{part.lower()}_STdC/driver/{part.lower()}_reg.h")
        save(_download(session, st_url, "header"), "header", st_url,
             "STMicroelectronics/STMems_Standard_C_drivers")

        search = _github_json(session, "https://api.github.com/search/repositories",
                              {"q": part, "sort": "stars", "order": "desc", "per_page": 8})
        candidates: list[tuple[int, str, str, str]] = []
        for repo in search.get("items", [])[:8]:
            full_name = repo.get("full_name", "")
            branch = repo.get("default_branch", "main")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", full_name):
                continue
            tree = _github_json(session, f"https://api.github.com/repos/{full_name}/git/trees/{quote(branch, safe='')}?recursive=1")
            if tree.get("truncated"):
                print(f"  [WARNING] GitHub tree for {full_name} was truncated; results may be incomplete")
            for item in tree.get("tree", []):
                path = item.get("path", "")
                if item.get("type") != "blob":
                    continue
                kind = "pdf" if path.lower().endswith(".pdf") else "header" if path.lower().endswith(".h") else None
                score = _candidate_score(path, part)
                if kind and score >= 10:
                    candidates.append((score + (3 if kind == "pdf" else 0), full_name, branch, path))
        # Bound network requests, but never truncate any downloaded document.
        for _, repo, branch, path in sorted(candidates, reverse=True)[:12]:
            url = f"https://raw.githubusercontent.com/{repo}/{quote(branch, safe='')}/{quote(path, safe='/')}"
            kind = "pdf" if path.lower().endswith(".pdf") else "header"
            save(_download(session, url, kind), kind, url, repo)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not files:
        print(f"[MISS] No usable sensor sources found for {part}. Provide a verified datasheet manually.")
    return files


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fetch_sensor_spec(" ".join(sys.argv[1:]))
    else:
        while True:
            try:
                part = input("Sensor part number (or exit): ").strip()
            except EOFError:
                break
            if not part or part.lower() in {"exit", "quit", "q"}:
                break
            fetch_sensor_spec(part)
