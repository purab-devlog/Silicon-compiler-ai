"""Lossless sensor source catalog + on-demand text retrieval.

Compatibility API: extract_sensor_spec(part, spec_dir, max_chars=...).
Default now returns the full extracted text, never an arbitrary 8000-char slice.
The original PDF/header is always preserved by sensor_fetcher.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

KEYWORDS = ("slave address", "device address", "i2c", "spi", "sccb",
            "register", "command", "opcode", "instruction", "0x")
_BINARY_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])(0[bB][01](?:[01_]*[01])?|[01]{4,8}(?:_[01]{1,8})+)(?![A-Za-z0-9_])")


def _part_name(part: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]", "", part.strip()).upper()
    if not result:
        raise ValueError("Sensor part number is empty")
    return result


def _normalize_binary_literals(text: str) -> str:
    """Optional display-only annotations; source text is retained separately."""
    def annotate(match: re.Match) -> str:
        raw = match.group(1)
        bits = re.sub(r"^0[bB]", "", raw).replace("_", "")
        if len(bits) not in (7, 8) or any(c not in "01" for c in bits):
            return raw
        return f"{raw} (0x{int(bits, 2):02X})"
    return _BINARY_LITERAL_RE.sub(annotate, text)


def _extract_from_header(header_path: Path, max_chars: int | None = None) -> str:
    """Preserve ALL header content, including multiline macros and comments."""
    text = header_path.read_text(encoding="utf-8", errors="replace")
    return text if max_chars is None else text[:max_chars]


def _extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        import pymupdf
        with pymupdf.open(pdf_path) as doc:
            return [(idx + 1, page.get_text("text", sort=True)) for idx, page in enumerate(doc)]
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pymupdf or pypdf to extract sensor PDFs") from exc
        reader = PdfReader(str(pdf_path))
        return [(idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)]


def _extract_from_pdf(pdf_path: Path, max_chars: int | None = None) -> str:
    pages = _extract_pdf_pages(pdf_path)
    text = "\n\n".join(f"--- Page {number} ---\n{content}" for number, content in pages)
    return text if max_chars is None else text[:max_chars]


def _source_paths(part: str, spec_dir: Path) -> list[Path]:
    """Use provenance manifest when available; fall back to legacy filenames."""
    manifest_path = spec_dir / f"{part}_sources.json"
    paths: list[Path] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("sources", []):
                filename = item.get("filename", "")
                if Path(filename).name != filename:
                    continue
                path = spec_dir / filename
                if path.is_file() and path.suffix.lower() in {".h", ".pdf"}:
                    paths.append(path)
        except (OSError, ValueError):
            pass
    for suffix in ("_regs.h", "_reg.h", "_datasheet.pdf"):
        path = spec_dir / f"{part}{suffix}"
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def build_sensor_catalog(sensor_part: str, spec_dir: Path = Path("./sensor_specs"),
                         output_path: Path | None = None) -> dict:
    """Index every source and every PDF page; preserve all text and macro definitions.

    JSON is a searchable *text catalog*, not a verified sensor register model.
    Macro values are strings, not assumed to be register addresses.
    """
    part, spec_dir = _part_name(sensor_part), Path(spec_dir)
    sources = _source_paths(part, spec_dir)
    if not sources:
        raise FileNotFoundError(f"No cached sensor sources for {part} in {spec_dir}")
    catalog: dict = {"sensor": part, "sources": [], "schema_version": 1}
    for path in sources:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry: dict = {"filename": path.name, "sha256": digest, "type": path.suffix.lower()}
        if path.suffix.lower() == ".pdf":
            pages = _extract_pdf_pages(path)
            entry["pages"] = [{"page": n, "text": t, "has_extractable_text": bool(t.strip())}
                              for n, t in pages]
            entry["unextractable_pages"] = [n for n, t in pages if not t.strip()]
        else:
            text = _extract_from_header(path)
            entry["text"] = text
            # Retain all macros and their source lines, including multiline definitions.
            macros = []
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if re.match(r"^\s*#\s*define\s+", line):
                    start = i + 1
                    definition = [line]
                    while definition[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                        i += 1
                        definition.append(lines[i])
                    match = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)", line)
                    if match:
                        macros.append({"name": match.group(1), "line": start,
                                       "definition": "\n".join(definition)})
                i += 1
            entry["macros"] = macros
        catalog["sources"].append(entry)
    destination = Path(output_path) if output_path is not None else spec_dir / f"{part}_catalog.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return catalog


def retrieve_sensor_context(sensor_part: str, query: str,
                            spec_dir: Path = Path("./sensor_specs"),
                            max_pages: int | None = 8) -> str:
    """Deterministic keyword retrieval for future Qwen hardware selection.

    Selects whole PDF pages (never partial pages). Header matching retrieves
    full macro definitions; the full source remains available in the catalog.
    """
    part = _part_name(sensor_part)
    path = Path(spec_dir) / f"{part}_catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else build_sensor_catalog(part, spec_dir)
    terms = [x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]*|0[xX][0-9a-fA-F]+", query) if len(x) >= 2]
    if not terms:
        raise ValueError("Retrieval query must contain searchable terms")
    ranked = []
    for source in catalog["sources"]:
        if source["type"] == ".pdf":
            for page in source["pages"]:
                content = page["text"]
                score = sum(content.lower().count(term) for term in terms)
                if score:
                    ranked.append((score, source["filename"], page["page"], content))
        else:
            for macro in source["macros"]:
                content = macro["definition"]
                score = sum(content.lower().count(term) for term in terms)
                if score:
                    ranked.append((score, source["filename"], macro["line"], content))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected = ranked if max_pages is None else ranked[:max_pages]
    return "\n\n".join(f"--- {filename} | page/line {number} | score {score} ---\n{text}"
                       for score, filename, number, text in selected)


def extract_sensor_spec(sensor_part: str, spec_dir: Path = Path("./sensor_specs"),
                        max_chars: int | None = None) -> str:
    """Return complete text of ALL cached source documents, unless explicitly limited.

    `max_chars` is retained for backwards compatibility; avoid using it when
    building the authoritative sensor catalog or passing source to retrieval.
    """
    part = _part_name(sensor_part)
    sources = _source_paths(part, Path(spec_dir))
    if not sources:
        raise FileNotFoundError(f"No cached spec found for '{part}' in {spec_dir}")
    chunks = []
    for path in sources:
        content = (_extract_from_pdf(path) if path.suffix.lower() == ".pdf"
                   else _extract_from_header(path))
        chunks.append(f"=== SOURCE: {path.name} ===\n{content}")
    text = "\n\n".join(chunks)
    return text if max_chars is None else text[:max_chars]


if __name__ == "__main__":
    part = input("Cached sensor part (e.g., BH1750): ").strip()
    directory = Path("./sensor_specs")
    catalog = build_sensor_catalog(part, directory)
    output = directory / f"{_part_name(part)}_extracted.txt"
    text = extract_sensor_spec(part, directory)
    output.write_text(text, encoding="utf-8")
    print(f"Saved complete text ({len(text)} characters) to {output}")
    print(f"Indexed {len(catalog['sources'])} sources in {_part_name(part)}_catalog.json")
