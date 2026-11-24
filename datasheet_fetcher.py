"""Download and cache complete, validated CMSIS-SVD source files.

Never modifies or shortens the downloaded XML. Ambiguous MCU matches fail closed.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from difflib import get_close_matches
from pathlib import Path
from urllib.parse import quote

import requests

SVD_REPO_TREE_API = 'https://api.github.com/repos/cmsis-svd/cmsis-svd-data/git/trees/main?recursive=1'
SVD_RAW_BASE = 'https://raw.githubusercontent.com/cmsis-svd/cmsis-svd-data/main/'
INDEX_CACHE = Path('./hardware_specs/svd_registry_index.json')
EXTERNAL_SVD_MIRRORS = {
    'RP2040': 'https://raw.githubusercontent.com/raspberrypi/pico-sdk/master/src/rp2040/hardware_regs/RP2040.svd',
    'RP2350': 'https://raw.githubusercontent.com/raspberrypi/pico-sdk/master/src/rp2350/hardware_regs/RP2350.svd',
    'ESP32': 'https://raw.githubusercontent.com/espressif/svd/main/svd/esp32.svd',
    'ESP32S2': 'https://raw.githubusercontent.com/espressif/svd/main/svd/esp32s2.svd',
    'ESP32S3': 'https://raw.githubusercontent.com/espressif/svd/main/svd/esp32s3.svd',
    'ESP32C3': 'https://raw.githubusercontent.com/espressif/svd/main/svd/esp32c3.svd',
    'ESP32C6': 'https://raw.githubusercontent.com/espressif/svd/main/svd/esp32c6.svd',
}


def _normalize(value):
    return re.sub(r'[^A-Z0-9]', '', value.upper())


def _validate_svd(data):
    root = ET.fromstring(data)
    if root.tag.rsplit('}', 1)[-1] != 'device':
        raise ValueError('XML root is not a CMSIS-SVD device')
    def child(name):
        return next((n for n in root if n.tag.rsplit('}', 1)[-1] == name), None)
    periphs = child('peripherals')
    if periphs is None or not any(n.tag.rsplit('}', 1)[-1] == 'peripheral' for n in periphs):
        raise ValueError('SVD has no peripherals')
    return root


def build_or_load_svd_index(force_refresh: bool = False) -> dict[str, str]:
    """Use existing cache if available; reject truncated or empty GitHub trees."""
    if INDEX_CACHE.exists() and not force_refresh:
        try:
            cached = json.loads(INDEX_CACHE.read_text(encoding='utf-8'))
            if isinstance(cached, dict) and cached:
                return cached
        except (OSError, ValueError):
            pass
    print('[INDEXING] Fetching CMSIS-SVD catalog...')
    try:
        response = requests.get(SVD_REPO_TREE_API, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get('truncated'):
            raise RuntimeError('GitHub tree was truncated; refusing incomplete index')
        index = {}
        for item in payload.get('tree', []):
            path = item.get('path', '')
            if path.startswith('data/') and path.lower().endswith('.svd'):
                name = Path(path).stem.upper()
                if name in index and index[name] != path:
                    # Avoid silent overwrite when vendors share an SVD basename.
                    index[f"{Path(path).parent.name.upper()}__{name}"] = path
                else:
                    index[name] = path
        if not index:
            raise RuntimeError('No SVD entries found in catalog')
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temp = INDEX_CACHE.with_suffix('.tmp')
        temp.write_text(json.dumps(index, indent=2), encoding='utf-8')
        temp.replace(INDEX_CACHE)
        return index
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        if INDEX_CACHE.exists():
            try:
                cached = json.loads(INDEX_CACHE.read_text(encoding='utf-8'))
                if isinstance(cached, dict) and cached:
                    print(f'[WARNING] Using cached index after refresh failure: {exc}')
                    return cached
            except (OSError, ValueError):
                pass
        raise


def _select_match(query, index):
    tokens = [_normalize(t) for t in re.findall(r'[A-Za-z0-9_-]+', query)]
    tokens = [t for t in tokens if t]
    if 'STM32M7' in tokens:
        print('[CLARIFY] Cortex-M7 is a core, not an STM32 device family; specify STM32F7/H7 part number.')
        return None
    # Exact matches first; then longest SVD stem matching the supplied full part number.
    normalized = {key: _normalize(key.split('__')[-1]) for key in index}
    for token in sorted(tokens, key=len, reverse=True):
        exact = [key for key, value in normalized.items() if value == token]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            print(f'[AMBIGUOUS] {token}: {exact}')
            return None
        prefixes = [key for key, value in normalized.items() if len(value) >= 6 and token.startswith(value)]
        if prefixes:
            longest = max(len(normalized[key]) for key in prefixes)
            best = [key for key in prefixes if len(normalized[key]) == longest]
            if len(best) == 1:
                return best[0]
            print(f'[AMBIGUOUS] {token}: {best}')
            return None
        # Query shorter than the SVD entry is only accepted if unambiguous.
        if len(token) >= 7:
            candidates = [key for key, value in normalized.items() if value.startswith(token)]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                print(f'[AMBIGUOUS] {token}: {candidates[:12]}')
                return None
    # Fuzzy results are suggestions, not automatically downloaded hardware.
    suggestions = get_close_matches(_normalize(query), list(normalized.values()), n=5, cutoff=0.85)
    if suggestions:
        print(f'[NO EXACT MATCH] Possible SVD names: {suggestions}. Please specify the MCU.')
    return None


def resolve_and_download_svd(user_board_query: str, output_dir: Path = Path('./hardware_specs')) -> Path | None:
    tokens = [_normalize(t) for t in re.findall(r'[A-Za-z0-9_-]+', user_board_query)]
    for token in tokens:
        if token in EXTERNAL_SVD_MIRRORS:
            return _fetch_and_validate_xml(token, EXTERNAL_SVD_MIRRORS[token], output_dir)
    try:
        index = build_or_load_svd_index()
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f'[ERROR] Unable to load SVD catalog: {exc}')
        return None
    matched = _select_match(user_board_query, index)
    if matched is None:
        print(f'[MISS] No unambiguous SVD for {user_board_query!r}. Use an exact MCU part number.')
        return None
    return _fetch_and_validate_xml(matched, SVD_RAW_BASE + quote(index[matched], safe='/'), output_dir)


def _fetch_and_validate_xml(chip_key: str, raw_url: str, output_dir: Path) -> Path | None:
    output_dir = Path(output_dir)
    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', chip_key)
    save_path = output_dir / f'{safe_name}.svd'
    if save_path.exists():
        try:
            _validate_svd(save_path.read_bytes())
            print(f'[CACHE HIT] Validated full SVD: {save_path}')
            return save_path
        except (ET.ParseError, ValueError, OSError) as exc:
            print(f'[INVALID CACHE] Redownloading {save_path}: {exc}')
    print(f'[DOWNLOADING] {chip_key} from {raw_url}')
    try:
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()
        _validate_svd(response.content)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = save_path.with_suffix('.svd.tmp')
        temp_path.write_bytes(response.content)
        temp_path.replace(save_path)
        print(f'[SUCCESS] Full SVD saved: {save_path} ({len(response.content)} bytes)')
        return save_path
    except (requests.RequestException, ET.ParseError, ValueError, OSError) as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}')
        return None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        resolve_and_download_svd(' '.join(sys.argv[1:]))
    else:
        print('Silicon-Compiler AI | SVD resolver (type exit to quit)')
        try:
            while True:
                query = input('MCU: ').strip()
                if query.lower() in ('exit', 'quit', 'q'):
                    break
                if query:
                    resolve_and_download_svd(query)
        except EOFError:
            pass
