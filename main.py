"""Silicon-Compiler AI FastAPI orchestrator.

Setup stores an explicit, isolated hardware session. Generation consumes that
session, validates the plan and invokes the hardware-grounded code generator.
No truncated context, inferred SVD match or silent hardware fallback is used.
"""
from __future__ import annotations

import json
import queue
import threading
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from datasheet_fetcher import EXTERNAL_SVD_MIRRORS, build_or_load_svd_index, resolve_and_download_svd
from sensor_fetcher import fetch_sensor_spec
from svd_extractor import build_svd_catalog
from sensor_extractor import build_sensor_catalog, retrieve_sensor_context
from Task_decomposer import plan_firmware_tasks
from plan_validator import validate_firmware_plan
from code_generator import generate_firmware
from toolchain_resolver import resolve_toolchain

BASE_DIR = Path(__file__).resolve().parent
HW_DIR = BASE_DIR / 'hardware_specs'
SENSOR_DIR = BASE_DIR / 'sensor_specs'
SESSION_DIR = BASE_DIR / 'sessions'
OUTPUT_DIR = BASE_DIR / 'output'
for directory in (HW_DIR, SENSOR_DIR, SESSION_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Silicon-Compiler AI')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/')
def serve_frontend():
    """Serve the frontend from the project's ui/index.html."""
    return FileResponse(BASE_DIR / 'ui' / 'index.html')

if (BASE_DIR / 'ui').is_dir():
    app.mount('/ui', StaticFiles(directory=str(BASE_DIR / 'ui'), html=True), name='ui')


class SetupRequest(BaseModel):
    board: str = Field(min_length=2)
    sensors: list[str] = Field(default_factory=list)
    # Verified board pin map, never generated from an SVD alone.
    pin_map: dict[str, Any] | None = None
    compiler: str | None = None
    compiler_flags: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    instruction: str = Field(min_length=1)
    session_id: str | None = None


class CompileRequest(BaseModel):
    code: str
    session_id: str | None = None
    mcu: str | None = None  # Legacy compiler CPU option, not an MCU part number


def _sse(event: dict) -> str:
    return 'data: ' + json.dumps(event, ensure_ascii=False, default=str) + '\n\n'


def _session_file(session_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{32}', session_id):
        raise ValueError('Invalid session ID')
    return SESSION_DIR / f'{session_id}.json'


def _load_session(session_id: str) -> dict:
    path = _session_file(session_id)
    if not path.is_file():
        raise FileNotFoundError('Unknown hardware session; run /setup first')
    return json.loads(path.read_text(encoding='utf-8'))


def _latest_session() -> str | None:
    # Backward-compatible single-user UI fallback. Multi-user clients must send session_id.
    paths = sorted(SESSION_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0].stem if paths else None


def _target_compiler(session: dict) -> tuple[str, list[str]]:
    target = resolve_toolchain(session['mcu'], session.get('cpu'),
                               compiler=session.get('compiler'),
                               compiler_flags=session.get('compiler_flags'))
    if not target.available:
        raise ValueError(f"Compiler {target.compiler!r} is not installed or not on PATH "
                         f"for {target.architecture} target {session['mcu']}")
    return target.compiler, list(target.flags)


def _parse_gcc_errors(stderr: str) -> list[dict]:
    pattern = re.compile(r'^.*?:(\d+)(?::(\d+))?:\s*(fatal error|error|warning):\s*(.*)$', re.M)
    return [{'line': int(m.group(1)), 'col': int(m.group(2) or 0),
             'severity': m.group(3), 'msg': m.group(4)} for m in pattern.finditer(stderr)]


@app.get('/boards/search')
def boards_search(q: str = Query(min_length=1)):
    index = build_or_load_svd_index()
    needle = re.sub(r'[^A-Z0-9]', '', q.upper())
    hits = [key for key in index if needle in re.sub(r'[^A-Z0-9]', '', key)]
    hits.extend(key for key in EXTERNAL_SVD_MIRRORS if needle in key and key not in hits)
    return {'results': sorted(hits, key=lambda n: (not n.startswith(needle), n))[:20]}


@app.post('/setup')
def setup(req: SetupRequest):
    def stream() -> Iterator[str]:
        sid = uuid.uuid4().hex
        yield _sse({'type': 'step', 'msg': f'Resolving exact MCU SVD for {req.board}...'})
        try:
            svd_path = resolve_and_download_svd(req.board, output_dir=HW_DIR)
            if svd_path is None:
                raise ValueError(f'Unable to resolve unambiguous SVD for {req.board}')
            catalog_path = HW_DIR / f'{svd_path.stem}_catalog.json'
            catalog = build_svd_catalog(svd_path, output_path=catalog_path)
            yield _sse({'type': 'step', 'msg': f'Indexed {len(catalog["peripherals"])} complete MCU peripherals; no register truncation'})
            sensors = list(dict.fromkeys(s.strip().upper() for s in req.sensors if s.strip()))
            sensor_status = {}
            for part in sensors:
                yield _sse({'type': 'step', 'msg': f'Fetching full documentation for {part}...'})
                paths = fetch_sensor_spec(part, output_dir=SENSOR_DIR)
                if not paths:
                    raise FileNotFoundError(f'No sensor documentation for {part}; cannot safely infer it')
                sc = build_sensor_catalog(part, SENSOR_DIR,
                                          output_path=SENSOR_DIR / f'{part}_catalog.json')
                sensor_status[part] = {'sources': len(sc['sources'])}
                yield _sse({'type': 'step', 'msg': f'Indexed {part}: {len(sc["sources"])} source(s)'})
            # Pin mapping is optional at setup; plans needing pin claims fail closed without it.
            session = {'session_id': sid, 'board_query': req.board,
                       'mcu': catalog['device'], 'cpu': catalog.get('cpu'), 'svd_path': str(svd_path.resolve()),
                       'catalog_path': str(catalog_path.resolve()), 'sensors': sensors,
                       'sensor_status': sensor_status, 'pin_map': req.pin_map,
                       'compiler': req.compiler, 'compiler_flags': req.compiler_flags}
            target = resolve_toolchain(session['mcu'], session.get('cpu'),
                                       compiler=req.compiler, compiler_flags=req.compiler_flags)
            session['toolchain'] = target.as_dict()
            _session_file(sid).write_text(json.dumps(session, indent=2), encoding='utf-8')
            yield _sse({'type': 'done', 'msg': 'Complete hardware catalogs ready',
                        'session_id': sid, 'mcu': session['mcu'], 'sensors': sensors,
                        'toolchain': target.as_dict(),
                        'warnings': ([] if req.pin_map else ['No verified pin map supplied']) +
                                    ([] if target.available else [f'Compiler {target.compiler} not found on PATH'])})
        except Exception as exc:
            yield _sse({'type': 'error', 'msg': f'Setup failed: {exc}'})
    return StreamingResponse(stream(), media_type='text/event-stream')


@app.post('/generate')
def generate(req: GenerateRequest):
    def worker(emit) -> None:
        try:
            sid = req.session_id or _latest_session()
            if not sid:
                raise ValueError('No hardware setup session; run /setup first')
            session = _load_session(sid)
            svd_path = Path(session['svd_path'])
            if not svd_path.is_file():
                raise FileNotFoundError('Session SVD is missing; run /setup again')
            compiler, flags = _target_compiler(session)
            emit({'type': 'step', 'msg': 'Running two-pass Qwen task planner...'})
            plan = plan_firmware_tasks(req.instruction, mcu=svd_path.stem,
                                       sensors=session['sensors'], base_dir=BASE_DIR, progress=emit)
            if plan.get('svd', '').upper() != svd_path.name.upper():
                raise ValueError('Planner selected a different SVD than the hardware setup session')
            catalog = build_svd_catalog(svd_path)
            report = validate_firmware_plan(plan, catalog=catalog, pin_map=session.get('pin_map'))
            emit({'type': 'plan', 'tasks': plan['tasks'], 'mcu': plan.get('mcu'),
                        'svd': plan.get('svd'), 'session_id': sid})
            emit({'type': 'plan_validation', **report.as_dict()})
            if not report.valid:
                emit({'type': 'error', 'msg': 'Hardware plan rejected; no C generation attempted',
                            'diagnostics': report.as_dict()})
                return
            # Retrieve sensor passages by task relevance, not fixed source truncation.
            sensor_context_parts = []
            query = req.instruction + ' ' + ' '.join(plan.get('operations', []))
            for sensor in session['sensors']:
                sensor_context_parts.append(f'=== {sensor} ===\n' +
                    retrieve_sensor_context(sensor, query, SENSOR_DIR, max_pages=4))
            emit({'type': 'step', 'msg': 'Plan accepted. Generating validated C functions...'})
            for event in generate_firmware(plan, sensor_context='\n\n'.join(sensor_context_parts),
                                           svd_path=svd_path, pin_map=session.get('pin_map'),
                                           compiler=compiler, compiler_flags=flags, output_dir=OUTPUT_DIR / sid):
                emit(event)
        except Exception as exc:
            emit({'type': 'error', 'msg': f'Generation failed: {exc}'})
    def stream() -> Iterator[str]:
        messages = queue.Queue()
        finished = object()
        def run():
            try:
                worker(messages.put)
            finally:
                messages.put(finished)
        threading.Thread(target=run, daemon=True).start()
        while True:
            try:
                item = messages.get(timeout=5)
            except queue.Empty:
                yield _sse({'type': 'heartbeat', 'msg': 'Backend active; waiting for model output'})
                continue
            if item is finished:
                break
            yield _sse(item)
    return StreamingResponse(stream(), media_type='text/event-stream')


@app.post('/compile')
def compile_code(req: CompileRequest):
    """Syntax/type check only; not a linked firmware build or hardware verification."""
    try:
        sid = req.session_id or _latest_session()
        if not sid:
            raise ValueError('No setup session available')
        session = _load_session(sid)
        compiler, flags = _target_compiler(session)
        # Do not silently apply an ARM target to non-ARM MCUs.
        if req.mcu and not flags:
            raise ValueError('Legacy mcu compiler flag is not accepted without target configuration')
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'firmware.c'
            src.write_text(req.code, encoding='utf-8')
            result = subprocess.run([compiler, *flags, '-Wall', '-Wextra', '-fsyntax-only', str(src)],
                                    capture_output=True, text=True, timeout=30)
        return {'ok': result.returncode == 0, 'errors': _parse_gcc_errors(result.stderr),
                'stderr': result.stderr}
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'errors': [], 'stderr': str(exc)}
