"""Contract-driven, SVD-grounded firmware generation for Silicon-Compiler AI.

Public API: generate_firmware(plan, hw_context='', sensor_context='', *, svd_path=None,
                             pin_map=None, validated=False, ...).
The legacy text arguments are accepted for callers, but hardware definitions MUST
come from the complete structured SVD catalog, not from truncated prose.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator
import requests
from svd_extractor import build_svd_catalog
from plan_validator import validate_firmware_plan

OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://localhost:11434')
CODER_MODEL = os.getenv('CODER_MODEL', 'qwen2.5-coder:3b')
OUTPUT_DIR = Path('./output')
MAX_RETRIES = 2


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value) is not None


def _function_name(signature):
    m = re.fullmatch(r'\s*(?:[\w\s*]+?)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*', signature or '')
    if not m:
        raise ValueError(f'Invalid function signature: {signature!r}')
    return m.group(1)


def _flatten_registers(catalog):
    for pname, p in catalog['peripherals'].items():
        for reg in p['registers']:
            yield pname.upper(), p, reg


def _required_registers(plan, catalog):
    """Only exact, validated register claims enter generated headers."""
    available = {}
    for pname, p, reg in _flatten_registers(catalog):
        available[(pname, reg['name'].upper())] = (p, reg)
    needed = {}
    for task in plan['tasks']:
        for ref in task.get('register_refs', []):
            parts = ref.upper().split('.')
            if len(parts) not in (2, 3):
                raise ValueError(f'Invalid register reference {ref!r}')
            key = (parts[0], parts[1])
            if key not in available:
                raise ValueError(f'Register {ref} absent from SVD')
            if len(parts) == 3 and parts[2] not in {f['name'].upper() for f in available[key][1]['fields']}:
                raise ValueError(f'Field {ref} absent from SVD')
            needed[key] = available[key]
    if not needed:
        raise ValueError('No verified register_refs supplied; refusing to infer hardware')
    return needed


def _header_from_catalog(plan, catalog):
    needed = _required_registers(plan, catalog)
    lines = ['#ifndef SILICON_COMPILER_HW_H', '#define SILICON_COMPILER_HW_H',
             '#include <stdint.h>', '/* Exact register facts from selected MCU SVD. */']
    seen_peripherals = set()
    for (pname, rname), (periph, reg) in sorted(needed.items()):
        if not (_identifier(pname) and _identifier(rname)):
            raise ValueError(f'Unsupported register identifier: {pname}.{rname}')
        base = periph['baseAddress']
        offset = reg['offset']
        size = reg.get('size')
        if base is None or offset is None:
            raise ValueError(f'Missing address: {pname}.{rname}')
        if size not in (8, 16, 32, 64):
            raise ValueError(f'Unsupported or unspecified register width {size}: {pname}.{rname}')
        if pname not in seen_peripherals:
            lines.append(f'#define {pname}_BASE UINT32_C(0x{base:08X})')
            seen_peripherals.add(pname)
        prefix = f'{pname}_{rname}'
        lines.extend([f'#define {prefix}_OFFSET UINT32_C(0x{offset:X})',
                      f'#define {prefix}_ADDR ({pname}_BASE + {prefix}_OFFSET)',
                      f'#define {prefix} ((volatile uint{size}_t *)(uintptr_t){prefix}_ADDR)'])
        for field in reg['fields']:
            fname = field.get('name')
            pos, width = field.get('offset'), field.get('width')
            if not _identifier(fname) or pos is None or width is None or width < 1 or pos < 0 or pos + width > size:
                continue
            mask = ((1 << width) - 1) << pos
            macro = f'{prefix}_{fname}'
            lines.extend([f'#define {macro}_POS {pos}U',
                          f'#define {macro}_MASK UINT{size}_C(0x{mask:X})'])
    lines.append('#endif /* SILICON_COMPILER_HW_H */')
    return '\n'.join(lines) + '\n', needed


def _task_hardware(task, needed):
    lines = []
    for ref in task['register_refs']:
        pname, rname, *_ = ref.upper().split('.')
        p, r = needed[(pname, rname)]
        lines.append(f'{pname}.{rname}: address=0x{r["address"]:X}; size={r.get("size")}; '
                     f'access={r.get("access") or "unspecified"}; description={r.get("desc", "")}')
        for field in r['fields']:
            lines.append(f'  {field["name"]}: offset={field.get("offset")}, width={field.get("width")}, '
                         f'access={field.get("access") or "unspecified"}; {field.get("desc", "")}')
    return '\n'.join(lines)


def _prompt(task, plan, needed, sensor_context, previous_signatures, error_feedback=''):
    contract = {key: task.get(key) for key in ('id', 'signature', 'goal', 'requirements', 'preconditions',
                'postconditions', 'acceptance_criteria', 'depends_on', 'register_refs', 'pin_claims', 'calls')}
    return f'''You are a register-level embedded C implementer. Target: {plan.get('mcu', 'unspecified')}.
Implement EXACTLY ONE function defined by the contract below. No other function definitions.
Do not invent registers, pins, bit positions, commands, or timing. If evidence is insufficient,
reply exactly: INSUFFICIENT_HARDWARE_EVIDENCE. No HAL or CMSIS struct access.
Use only pointer macros declared in hw_defines.h; do not emit includes or #defines.
Use named _MASK and _POS macros for fields. Never use a register not explicitly allowed
by this task's register_refs. Never change any assigned function signature.
Return ONLY C source for that one function, without markdown fences.

CONTRACT:\n{json.dumps(contract, indent=2)}

VERIFIED REGISTER RECORDS:\n{_task_hardware(task, needed)}

SENSOR CONTEXT (evidence, not necessarily authoritative):\n{sensor_context}

EXISTING FUNCTION SIGNATURES (call only, do not redefine):\n{chr(10).join(previous_signatures)}

PREVIOUS VALIDATION ERRORS (fix these, preserve contract):\n{error_feedback}
'''


def _stream_ollama(prompt):
    payload = {'model': CODER_MODEL, 'prompt': prompt, 'stream': True,
               'options': {'temperature': 0.1, 'num_ctx': 8192, 'num_predict': 2048}}
    try:
        with requests.post(f'{OLLAMA_BASE}/api/generate', json=payload, stream=True, timeout=(15, None)) as resp:
            resp.raise_for_status()
            completed = False
            for line in resp.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                if item.get('response'):
                    yield item['response']
                if item.get('done'):
                    completed = True
                    break
            if not completed:
                raise RuntimeError('Ollama stream ended without done=true')
    except requests.RequestException as exc:
        raise RuntimeError(f'Ollama request failed: {exc}') from exc


def _clean(raw):
    code = raw.strip()
    if code.startswith('```'):
        code = re.sub(r'^```(?:c|C)?\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
    return code.strip()


def _remove_comments_and_strings(code):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: ' ' * len(m.group()), code)


def _structural_errors(code, task, all_tasks, header):
    errors = []
    if not code or 'INSUFFICIENT_HARDWARE_EVIDENCE' in code:
        return ['Model returned no implementation or reported insufficient hardware evidence']
    if re.search(r'^\s*#\s*(include|define|undef)', code, re.M):
        errors.append('Includes or macro definitions are forbidden inside a task')
    stripped = _remove_comments_and_strings(code)
    if stripped.count('{') != stripped.count('}'):
        errors.append('Unbalanced braces')
    declared = _function_name(task['signature'])
    definitions = re.findall(r'\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{', stripped)
    # Ignore control constructs; reject extra function definitions.
    definitions = [n for n in definitions if n not in ('if', 'while', 'for', 'switch')]
    if definitions != [declared]:
        errors.append(f'Expected exactly one function definition ({declared}), got {definitions}')
    if not re.search(r'\b' + re.escape(declared) + r'\s*\(', stripped):
        errors.append('Assigned function missing')
    # Check generated register-pointer names, not arbitrary C identifiers.
    allowed = set(re.findall(r'^#define\s+(\w+)\s+\(\(volatile', header, re.M))
    used = set(re.findall(r'\*\s*([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b', stripped))
    for reg in sorted(used - allowed):
        errors.append(f'Unapproved register pointer: {reg}')
    return errors


def _compile_check(source, compiler, timeout=20, compiler_flags=()):
    if not compiler:
        return []
    executable = shutil.which(compiler)
    if not executable:
        return [f'Compiler not found: {compiler}']
    with tempfile.TemporaryDirectory(prefix='silicon_check_') as directory:
        path = Path(directory) / 'firmware.c'
        path.write_text(source, encoding='utf-8')
        try:
            result = subprocess.run([executable, *compiler_flags, '-std=c11', '-fsyntax-only', '-Wall', '-Werror=implicit-function-declaration',
                                     str(path)], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f'Compiler execution failed: {exc}']
        return [] if result.returncode == 0 else [result.stderr[-6000:] or 'Compiler failed']


def _assemble(plan, header, implementations):
    prototypes = '\n'.join(task['signature'].rstrip(';') + ';' for task in plan['tasks'])
    functions = '\n\n'.join(implementations[task['id']] for task in plan['tasks'])
    return f'/* Silicon-Compiler AI: {plan.get("mcu", "MCU")} */\n{header}\n{prototypes}\n\n{functions}\n'


def generate_firmware(plan: dict, hw_context: str = '', sensor_context: str = '', *,
                      svd_path: str | Path | None = None, pin_map: dict | None = None,
                      catalog: dict | None = None, compiler: str | None = 'arm-none-eabi-gcc',
                      compiler_flags: list[str] | tuple[str, ...] = (),
                      max_retries: int = MAX_RETRIES, output_dir: Path = OUTPUT_DIR) -> Generator[dict, None, None]:
    """Fail closed unless the full plan passes deterministic validation.

    hw_context is legacy input, intentionally NOT parsed into hardware definitions.
    Pass svd_path or catalog from the selected MCU, and verified pin_map when needed.
    """
    try:
        if catalog is None:
            if svd_path is None:
                raise ValueError('Pass svd_path or catalog; legacy hw_context text is not authoritative')
            catalog = build_svd_catalog(Path(svd_path))
        report = validate_firmware_plan(plan, catalog=catalog, pin_map=pin_map)
        if not report.valid:
            yield {'type': 'error', 'msg': 'Plan validation failed', 'diagnostics': report.as_dict()}
            return
        header, needed = _header_from_catalog(plan, catalog)
        if not compiler or not shutil.which(compiler):
            raise ValueError(f'Target compiler unavailable: {compiler!r}. Configure an installed compiler explicitly.')
        tasks = plan['tasks']
        output_dir = Path(output_dir)
        implementations = {}
        signatures = []
        for task in tasks:
            tid, sig = task['id'], task['signature']
            yield {'type': 'task_start', 'id': tid, 'sig': sig}
            errors = []
            for attempt in range(max_retries + 1):
                prompt = _prompt(task, plan, needed, sensor_context, signatures, '\n'.join(errors))
                raw = ''
                try:
                    for token in _stream_ollama(prompt):
                        raw += token
                        yield {'type': 'token', 'id': tid, 'text': token, 'attempt': attempt + 1}
                except (RuntimeError, ValueError) as exc:
                    yield {'type': 'error', 'msg': f'Task {tid}: {exc}'}
                    return
                code = _clean(raw)
                errors = _structural_errors(code, task, tasks, header)
                if not errors:
                    trial = dict(implementations)
                    trial[tid] = code
                    # Compile a self-contained partial translation unit with prototypes.
                    partial = _assemble({'mcu': plan.get('mcu'), 'tasks': tasks[:len(trial)]}, header, trial)
                    errors = _compile_check(partial, compiler, compiler_flags=compiler_flags)
                if not errors:
                    implementations[tid] = code
                    signatures.append(sig)
                    yield {'type': 'task_done', 'id': tid}
                    break
                yield {'type': 'validation_failed', 'id': tid, 'attempt': attempt + 1, 'errors': errors}
            else:
                yield {'type': 'error', 'msg': f'Task {tid} failed after {max_retries + 1} attempts', 'diagnostics': errors}
                return
        merged = _assemble(plan, header, implementations)
        errors = _compile_check(merged, compiler, compiler_flags=compiler_flags)
        if errors:
            yield {'type': 'error', 'msg': 'Final compilation failed', 'diagnostics': errors}
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'hw_defines.h').write_text(header, encoding='utf-8')
        (output_dir / 'firmware.c').write_text(merged, encoding='utf-8')
        yield {'type': 'merge_done', 'code': merged}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        yield {'type': 'error', 'msg': str(exc)}
