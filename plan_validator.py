"""Deterministic, evidence-based validation of Silicon-Compiler firmware plans.

No LLM calls. A valid SVD proves register layout, NOT pinmux or protocol timing.
Unknown critical facts block code generation unless verified board/pin evidence is
supplied. Use validate_firmware_plan(...), or require_valid_plan(...).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from svd_extractor import build_svd_catalog

IDENT = re.compile(r'^[A-Za-z_]\w*$')
SIGNATURE = re.compile(r'^\s*(?P<ret>(?:[A-Za-z_]\w*\s+|\*\s*)+)(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^{};]*)\)\s*$')
REGREF = re.compile(r'^([A-Za-z_]\w*)\.([A-Za-z_][\w.\[\]%]*)' r'(?:\.([A-Za-z_]\w*))?$')
PIN = re.compile(r'^P([A-Z])(\d{1,2})$', re.I)
GPIO_AF = re.compile(r'^AF(\d{1,2})$', re.I)
RESERVED = {'main', 'printf', 'malloc', 'free'}

@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    task_id: int | None = None
    evidence: str | None = None

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if v is not None}

@dataclass
class ValidationReport:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    verified_registers: list[dict] = field(default_factory=list)
    verified_pins: list[dict] = field(default_factory=list)
    function_owners: dict[str, int] = field(default_factory=dict)
    dependency_order: list[int] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(d.severity == 'error' for d in self.diagnostics)

    def add(self, code, message, task_id=None, severity='error', evidence=None):
        self.diagnostics.append(Diagnostic(code, severity, message, task_id, evidence))

    def as_dict(self):
        return {'valid': self.valid, 'diagnostics': [d.as_dict() for d in self.diagnostics],
                'verified_registers': self.verified_registers, 'verified_pins': self.verified_pins,
                'function_owners': self.function_owners, 'dependency_order': self.dependency_order}


class FirmwarePlanRejected(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__('Firmware plan rejected: ' + '; '.join(d.message for d in report.diagnostics if d.severity == 'error'))


def _register_index(catalog: dict) -> dict:
    result = {}
    for peripheral, details in catalog.get('peripherals', {}).items():
        regs = {}
        for reg in details.get('registers', []):
            if reg.get('name'):
                regs[reg['name'].upper()] = reg
        result[peripheral.upper()] = regs
    return result


def _verify_register(ref: str, registers: dict) -> tuple[bool, str, dict | None]:
    # Exact peripheral.register[.field] references only; no fuzzy matches.
    bits = ref.split('.')
    if len(bits) not in (2, 3):
        return False, 'Expected PERIPHERAL.REGISTER[.FIELD]', None
    p, r = bits[0].upper(), bits[1].upper()
    if p not in registers:
        return False, f'Peripheral {p} does not exist in SVD', None
    register = registers[p].get(r)
    if register is None:
        return False, f'Register {p}.{r} does not exist in SVD', None
    if register.get('address') is None:
        return False, f'Register {p}.{r} has no resolved address', None
    if len(bits) == 3:
        fields = {str(f.get('name', '')).upper(): f for f in register.get('fields', [])}
        field = fields.get(bits[2].upper())
        if field is None:
            return False, f'Field {ref} does not exist in SVD', None
        if field.get('offset') is None or field.get('width') is None:
            return False, f'Field {ref} lacks a resolved bit position', None
    return True, '', {'reference': ref, 'address': register['address'],
                       'source': 'CMSIS-SVD', 'access': register.get('access')}


def _pins_database(pin_map: dict | None) -> dict:
    """Pin map is explicit evidence, not inferred from SVD.

    Expected shape: {"PB8": {"peripheral": "I2C1", "signal": "SCL",
    "af": 4, "source": "STM32F411 datasheet table ..."}}.
    """
    return {str(k).upper(): v for k, v in (pin_map or {}).items()}


def validate_firmware_plan(plan: dict, *, svd_path: str | Path | None = None,
                           catalog: dict | None = None, pin_map: dict | None = None,
                           require_pin_evidence: bool = True,
                           require_resolved_requirements: bool = True,
                           require_claim_coverage: bool = True) -> ValidationReport:
    """Validate a planner result without making assumptions or modifying the plan.

    Structured optional task fields:
      register_refs: ["I2C1.CR1.PE", "RCC.APB1ENR"]
      pin_claims: [{"pin":"PB8","peripheral":"I2C1","signal":"SCL","af":4}]
      calls: ["other_function_name"]
    For pin-dependent plans, provide independently verified pin_map. Unspecified
    claims are not fabricated or certified. A successful result does not prove
    firmware execution or sensor protocol correctness.
    """
    report = ValidationReport()
    if not isinstance(plan, dict):
        report.add('INVALID_PLAN', 'Plan must be a JSON object')
        return report
    if catalog is None:
        if svd_path is None:
            report.add('MISSING_SVD', 'An SVD catalog or explicit SVD path is required')
            return report
        try:
            catalog = build_svd_catalog(Path(svd_path))
        except (OSError, ValueError, KeyError) as exc:
            report.add('INVALID_SVD', f'Unable to load SVD: {exc}')
            return report
    registers = _register_index(catalog)
    selected = plan.get('selected_peripherals', [])
    if not isinstance(selected, list) or any(not isinstance(p, str) for p in selected):
        report.add('INVALID_PERIPHERALS', 'selected_peripherals must be a list of names')
        selected = []
    selected_upper = [p.upper() for p in selected]
    if len(set(selected_upper)) != len(selected_upper):
        report.add('DUPLICATE_PERIPHERALS', 'selected_peripherals contains duplicates')
    for p in selected_upper:
        if p not in registers:
            report.add('UNKNOWN_PERIPHERAL', f'Peripheral {p} not found in SVD')

    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not (1 <= len(tasks) <= 16):
        report.add('INVALID_TASKS', 'tasks must contain 1–16 task objects')
        return report
    ids, owners, deps_by_id = set(), {}, {}
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            report.add('INVALID_TASK', f'Task {index} must be an object')
            continue
        tid = task.get('id')
        if type(tid) is not int or tid != index:
            report.add('TASK_ID', f'Task at position {index} must have id {index}')
            tid = index
        if tid in ids:
            report.add('DUPLICATE_TASK_ID', f'Duplicate task id {tid}', tid)
        ids.add(tid)
        sig = task.get('signature', '')
        match = SIGNATURE.fullmatch(sig) if isinstance(sig, str) else None
        if not match:
            report.add('INVALID_SIGNATURE', f'Invalid function signature: {sig!r}', tid)
        else:
            name = match.group('name')
            if name in owners or name in RESERVED:
                report.add('FUNCTION_OWNERSHIP', f'Function {name} has duplicate/reserved ownership', tid)
            else:
                owners[name] = tid
                report.function_owners[name] = tid
        dependencies = task.get('depends_on', [])
        if not isinstance(dependencies, list) or any(type(d) is not int for d in dependencies):
            report.add('INVALID_DEPENDENCIES', 'depends_on must be integer task IDs', tid)
            dependencies = []
        if len(dependencies) != len(set(dependencies)):
            report.add('DUPLICATE_DEPENDENCY', 'Duplicate depends_on entry', tid)
        if any(d >= tid or d < 1 for d in dependencies):
            report.add('DEPENDENCY_ORDER', 'Dependencies must reference earlier tasks', tid)
        deps_by_id[tid] = dependencies
        periphs = task.get('peripherals', [])
        if not isinstance(periphs, list) or any(not isinstance(p, str) for p in periphs):
            report.add('INVALID_TASK_PERIPHERALS', 'Task peripherals must be strings', tid)
            periphs = []
        for p in periphs:
            if p.upper() not in selected_upper:
                report.add('UNSELECTED_PERIPHERAL', f'Task references unselected peripheral {p}', tid)
        refs = task.get('register_refs', [])
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            report.add('INVALID_REGISTER_REFS', 'register_refs must be a list of strings', tid)
            refs = []
        for ref in refs:
            valid, error, record = _verify_register(ref, registers)
            if not valid:
                report.add('INVALID_REGISTER', f'{ref}: {error}', tid)
            else:
                if ref.split('.')[0].upper() not in [p.upper() for p in periphs]:
                    report.add('UNDECLARED_REGISTER_PERIPHERAL', f'{ref} is not listed in task peripherals', tid)
                report.verified_registers.append({'task_id': tid, **record})
        claims = task.get('pin_claims', [])
        if not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims):
            report.add('INVALID_PIN_CLAIMS', 'pin_claims must be a list of objects', tid)
            claims = []
        pin_db = _pins_database(pin_map)
        for claim in claims:
            pin = str(claim.get('pin', '')).upper()
            expected = pin_db.get(pin)
            if not PIN.fullmatch(pin):
                report.add('INVALID_PIN', f'Invalid pin identifier {pin!r}', tid)
            elif expected is None:
                report.add('UNVERIFIED_PIN', f'No verified pin mapping for {pin}', tid,
                           'error' if require_pin_evidence else 'warning')
            elif not expected.get('source'):
                report.add('MISSING_PIN_SOURCE', f'Pin mapping for {pin} has no provenance', tid)
            else:
                matches = all(str(claim.get(k, '')).upper() == str(expected.get(k, '')).upper()
                              for k in ('peripheral', 'signal', 'af'))
                if not matches:
                    report.add('PINMUX_CONFLICT', f'{pin} does not match verified peripheral/signal/AF', tid,
                               evidence=str(expected.get('source')))
                elif str(claim.get('peripheral', '')).upper() not in [p.upper() for p in periphs]:
                    report.add('PIN_PERIPHERAL_UNDECLARED', f'{pin} peripheral not declared in task', tid)
                else:
                    report.verified_pins.append({'task_id': tid, **claim, 'source': expected['source']})
        calls = task.get('calls', [])
        if not isinstance(calls, list) or any(not isinstance(c, str) for c in calls):
            report.add('INVALID_CALLS', 'calls must be a list of function names', tid)
    for task in tasks:
        if not isinstance(task, dict) or type(task.get('id')) is not int:
            continue
        tid = task['id']
        for dep in deps_by_id.get(tid, []):
            if dep not in ids:
                report.add('MISSING_DEPENDENCY', f'Task {tid} depends on missing task {dep}', tid)
        for call in task.get('calls', []):
            if not isinstance(call, str):
                continue
            if call not in owners:
                report.add('UNKNOWN_FUNCTION_CALL', f'Function {call} has no task owner', tid)
            elif owners[call] not in deps_by_id.get(tid, []):
                report.add('UNDECLARED_CALL_DEPENDENCY', f'Call to {call} requires depends_on task {owners[call]}', tid)
    report.dependency_order = [task.get('id') for task in tasks if isinstance(task, dict) and type(task.get('id')) is int]
    unknowns = plan.get('unverified_requirements', [])
    if not isinstance(unknowns, list):
        report.add('INVALID_UNVERIFIED', 'unverified_requirements must be a list')
    elif unknowns:
        report.add('UNRESOLVED_HARDWARE', 'Plan has unresolved hardware requirements: ' + '; '.join(map(str, unknowns)),
                   severity='error' if require_resolved_requirements else 'warning')
    if not any(isinstance(t, dict) and t.get('register_refs') for t in tasks):
        report.add('REGISTER_COVERAGE_UNKNOWN', 'No register references were claimed; register selection cannot be certified', severity='error' if require_claim_coverage else 'warning')
    if not any(isinstance(t, dict) and t.get('pin_claims') for t in tasks):
        report.add('PIN_COVERAGE_UNKNOWN', 'No verified pin claims were supplied; pinmux correctness cannot be certified', severity='error' if require_claim_coverage else 'warning')
    return report


def require_valid_plan(plan: dict, **kwargs) -> ValidationReport:
    report = validate_firmware_plan(plan, **kwargs)
    if not report.valid:
        raise FirmwarePlanRejected(report)
    return report


def save_validation_report(report: ValidationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding='utf-8')
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate a firmware task plan against a complete CMSIS-SVD catalog')
    parser.add_argument('plan', type=Path)
    parser.add_argument('--svd', required=True, type=Path)
    parser.add_argument('--pins', type=Path, help='Optional verified pin map JSON')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = validate_firmware_plan(json.loads(args.plan.read_text(encoding='utf-8')),
                                    svd_path=args.svd,
                                    pin_map=json.loads(args.pins.read_text(encoding='utf-8')) if args.pins else None)
    if args.report:
        save_validation_report(result, args.report)
    print(json.dumps(result.as_dict(), indent=2))
    raise SystemExit(0 if result.valid else 1)
