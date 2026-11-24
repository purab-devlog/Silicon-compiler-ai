"""Two-pass, SVD-grounded firmware task planner for Silicon-Compiler AI.

This module plans firmware; it does NOT certify pinmux, electrical timing, or
sensor protocol correctness. Those require additional authoritative sources.

Backward compatible entry point: plan_firmware_tasks(instruction, ...).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
import time
import re
from pathlib import Path
from typing import Any

import requests

from svd_extractor import build_svd_catalog
from sensor_extractor import build_sensor_catalog, retrieve_sensor_context

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
PLANNER_MODEL = "qwen2.5:3b"
MCU_RE = re.compile(r"\b(?:STM32[A-Z0-9]{4,}|RP2040|RP2350|ESP32[-A-Z0-9]*|NRF52[A-Z0-9]*|ATMEGA[A-Z0-9]+|LPC[A-Z0-9]+|EFM32[A-Z0-9]+)\b", re.I)
SIGNATURE_RE = re.compile(r"^\s*(?:[A-Za-z_]\w*\s+|\*\s*)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$")
SENSOR_SUFFIXES = ("_catalog.json", "_datasheet.pdf", "_regs.h", "_reg.h", "_extracted.txt")


class PlanValidationError(ValueError):
    pass


def _base_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "hardware_specs").exists() or (here / "sensor_specs").exists():
        return here
    return Path.cwd()


def _match_svd(mcu: str | None, directory: Path) -> Path | None:
    files = list(directory.glob("*.svd")) if directory.exists() else []
    if not mcu:
        return files[0] if len(files) == 1 else None
    norm = re.sub(r"[^A-Z0-9]", "", mcu.upper())
    matches = [p for p in files if norm.startswith(re.sub(r"[^A-Z0-9]", "", p.stem.upper()))]
    if not matches:
        return None
    longest = max(len(re.sub(r"[^A-Z0-9]", "", p.stem.upper())) for p in matches)
    winners = [p for p in matches if len(re.sub(r"[^A-Z0-9]", "", p.stem.upper())) == longest]
    if len(winners) != 1:
        raise PlanValidationError(f"Ambiguous SVD match for {mcu}: {[p.name for p in winners]}")
    return winners[0]


def _cached_sensors(instruction: str, directory: Path) -> list[str]:
    if not directory.exists():
        return []
    parts = set()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        for suffix in SENSOR_SUFFIXES:
            if path.name.lower().endswith(suffix):
                parts.add(path.name[:-len(suffix)].upper())
                break
    upper = instruction.upper()
    return sorted((p for p in parts if re.search(rf"(?<![A-Z0-9]){re.escape(p)}(?![A-Z0-9])", upper)), key=len, reverse=True)


def _json_object(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        begin, end = text.find("{"), text.rfind("}")
        if begin < 0 or end <= begin:
            raise PlanValidationError("Model response is not JSON")
        try:
            data = json.loads(text[begin:end + 1])
        except json.JSONDecodeError as exc:
            raise PlanValidationError(f"Malformed model JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanValidationError("Expected a JSON object")
    return data


def _call_model(prompt: str, model: str, url: str, timeout: int, num_predict: int = -1, progress=None, *, log_dir=None, run_id=None, stage="planner", attempt=1) -> dict:
    """Stream Ollama output and durably append each fragment before notifying the UI.

    .partial.txt is the exact raw model output (may be incomplete JSON on Ctrl+C).
    .events.jsonl is an append-only audit trail. Files are never overwritten.
    """
    payload = {"model": model, "prompt": prompt, "stream": True, "format": "json",
               "keep_alive": "15m", "options": {"temperature": 0.1, "num_ctx": int(os.getenv("PLANNER_NUM_CTX", "16384")),
                                                "num_predict": int(os.getenv("PLANNER_NUM_PREDICT", str(num_predict)))}}
    directory = Path(log_dir) if log_dir is not None else _base_dir() / "planner_logs"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = run_id or uuid.uuid4().hex
    stem = f"{run_id}_{stage}_attempt{attempt}"
    raw_path = directory / f"{stem}.partial.txt"
    events_path = directory / f"{stem}.events.jsonl"
    chunks, count, done_reason = [], 0, None

    def durable_write(handle, value):
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())

    def record(handle, event_type, **fields):
        durable_write(handle, json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                          "event": event_type, "run_id": run_id,
                                          "stage": stage, "attempt": attempt, **fields},
                                         ensure_ascii=False) + "\n")

    # Open the files BEFORE making the model request. No in-memory-only output.
    with raw_path.open("x", encoding="utf-8", newline="") as raw, events_path.open("x", encoding="utf-8") as events:
        record(events, "started", model=model, raw_file=str(raw_path), prompt_chars=len(prompt))
        if progress:
            progress({"type": "planner_log", "msg": f"Saving model output to {raw_path}",
                      "path": str(raw_path), "events_path": str(events_path)})
        try:
            with requests.post(url, json=payload, timeout=(15, None), stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    delta = event.get("response", "")
                    if delta:
                        # Write and fsync BEFORE emitting to the UI, so Ctrl+C
                        # leaves everything received so far on disk.
                        durable_write(raw, delta)
                        chunks.append(delta)
                        count += len(delta)
                        if progress:
                            progress({"type": "planner_token", "model": model,
                                      "text": delta, "characters": count})
                    if event.get("done"):
                        done_reason = event.get("done_reason") or "stop"
                        break
            record(events, "stream_finished", characters=count, done_reason=done_reason)
        except BaseException as exc:
            record(events, "interrupted", characters=count, error=f"{type(exc).__name__}: {exc}")
            raise
        if done_reason == "length":
            record(events, "validation_failed", error="Output token limit reached")
            raise PlanValidationError(f"Planner reached output token limit after {count} characters; partial output: {raw_path}")
        if done_reason is None:
            record(events, "validation_failed", error="Stream ended without completion marker")
            raise RuntimeError(f"Ollama stream ended without a completion marker; partial output: {raw_path}")
        if progress:
            progress({"type": "planner_stream_done", "characters": count, "path": str(raw_path)})
        try:
            parsed = _json_object("".join(chunks))
        except (PlanValidationError, ValueError) as exc:
            record(events, "validation_failed", error=str(exc))
            raise
        record(events, "json_parsed", characters=count)
        return parsed


def _list_strings(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise PlanValidationError(f"{label} must be a list of strings")
    values = list(dict.fromkeys(v.strip() for v in value if v.strip()))
    if not allow_empty and not values:
        raise PlanValidationError(f"{label} must not be empty")
    return values


def _validate_hardware(data: dict, allowed: set[str]) -> dict:
    selected = [v.upper() for v in _list_strings(data.get("selected_peripherals"), "selected_peripherals")]
    if len(selected) != len(set(selected)):
        raise PlanValidationError("Duplicate peripheral selection")
    if allowed:
        unknown = sorted(set(selected) - allowed)
        if unknown:
            raise PlanValidationError(f"Peripherals not in SVD: {unknown}")
    elif selected:
        raise PlanValidationError("Cannot verify peripherals without an SVD; provide the correct MCU SVD")
    operations = _list_strings(data.get("operations"), "operations", allow_empty=False)
    dependencies = _list_strings(data.get("dependencies", []), "dependencies")
    assumptions = _list_strings(data.get("assumptions", []), "assumptions")
    unknowns = _list_strings(data.get("unverified_requirements", []), "unverified_requirements")
    return {"selected_peripherals": selected, "operations": operations,
            "dependencies": dependencies, "assumptions": assumptions,
            "unverified_requirements": unknowns}


def _validate_tasks(data: dict, selected: list[str]) -> list[dict]:
    raw = data.get("tasks")
    if not isinstance(raw, list) or not raw or len(raw) > 16:
        raise PlanValidationError("tasks must contain 1–16 task contracts")
    result, signatures, ids = [], set(), set()
    for index, task in enumerate(raw, 1):
        if not isinstance(task, dict):
            raise PlanValidationError(f"Task {index} must be an object")
        tid = task.get("id")
        if type(tid) is not int or tid != index or tid in ids:
            raise PlanValidationError(f"Task ids must be sequential integers starting at 1 (task {index})")
        ids.add(tid)
        signature = task.get("signature")
        match = SIGNATURE_RE.fullmatch(signature) if isinstance(signature, str) else None
        if not match:
            raise PlanValidationError(f"Task {index} has an invalid C function signature: {signature!r}")
        name = match.group(1)
        if name in signatures or name in {"main", "printf", "malloc", "free"}:
            raise PlanValidationError(f"Duplicate or reserved function owner: {name}")
        signatures.add(name)
        deps = task.get("depends_on", [])
        if not isinstance(deps, list) or any(type(d) is not int or d < 1 or d >= tid for d in deps):
            raise PlanValidationError(f"Task {index} has invalid depends_on; only earlier task ids allowed")
        periphs = [p.upper() for p in _list_strings(task.get("peripherals", []), f"task {index} peripherals")]
        if set(periphs) - set(selected):
            raise PlanValidationError(f"Task {index} references unselected peripherals: {sorted(set(periphs)-set(selected))}")
        goal = task.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError(f"Task {index} requires a goal")
        requirements = _list_strings(task.get("requirements"), f"task {index} requirements", allow_empty=False)
        preconditions = _list_strings(task.get("preconditions", []), f"task {index} preconditions")
        postconditions = _list_strings(task.get("postconditions"), f"task {index} postconditions", allow_empty=False)
        checks = _list_strings(task.get("acceptance_criteria"), f"task {index} acceptance_criteria", allow_empty=False)
        register_refs = _list_strings(task.get("register_refs"), f"task {index} register_refs")
        calls = _list_strings(task.get("calls"), f"task {index} calls")
        pin_claims = task.get("pin_claims")
        if not isinstance(pin_claims, list) or any(not isinstance(c, dict) for c in pin_claims):
            raise PlanValidationError(f"task {index} pin_claims must be a list of objects")
        for claim in pin_claims:
            if not all(k in claim for k in ("pin", "peripheral", "signal", "af")):
                raise PlanValidationError(f"task {index} pin claim requires pin, peripheral, signal, af")
        result.append({"register_refs": register_refs, "pin_claims": pin_claims, "calls": calls,
                       "id": tid, "signature": signature.strip(), "peripherals": periphs,
                       "goal": goal.strip(), "depends_on": list(dict.fromkeys(deps)),
                       "requirements": requirements, "preconditions": preconditions,
                       "postconditions": postconditions, "acceptance_criteria": checks})
    return result


def _repairable_call(prompt: str, validator, model: str, url: str, timeout: int, retries: int, progress=None, *, log_dir=None, run_id=None, stage="planner") -> Any:
    feedback = ""
    for attempt in range(retries + 1):
        try:
            if progress:
                progress({"type": "model_progress", "model": model, "msg": f"Planner attempt {attempt + 1}/{retries + 1}", "attempt": attempt + 1, "total_attempts": retries + 1})
            result = _call_model(prompt + feedback, model, url, timeout, progress=progress,
                                 log_dir=log_dir, run_id=run_id, stage=stage, attempt=attempt + 1)
            return validator(result)
        except (PlanValidationError, ValueError) as exc:
            if attempt == retries:
                raise PlanValidationError(f"Planner failed validation after {attempt+1} attempts: {exc}") from exc
            if progress:
                progress({"type": "validation_failed", "msg": str(exc), "attempt": attempt + 1})
            feedback = ("\n\nPREVIOUS OUTPUT FAILED VALIDATION. Regenerate the ENTIRE JSON object. "
                        f"Fix this specific error: {exc}. No markdown, no commentary.\n")


def plan_firmware_tasks(
    instruction: str,
    model: str = PLANNER_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout_sec: int = 240,
    *,
    mcu: str | None = None,
    sensors: list[str] | None = None,
    base_dir: Path | None = None,
    max_retries: int = 2,
    progress=None,
) -> dict:
    """Plan firmware in two validated Qwen passes.

    Legacy fields (mcu, svd, sensors, selected_peripherals, tasks) remain intact.
    A cached SVD is required for verified peripheral selection. Missing board
    pinmux and sensor protocol facts are reported, never guessed as verified.
    """
    if not instruction or not instruction.strip():
        raise ValueError("Firmware instruction is empty")
    base = Path(base_dir) if base_dir is not None else _base_dir()
    hw_dir, sensor_dir = base / "hardware_specs", base / "sensor_specs"
    target_mcu = mcu or (MCU_RE.search(instruction).group(0).upper() if MCU_RE.search(instruction) else None)
    svd = _match_svd(target_mcu, hw_dir)
    if svd is None:
        raise FileNotFoundError(f"No unambiguous cached SVD for MCU {target_mcu!r}; run hardware setup first")
    catalog = build_svd_catalog(svd)
    allowed = set(catalog["peripherals"])
    selected_sensors = sensors if sensors is not None else _cached_sensors(instruction, sensor_dir)
    selected_sensors = list(dict.fromkeys(s.upper() for s in selected_sensors))

    # A unique run ID groups both planning passes and all retries.
    run_id = uuid.uuid4().hex
    log_dir = base / "planner_logs"
    # Pass 1 sees names and metadata only, not megabytes of registers.
    hardware_prompt = f"""You are a cautious embedded-firmware hardware operation planner.
You are NOT writing C code. Return ONLY a valid JSON object.
Never invent a peripheral name, board pin, register, clock, sensor command or timing value.
Select peripherals ONLY from the authoritative SVD name list below.
Include necessary clock and GPIO peripherals if present; do not assume GPIO pin mappings.
If a required fact is not proven by the supplied sources, put it in unverified_requirements.
A CMSIS-SVD lists registers, NOT necessarily board wiring, alternate-function mappings or sensor protocols.
Instruction: {instruction}
Target MCU: {target_mcu or catalog['device']}
Available SVD peripherals: {json.dumps(sorted(allowed))}
Detected sensor parts: {json.dumps(selected_sensors)}
Return exactly these JSON fields:
{{"selected_peripherals":["I2C1","RCC","GPIOB"],"operations":["enable clock","configure pins","initialize bus","transact with sensor"],"dependencies":["clock before bus"],"assumptions":[],"unverified_requirements":["verify board pin mapping"]}}
The example values above are illustrative ONLY; choose based on the actual target.
"""
    if progress:
        progress({"type": "stage", "stage": "hardware", "msg": "Model 1: selecting peripherals and dependencies"})
    hardware = _repairable_call(hardware_prompt, lambda d: _validate_hardware(d, allowed),
                                model, ollama_url, timeout_sec, max_retries, progress=progress,
                                log_dir=log_dir, run_id=run_id, stage="hardware")

    # Retrieve the complete records of selected peripherals. For a 3B planner,
    # send a concise register/field inventory; full catalog stays on disk/in memory.
    periph_summary = {}
    for name in hardware["selected_peripherals"]:
        p = catalog["peripherals"][name]
        periph_summary[name] = {"registers": [{"name": r["name"],
                                                "fields": [f["name"] for f in r.get("fields", [])]}
                                               for r in p["registers"]]}
    sensor_context = {}
    for part in selected_sensors:
        try:
            build_sensor_catalog(part, sensor_dir)
            context = retrieve_sensor_context(part, instruction + " " + " ".join(hardware["operations"]),
                                              sensor_dir, max_pages=4)
            sensor_context[part] = context[:12000] if context else "No matching textual excerpts; consult full catalog"
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            sensor_context[part] = f"Sensor source unavailable: {exc}"
            hardware["unverified_requirements"].append(f"Verify sensor {part} documentation")

    # Explicitly cap *prompt payload*, never underlying source documents.
    periph_text = json.dumps(periph_summary, separators=(",", ":"))
    if len(periph_text) > 22000:
        # Names remain available; downstream retrieval must supply full records.
        periph_summary = {name: {"register_names": [r["name"] for r in catalog["peripherals"][name]["registers"]]}
                          for name in hardware["selected_peripherals"]}

    task_prompt = f"""You are a senior bare-metal firmware architect planning for a small C coding model.
Return ONLY JSON. Do not write C implementation code.
Design 1–16 ordered, non-overlapping FUNCTION CONTRACTS, each owned by exactly one task.
Each function has one responsibility; no task may redefine an earlier function.
Tasks may call earlier functions, but must declare their ids in depends_on.
Do not place entire application implementation in one helper. Include initialization, transactions, error handling and final measurement/application logic when requested.
Do not invent hardware facts. Unknown board pin mapping, timing, or protocol details must remain explicit preconditions or unresolved requirements, not fabricated constants.
Function signatures must be valid C declarations without a semicolon or body.
Each task requires id (1..N), signature, peripherals, goal, depends_on (earlier ids only), requirements, preconditions, postconditions, acceptance_criteria, register_refs, pin_claims, calls.
register_refs MUST be a list of exact PERIPHERAL.REGISTER or PERIPHERAL.REGISTER.FIELD references from the supplied SVD inventory. Never invent registers.
pin_claims MUST be a list of objects with pin, peripheral, signal, af. Only claim a pin when the provided evidence proves the pin mapping. Otherwise use [] and leave pin mapping unresolved.
calls MUST be function names owned by earlier tasks, and every called function's task id MUST appear in depends_on.
Avoid overlapping initialization functions; each task owns a distinct operation.
Include all three structured fields even if their lists are empty.
Acceptance criteria must be specific and testable where possible.
Instruction: {instruction}
Target: {target_mcu or catalog['device']}
Verified peripheral names: {json.dumps(hardware['selected_peripherals'])}
Hardware operations: {json.dumps(hardware['operations'])}
Dependencies: {json.dumps(hardware['dependencies'])}
Unverified requirements: {json.dumps(hardware['unverified_requirements'])}
Register and field inventory (SVD only): {json.dumps(periph_summary, separators=(',', ':'))}
Sensor excerpts (unverified until checked against manufacturer source): {json.dumps(sensor_context)}
Return this exact shape (fill actual values):
{{"tasks":[{{"id":1,"signature":"void example_init(void)","peripherals":[],"goal":"...","depends_on":[],"requirements":["..."],"preconditions":[],"postconditions":["..."],"acceptance_criteria":["..."],"register_refs":[],"pin_claims":[],"calls":[]}}]}}
"""
    if progress:
        progress({"type": "stage", "stage": "contracts", "msg": "Model 1: writing function contracts"})
    tasks = _repairable_call(task_prompt, lambda d: _validate_tasks(d, hardware["selected_peripherals"]),
                             model, ollama_url, timeout_sec, max_retries, progress=progress,
                             log_dir=log_dir, run_id=run_id, stage="contracts")
    return {"schema_version": 2, "mcu": target_mcu or catalog["device"], "svd": svd.name,
            "sensors": selected_sensors, "selected_peripherals": hardware["selected_peripherals"],
            "operations": hardware["operations"], "hardware_dependencies": hardware["dependencies"],
            "assumptions": hardware["assumptions"],
            "unverified_requirements": list(dict.fromkeys(hardware["unverified_requirements"])),
            "tasks": tasks}


if __name__ == "__main__":
    instruction = input("Firmware instruction: ").strip()
    print(json.dumps(plan_firmware_tasks(instruction), indent=2))
