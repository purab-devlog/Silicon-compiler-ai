"""Conservative toolchain resolution from MCU CPU metadata and explicit target identity.

This selects a GCC *syntax-check* toolchain; it does not configure SDKs, linkers,
startup code or ESP-IDF. Unknown architectures fail closed.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Toolchain:
    architecture: str
    compiler: str
    flags: tuple[str, ...]
    available: bool
    cpu: str

    def as_dict(self) -> dict[str, Any]:
        return {'architecture': self.architecture, 'compiler': self.compiler,
                'flags': list(self.flags), 'available': self.available, 'cpu': self.cpu}


def resolve_toolchain(mcu: str, cpu: str | None = None, *,
                      compiler: str | None = None,
                      compiler_flags: list[str] | None = None) -> Toolchain:
    """Resolve ARM, ESP Xtensa or ESP RISC-V compiler without assuming ARM.

    Explicit compiler overrides are supported, but an unknown architecture still
    requires explicit flags to prevent accidentally compiling for the host.
    """
    name = re.sub(r'[^A-Z0-9]', '', (mcu or '').upper())
    cpu_name = (cpu or '').strip()
    normalized_cpu = re.sub(r'[^a-z0-9]', '', cpu_name.lower())
    arm = {'cm0': 'cortex-m0', 'cm0plus': 'cortex-m0plus',
           'cm3': 'cortex-m3', 'cm4': 'cortex-m4', 'cm7': 'cortex-m7',
           'cm23': 'cortex-m23', 'cm33': 'cortex-m33',
           'cortexm0': 'cortex-m0', 'cortexm0plus': 'cortex-m0plus',
           'cortexm3': 'cortex-m3', 'cortexm4': 'cortex-m4',
           'cortexm7': 'cortex-m7', 'cortexm23': 'cortex-m23',
           'cortexm33': 'cortex-m33'}
    if normalized_cpu in arm:
        architecture, default_compiler = 'arm', 'arm-none-eabi-gcc'
        flags = ('-mcpu=' + arm[normalized_cpu], '-mthumb')
    elif name.startswith(('ESP32C3', 'ESP32C6', 'ESP32H2', 'ESP32C2', 'ESP32C5', 'ESP32P4')):
        architecture, default_compiler, flags = 'riscv', 'riscv32-esp-elf-gcc', ()
    elif name.startswith(('ESP32S2', 'ESP32S3', 'ESP32')):
        architecture, default_compiler, flags = 'xtensa', 'xtensa-esp-elf-gcc', ()
    elif normalized_cpu in ('lx6', 'lx7', 'xtensalx6', 'xtensalx7'):
        architecture, default_compiler, flags = 'xtensa', 'xtensa-esp-elf-gcc', ()
    elif normalized_cpu.startswith('riscv') and name.startswith('ESP'):
        architecture, default_compiler, flags = 'riscv', 'riscv32-esp-elf-gcc', ()
    else:
        if not compiler or not compiler_flags:
            raise ValueError(f'Unsupported or ambiguous CPU for {mcu!r} (SVD cpu={cpu_name!r}); supply compiler and compiler_flags')
        architecture, default_compiler, flags = 'custom', compiler, ()
    selected = compiler or default_compiler
    selected_flags = tuple(compiler_flags) if compiler_flags else flags
    if any(not isinstance(flag, str) or not flag.startswith('-') for flag in selected_flags):
        raise ValueError('compiler_flags must be compiler option strings')
    return Toolchain(architecture, selected, selected_flags, shutil.which(selected) is not None, cpu_name)
