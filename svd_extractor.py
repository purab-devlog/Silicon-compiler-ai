"""Lossless-source CMSIS-SVD catalog and on-demand peripheral context extraction.

The original .svd is always retained. The JSON catalog is a structured retrieval
view, not a replacement for the source XML. No arbitrary register/description caps.
"""
import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_int(value, default=None):
    if value is None or not str(value).strip():
        return default
    value = str(value).strip()
    try:
        if value.startswith('#'):
            return int(value[1:].replace('x', '0').replace('X', '1'), 2)
        return int(value, 0) if value.lower().startswith(('0x', '0b', '0o')) else int(value, 10)
    except ValueError:
        return default


def _strip_namespaces(root):
    for node in root.iter():
        if isinstance(node.tag, str) and '}' in node.tag:
            node.tag = node.tag.rsplit('}', 1)[-1]


def _raw(node):
    """Preserve all XML attributes, text and descendants in a JSON-compatible tree."""
    return {'tag': node.tag, 'attributes': dict(node.attrib), 'text': node.text,
            'children': [_raw(child) for child in node]}


def _merge(base, override):
    """Merge derivedFrom nodes, preserving inherited child nodes unless overridden."""
    result = copy.deepcopy(base)
    result.attrib.update(override.attrib)
    for child in override:
        key = (child.tag, child.findtext('name') if child.tag in ('register', 'field', 'cluster') else None)
        matches = [old for old in result if (old.tag, old.findtext('name') if old.tag in ('register', 'field', 'cluster') else None) == key]
        if matches:
            old = matches[0]
            result.remove(old)
            result.append(_merge(old, child) if child.tag in ('register', 'field', 'cluster', 'fields', 'registers') else copy.deepcopy(child))
        else:
            result.append(copy.deepcopy(child))
    return result


def _resolve_peripheral(name, nodes, seen=None):
    seen = set() if seen is None else seen
    if name in seen:
        raise ValueError(f'Circular peripheral derivedFrom chain: {name}')
    node = nodes[name]
    parent = node.get('derivedFrom')
    if not parent:
        return copy.deepcopy(node)
    parent_key = parent.upper()
    if parent_key not in nodes:
        raise ValueError(f'Unresolved derivedFrom {parent!r} for {name}')
    resolved = _merge(_resolve_peripheral(parent_key, nodes, seen | {name}), node)
    resolved.attrib.pop('derivedFrom', None)
    return resolved


def _field_data(field, default_access=None):
    offset = _parse_int(field.findtext('bitOffset'))
    width = _parse_int(field.findtext('bitWidth'))
    bit_range = field.findtext('bitRange')
    if bit_range and ':' in bit_range:
        match = re.fullmatch(r'\[(\d+):(\d+)\]', bit_range.strip())
        if match:
            msb, lsb = map(int, match.groups())
            offset, width = lsb, msb - lsb + 1
    if offset is None:
        offset = _parse_int(field.findtext('lsb'))
    if width is None and offset is not None:
        msb = _parse_int(field.findtext('msb'))
        width = msb - offset + 1 if msb is not None else None
    return {'name': field.findtext('name'), 'desc': field.findtext('description') or '',
            'offset': offset, 'width': width, 'bits': f'[{offset + width - 1}:{offset}]' if offset is not None and width is not None else None,
            'access': field.findtext('access') or default_access,
            'derivedFrom': field.get('derivedFrom'), 'raw': _raw(field)}


def _resolve_fields(reg, default_access):
    nodes = {f.findtext('name'): f for f in reg.findall('./fields/field') if f.findtext('name')}
    def resolve(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f'Circular field derivedFrom: {name}')
        node = nodes[name]
        parent = node.get('derivedFrom')
        if not parent:
            return node
        key = parent.split('.')[-1]
        if key not in nodes:
            return node  # Keep unresolved source reference visible
        return _merge(resolve(key, seen | {name}), node)
    return [_field_data(resolve(name), default_access) for name in nodes]


def _registers(container, parent_offset=0, prefix='', defaults=None):
    defaults = defaults or {}
    if container is None:
        return []
    output = []
    for node in container:
        if node.tag == 'cluster':
            offset = _parse_int(node.findtext('addressOffset'), 0)
            cluster_name = node.findtext('name') or 'CLUSTER'
            output.extend(_registers(node, parent_offset + offset, prefix + cluster_name + '.', defaults))
            continue
        if node.tag != 'register':
            continue
        offset = _parse_int(node.findtext('addressOffset'))
        name = node.findtext('name') or 'UNKNOWN'
        access = node.findtext('access') or defaults.get('access')
        fields = _resolve_fields(node, access)
        output.append({'name': prefix + name, 'offset': parent_offset + offset if offset is not None else None,
                       'desc': node.findtext('description') or '',
                       'reset': _parse_int(node.findtext('resetValue'), defaults.get('resetValue')),
                       'resetMask': _parse_int(node.findtext('resetMask'), defaults.get('resetMask')),
                       'size': _parse_int(node.findtext('size'), defaults.get('size')),
                       'access': access, 'derivedFrom': node.get('derivedFrom'),
                       'dim': node.findtext('dim'), 'dimIncrement': node.findtext('dimIncrement'),
                       'fields': fields, 'raw': _raw(node)})
    return output


def build_svd_catalog(svd_path: Path, output_path: Path | None = None) -> dict:
    """Parse all peripherals without truncation; optionally persist a JSON retrieval catalog."""
    svd_path = Path(svd_path)
    root = ET.parse(svd_path).getroot()
    _strip_namespaces(root)
    if root.tag != 'device':
        raise ValueError(f'Not a CMSIS-SVD device: {svd_path}')
    nodes = {n.findtext('name').upper(): n for n in root.findall('./peripherals/peripheral') if n.findtext('name')}
    defaults = {'size': _parse_int(root.findtext('size')),
                'access': root.findtext('access'),
                'resetValue': _parse_int(root.findtext('resetValue')),
                'resetMask': _parse_int(root.findtext('resetMask'))}
    peripherals = {}
    for name, node in nodes.items():
        resolved = _resolve_peripheral(name, nodes)
        base = _parse_int(resolved.findtext('baseAddress'))
        periph_defaults = {key: _parse_int(resolved.findtext(key), defaults[key]) if key != 'access'
                           else resolved.findtext(key) or defaults[key] for key in defaults}
        regs = _registers(resolved.find('registers'), defaults=periph_defaults)
        for reg in regs:
            reg['address'] = base + reg['offset'] if base is not None and reg['offset'] is not None else None
        peripherals[name] = {'name': name, 'baseAddress': base,
                             'desc': resolved.findtext('description') or '',
                             'derivedFrom': node.get('derivedFrom'),
                             'registers': regs, 'raw': _raw(node)}
    catalog = {'schemaVersion': 2, 'source': str(svd_path),
               'device': root.findtext('name') or svd_path.stem,
               'cpu': root.findtext('./cpu/name'),
               'metadata': {child.tag: child.text for child in root if child.tag not in ('peripherals',)},
               'peripherals': peripherals}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding='utf-8')
    return catalog


def extract_peripheral_context(svd_path: Path, target_peripherals: list[str],
                               include_system_deps: bool = True,
                               max_registers_per_periph: int | None = None) -> str:
    """Backward-compatible text interface. Default is NO truncation.

    max_registers_per_periph is accepted for legacy callers but intentionally ignored:
    truncation would silently remove hardware facts.
    """
    catalog = build_svd_catalog(svd_path)
    selected = list(dict.fromkeys(p.strip().upper() for p in target_peripherals if p.strip()))
    if include_system_deps:
        for dep in ('RCC', 'CLOCK', 'GPIOA', 'GPIOB'):
            if dep in catalog['peripherals'] and dep not in selected:
                selected.append(dep)
    lines = [f"TARGET MCU: {catalog['device']} ({catalog['cpu'] or 'CPU unspecified'})",
             f"SOURCE SVD: {Path(svd_path).name}", '=' * 60]
    for name in selected:
        p = catalog['peripherals'].get(name)
        if not p:
            lines.append(f"[WARNING] Peripheral '{name}' not found in {Path(svd_path).name}.")
            continue
        base = f"0x{p['baseAddress']:08X}" if p['baseAddress'] is not None else 'UNKNOWN'
        lines.extend([f"\nPERIPHERAL: {name} | BASE_ADDRESS: {base}", p['desc'], '-' * 60])
        for reg in p['registers']:
            addr = f"0x{reg['address']:08X}" if reg['address'] is not None else 'UNKNOWN'
            offset = f"0x{reg['offset']:X}" if reg['offset'] is not None else 'UNKNOWN'
            reset = f"0x{reg['reset']:X}" if reg['reset'] is not None else 'UNSPECIFIED'
            lines.append(f"  REG {reg['name']} @ Offset {offset} (Abs: {addr}) | Reset: {reset} | Access: {reg['access'] or 'UNSPECIFIED'} | {reg['desc']}")
            for f in reg['fields']:
                lines.append(f"    - {f['name']} {f['bits'] or '[UNSPECIFIED]'} | Access: {f['access'] or 'UNSPECIFIED'}: {f['desc']}")
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build complete SVD catalog and optional focused text context')
    parser.add_argument('svd', type=Path)
    parser.add_argument('--peripherals', nargs='*', default=[])
    args = parser.parse_args()
    path = args.svd.with_suffix('.catalog.json')
    catalog = build_svd_catalog(args.svd, path)
    print(f"Saved full catalog: {path} ({len(catalog['peripherals'])} peripherals)")
    if args.peripherals:
        context = extract_peripheral_context(args.svd, args.peripherals)
        context_path = args.svd.with_suffix('.context.txt')
        context_path.write_text(context, encoding='utf-8')
        print(f'Saved selected context: {context_path}')
