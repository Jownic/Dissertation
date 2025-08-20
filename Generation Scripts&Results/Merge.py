#!/usr/bin/env python3
import re
import random
import ipaddress
from pathlib import Path
from collections import OrderedDict

# —— Edit these paths as needed ——
OSPF_DIR   = Path("OSPFs")
VLAN_DIR   = Path("VLANs")
EIGRP_DIR  = Path("EIGRP")
MERGE_DIR  = Path("merge")
MERGE_DIR.mkdir(exist_ok=True)
FNAME_RE   = re.compile(r"^(router\d+)_config\.txt$", re.IGNORECASE)

# Regex patterns
NO_IP_PATTERN       = re.compile(r"^\s*no ip address\s*$", re.IGNORECASE)
IP_ADDR_PATTERN     = re.compile(
    r"^(?P<prefix>\s*ip address )"
    r"(?P<ip>\d+\.\d+\.\d+\.\d+)"
    r"(?P<suffix>\s+255\.255\.255\.0)$"
)
BROKEN_MASK_PATTERN = re.compile(
    r"^(?P<prefix>\s*ip address )"
    r"(?P<ip>\d+\.\d+\.\d+\.\d+)\s+255\.255\.255\s*$"
)
OSPF_NET_PATTERN    = re.compile(r"^\s*network\s+(?P<ip>\d+\.\d+\.\d+\.0)\s+0\.0\.0\.255", re.IGNORECASE)
HOSTNAME_PATTERN    = re.compile(r"^\s*hostname\s+\S+", re.IGNORECASE)
EIGRP_HELLO_PATTERN = re.compile(r"^\s*ip hello-interval eigrp.*", re.IGNORECASE)
EIGRP_HOLD_PATTERN  = re.compile(r"^\s*ip hold-time eigrp.*", re.IGNORECASE)


def parse_config(path):
    if path is None:
        return [], OrderedDict(), [], []
    gl, ints, ospf, eigrp = [], OrderedDict(), [], []
    state, cur = 'global', None
    for line in path.read_text().splitlines():
        ln = line.rstrip()
        if ln in ('!', 'end'):
            continue
        m = re.match(r"^interface\s+(\S+)", ln)
        if m:
            state, cur = 'intf', m.group(1)
            ints[cur] = [ln]
        elif ln.lower().startswith('router ospf'):
            state = 'ospf'; ospf.append(ln)
        elif ln.lower().startswith('router eigrp'):
            state = 'eigrp'; eigrp.append(ln)
        else:
            if state == 'intf':
                if NO_IP_PATTERN.match(ln):
                    continue
                ints[cur].append(ln)
            elif state == 'ospf':
                ospf.append(ln)
            elif state == 'eigrp':
                eigrp.append(ln)
            else:
                gl.append(ln)
    return gl, ints, ospf, eigrp


def resolve_ip_conflicts(interfaces):
    used, out = set(), OrderedDict()
    for name, blk in interfaces.items():
        new_blk = []
        for ln in blk:
            if m := BROKEN_MASK_PATTERN.match(ln):
                ln = f"{m.group('prefix')}{m.group('ip')} 255.255.255.0"
            if m := IP_ADDR_PATTERN.match(ln):
                o1, o2, o3, o4 = m.group('ip').split('.')
                octet3 = int(o3)
                if m.group('ip') in used:
                    for cand3 in range(1, 255):
                        cand_ip = f"{o1}.{o2}.{cand3}.{o4}"
                        if cand_ip not in used:
                            octet3 = cand3
                            break
                used.add(f"{o1}.{o2}.{octet3}.{o4}")
                ln = f"{m.group('prefix')}{o1}.{o2}.{octet3}.{o4}{m.group('suffix')}"
            new_blk.append(ln)
        out[name] = new_blk
    return out


def merge_interfaces(ospf_if, vlan_if, eigrp_if):
    merged = OrderedDict(ospf_if)
    # merge VLAN
    for name, blk in vlan_if.items():
        if name in merged:
            base = merged[name]
            for ln in blk[1:]:
                if NO_IP_PATTERN.match(ln):
                    continue
                if ln not in base:
                    base.append(ln)
        else:
            merged[name] = [ln for ln in blk if not NO_IP_PATTERN.match(ln)]
    # merge EIGRP non-IP lines
    for name, blk in eigrp_if.items():
        if name in merged:
            base = merged[name]
            for ln in blk[1:]:
                if ln.strip().startswith('ip address'):
                    continue
                if EIGRP_HELLO_PATTERN.match(ln) or EIGRP_HOLD_PATTERN.match(ln):
                    continue
                if ln not in base:
                    base.append(ln)
        else:
            merged[name] = [ln for ln in blk if not ln.strip().startswith('ip address')]
    return resolve_ip_conflicts(merged)


def extract_hostname(globs):
    for ln in globs:
        if HOSTNAME_PATTERN.match(ln):
            return ln
    return None


def merge_three(ospf_p, vlan_p, eigrp_p):
    gO, iO, o_lines, _ = parse_config(ospf_p)
    gV, iV, _, _       = parse_config(vlan_p)
    gE, iE, _, e_lines = parse_config(eigrp_p)

    # 1) Merge globals
    hostname = extract_hostname(gO) or extract_hostname(gE) or extract_hostname(gV)
    combined = gO + gE + gV
    seen, globals_out = set(), []
    for ln in combined:
        if HOSTNAME_PATTERN.match(ln):
            continue
        if ln not in seen:
            seen.add(ln)
            globals_out.append(ln)
    if hostname:
        globals_out.insert(0, hostname)

    # 2) Merge base interfaces
    merged_if = merge_interfaces(iO, iV, iE)
    # remove unwanted phantom interface
    merged_if.pop('FastEthernet0/1', None)

    # 3) Identify OSPF network prefixes
    ospf_prefixes = set()
    for ln in o_lines:
        if m := OSPF_NET_PATTERN.match(ln):
            ospf_prefixes.add(m.group('ip'))

    # 4) Collect free interfaces (prefix not in ospf and no OSPF commands)
    free_ifaces = []
    for name, blk in merged_if.items():
        # skip interfaces that have any OSPF commands
        if any(line.strip().startswith('ip ospf') for line in blk):
            continue
        for ln in blk:
            if m2 := IP_ADDR_PATTERN.match(ln):
                prefix = '.'.join(m2.group('ip').split('.')[:3]) + '.0'
                if prefix not in ospf_prefixes:
                    free_ifaces.append((name, prefix))
                break

    # 5) Extract original EIGRP timers from interface blocks (preserve order) from interface blocks (preserve order)
    timer_list = []
    for name, blk in iE.items():
        hello = next((l.strip() for l in blk if EIGRP_HELLO_PATTERN.match(l)), None)
        hold  = next((l.strip() for l in blk if EIGRP_HOLD_PATTERN.match(l)), None)
        if hello or hold:
            timer_list.append((name, hello, hold))

    # 6) Reassign timers to free interfaces in order
    for idx, (_, hello, hold) in enumerate(timer_list):
        if idx < len(free_ifaces):
            if_name, _ = free_ifaces[idx]
            blk = merged_if[if_name]
            for j, ln in enumerate(blk):
                if IP_ADDR_PATTERN.match(ln):
                    if hello and hello not in blk:
                        blk.insert(j+1, f" {hello}")
                    if hold and hold not in blk:
                        blk.insert(j+2, f" {hold}")
                    break

    # 7) Extract original EIGRP stanza nets/neighbors
    orig_nets = [ln for ln in e_lines if ln.strip().startswith('network ')]
    orig_nbrs = [ln for ln in e_lines if ln.strip().startswith('neighbor ')]

    # 8) Reassign nets/neighbors to free interfaces in same order
    new_nets = []
    new_nbrs = []
    for idx, net in enumerate(orig_nets):
        if idx < len(free_ifaces):
            _, prefix = free_ifaces[idx]
            new_nets.append(f"network {prefix} 0.0.0.255")
        else:
            new_nets.append(net)
    for idx, nbr in enumerate(orig_nbrs):
        if idx < len(free_ifaces):
            if_name, prefix = free_ifaces[idx]
            new_nbrs.append(f"neighbor {'.'.join(prefix.split('.')[:-1])}.2 {if_name}")
        else:
            new_nbrs.append(nbr)

    # 9) Assemble final config
    lines = ['!', *globals_out, '!']
    for blk in merged_if.values():
        lines.extend(blk)
        lines.append('!')

    if o_lines:
        hdr = o_lines[0] if o_lines[0].startswith('router ospf') else 'router ospf 1'
        lines.append(hdr)
        lines.extend(o_lines[1:])
        lines.append('!')

    if e_lines:
        asn = re.match(r'router eigrp\s+(\d+)', e_lines[0]).group(1)
        lines.append(f"router eigrp {asn}")
        # preserve only non-interface, non-line, non-network, non-neighbor settings
        for ln in e_lines[1:]:
            stripped = ln.strip()
            if stripped.startswith(('network ', 'neighbor ', 'line ')):
                continue
            if re.match(r'^(exec-timeout|logging synchronous|privilege level|no login)', stripped):
                continue
            lines.append(f" {stripped}")
        # append reassigned neighbors and networks
        for ln in new_nbrs: lines.append(f" {ln}")
        for ln in new_nets: lines.append(f" {ln}")
        lines.append('!')

    lines.append('end')
    return '\n'.join(lines) + '\n'


def main():
    ospf_map  = {m.group(1).lower():p for p in OSPF_DIR.glob('router*_config.txt') for m in [FNAME_RE.match(p.name)] if m}
    vlan_map  = {m.group(1).lower():p for p in VLAN_DIR.glob('router*_config.txt') for m in [FNAME_RE.match(p.name)] if m}
    eigrp_map = {m.group(1).lower():p for p in EIGRP_DIR.glob('router*_config.txt') for m in [FNAME_RE.match(p.name)] if m}
    all_ids   = sorted(set(ospf_map)|set(vlan_map)|set(eigrp_map), key=lambda x:int(x.replace('router','')))

    print('Merging configs for:', all_ids)
    for rid in all_ids:
        out = MERGE_DIR / f'{rid}_merged_config.txt'
        out.write_text(merge_three(ospf_map.get(rid), vlan_map.get(rid), eigrp_map.get(rid)))
        print(f'  ✅ {rid}')

if __name__=='__main__':
    main()
