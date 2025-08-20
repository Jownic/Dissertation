#!/usr/bin/env python3
import os
import re
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from openai import OpenAI, RateLimitError

os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Configuration ───────────────────────────────────────────────────────────
EIGRP_DIR = Path(r"Z:\Thesis\Generation Scripts&Results\EIGRP")
OSPF_DIR  = Path(r"Z:\Thesis\Generation Scripts&Results\OSPFs")
RIP_DIR   = Path(r"Z:\Thesis\Generation Scripts&Results\RIP")
VLAN_DIR  = Path(r"Z:\Thesis\Generation Scripts&Results\VLANS")
AAA_DIR   = Path(r"Z:\Thesis\Generation Scripts&Results\AAA")
ios_template_DIR = Path(r"Z:\Thesis\ios_template.txt")
MERGE_DIR = Path(r"Z:\Thesis\Generation Scripts&Results\merge")
MERGE_DIR.mkdir(parents=True, exist_ok=True)
base_tpl = (ios_template_DIR).read_text()

# Regex to extract router identifier (e.g., 'router1')
FNAME_RE = re.compile(r"^(router\d+)_config\.txt$", re.IGNORECASE)

# Initialize LangChain ChatOpenAI client
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=2000)

def read_config(path: Path) -> str:
    return path.read_text() if path and path.exists() else ""

def merge_with_llm(router_id: str, eigrp_cfg: str, ospf_cfg: str, rip_cfg: str, aaa_cfg: str, vlan_cfg: str,
                   max_retries: int = 3) -> str:
    system_message = (
        "You are a network configuration assistant. "
        "Merge the provided OSPF, EIGRP, RIP, AAA and VLAN configurations into a single Cisco IOS config. "
        "OSPF areas and EIGRP AS numbers may differ per router. "
        "Allocate EIGRP only on interfaces without OSPF; if no free physical interface exists, use a VLAN subinterface. "
        "Treat subinterfaces as full interfaces. "
        "Ensure EIGRP network and neighbor statements use each interface's subnet (.1 → .2), and never overlap with OSPF subnets. "
        "Ensure RIP is merged realistically after EIGRP, and while it may share a few interfaces with other protocols, minimize overlap. "
        "Ensure OSPF network statements match the exact interface prefixes and do not duplicate any EIGRP or RIP addresses. "
        "Ensure AAA IP and server directives use local interfaces with subnets ending in .1 for the router and .2 for the TACACS+ server. "
        "Order the final config as follows: first OSPF, then EIGRP, then RIP, then AAA. "
        "If the router has no snippet for a protocol, output a line: '<protocol> section: [UNKNOWN]'. "
        "Provide only the merged IOS configuration text. Ensure that all `interface` blocks are merged: if multiple configurations affect the same interface, combine them into a single block per interface with all relevant commands."

    )
    user_message = f"""
Configuration merge request for **{router_id}**:

Here’s the **base IOS template** you should merge into:
{base_tpl}

--- OSPF Configuration ---
{ospf_cfg or '[No OSPF config present]'}

--- EIGRP Configuration ---
{eigrp_cfg or '[No EIGRP config present]'}

--- RIP Configuration ---
{rip_cfg or '[No RIP config present]'}

--- VLAN Configuration ---
{vlan_cfg or '[No VLAN config present]'}

--- AAA Configuration ---
{aaa_cfg or '[No AAA config present]'}

Please output only the merged IOS configuration.
"""
    messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAI API rate limit failure after retries.")

def main():
    eigrp_map = {m.group(1): p for p in EIGRP_DIR.glob('router*_config.txt') if (m := FNAME_RE.match(p.name))}
    ospf_map  = {m.group(1): p for p in OSPF_DIR .glob('router*_config.txt') if (m := FNAME_RE.match(p.name))}
    rip_map   = {m.group(1): p for p in RIP_DIR  .glob('router*_config.txt') if (m := FNAME_RE.match(p.name))}
    vlan_map  = {m.group(1): p for p in VLAN_DIR .glob('router*_config.txt') if (m := FNAME_RE.match(p.name))}
    aaa_map   = {m.group(1): p for p in AAA_DIR  .glob('router*_config.txt') if (m := FNAME_RE.match(p.name))}

    all_ids = sorted(set(eigrp_map) | set(ospf_map) | set(rip_map) | set(vlan_map) | set(aaa_map),
                     key=lambda x: int(x.replace('router', '')))

    print(f"Found routers to merge: {all_ids}")
    for router_id in all_ids:
        eigrp_cfg = read_config(eigrp_map.get(router_id, Path()))
        ospf_cfg  = read_config(ospf_map .get(router_id, Path()))
        rip_cfg   = read_config(rip_map  .get(router_id, Path()))
        vlan_cfg  = read_config(vlan_map .get(router_id, Path()))
        aaa_cfg   = read_config(aaa_map  .get(router_id, Path()))

        print(f"Merging configs for {router_id}...")
        merged_cfg = merge_with_llm(router_id, eigrp_cfg, ospf_cfg, rip_cfg, aaa_cfg, vlan_cfg)
        out_path = MERGE_DIR / f"{router_id}_merged_config.txt"
        out_path.write_text(merged_cfg + "\n")
        print(f"  ✔️ Wrote {out_path}")

if __name__ == '__main__':
    main()
