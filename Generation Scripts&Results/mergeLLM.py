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
base_tpl = ios_template_DIR.read_text()

# Regex to extract router identifier (e.g., 'router1')
FNAME_RE = re.compile(r"^(router\d+)_config\.txt$", re.IGNORECASE)

# Initialize LangChain ChatOpenAI client
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=10000)

def read_config(path: Path) -> str:
    return path.read_text() if path and path.exists() else ""

def extract_used_interfaces(cfg: str) -> list[str]:
    return re.findall(r'^interface (\S+)', cfg, flags=re.MULTILINE)

def extract_vlan_subinterfaces(cfg: str) -> list[tuple[str, str]]:
    return re.findall(r'interface (\S+)\n(?: .*?\n)*? ip address (\d+\.\d+\.\d+\.\d+)', cfg, flags=re.MULTILINE)

def merge_with_llm(router_id: str, eigrp_cfg: str, ospf_cfg: str, rip_cfg: str, aaa_cfg: str, vlan_cfg: str,
                   max_retries: int = 3) -> str:
    system_message = (
        "You are a Cisco IOS network configuration assistant. "
        "You will be provided with multiple protocol-specific configurations (OSPF, EIGRP, RIP, AAA) and a list of interface definitions (including VLAN subinterfaces). "
        "Your task is to merge all of them into a single IOS configuration file, combining all interface blocks when multiple protocols affect the same interface. "
        "\n\n"
        "⚠️ IMPORTANT RULES:\n"
        "- If all physical interfaces are used by OSPF, use the defined VLAN subinterfaces (e.g., FastEthernet0/0.x) for EIGRP.\n"
        "- RIP and EIGRP can overlap with other protocols, but should minimize this.\n"
        "- Keep EIGRP keychains under the chosen EIGRP interfaces"
        "- Keep ALL authentication under the appropriate interface if it is there already \n"
        "- OSPF interfaces must be used exactly as defined and must not be reused for EIGRP.\n"
        "- AAA server IPs must use subnets where the router has the .1 IP and the server has .2.\n"
        "\n"
        "👷 Additional Instructions:\n"
        "- All interface blocks must be merged together (no duplicates).\n"
        "- Subinterfaces (e.g., FastEthernet0/0.100) behave like normal interfaces.\n"
        "- Order the final config as follows: OSPF → EIGRP → RIP → AAA.\n"
        "- If any protocol config is missing, include a placeholder: '<protocol> section: [UNKNOWN]'.\n"
        "- Output ONLY the final merged configuration text, no explanation."
    )

    # Build interface usage preamble
    ospf_ifaces = extract_used_interfaces(ospf_cfg)
    vlan_subifs = extract_vlan_subinterfaces(vlan_cfg)

    preamble = "Before merging, analyze the interface usage:\n\n"
    if ospf_ifaces:
        preamble += "- Interfaces currently used by OSPF:\n"
        preamble += "\n".join(f"  - {iface}" for iface in ospf_ifaces)
        preamble += "\n\n"
    if vlan_subifs:
        preamble += "- VLAN subinterfaces available for EIGRP:\n"
        preamble += "\n".join(f"  - {iface} ({ip}/24)" for iface, ip in vlan_subifs)
        preamble += "\n\n"
    preamble += "Do not use OSPF interfaces for EIGRP. Use the above VLAN interfaces instead."

    # Construct user message
    user_message = f"""{preamble}

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

    # 🔹 Filter to only routers 51–69
    all_ids = [rid for rid in all_ids if 16 <= int(rid.replace('router', '')) <= 33]

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
