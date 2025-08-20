import random
import re
import os

# List of VLAN features to potentially remove
vlan_features = [
    r'^vlan \d+',
    r'^ name \S+',
    r'^interface vlan \d+',
    r'^ ip address \d+\.\d+\.\d+\.\d+ \d+\.\d+\.\d+\.\d+',
    r'^ no shutdown',
    r'^interface \S+',
    r'^ switchport mode access',
    r'^ switchport access vlan \d+',
    r'^ switchport trunk encapsulation dot1q',
    r'^ switchport mode trunk',
    r'^ switchport trunk allowed vlan (?:add )?\d+(?:,\d+)*',
    r'^ switchport trunk native vlan \d+',
    r'^ spanning-tree portfast',
    r'^ spanning-tree bpduguard enable',
    r'^ description .*'
]

input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\VLANError\3VLAN"
os.makedirs(output_dir, exist_ok=True)

def remove_vlan_lines(config_text, n=3):
    lines = config_text.splitlines()
    matches = [i for i, line in enumerate(lines) if any(re.match(p, line.lstrip()) for p in vlan_features)]
    
    if not matches:
        return config_text, []

    selected = random.sample(matches, min(n, len(matches)))
    removed = [lines[i] for i in selected]
    kept = [line for idx, line in enumerate(lines) if idx not in selected]

    return "\n".join(kept), removed

# Adjust the range depending on which routers you're targeting
for i in range(79, 85):  # Example: router79 to router85
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_broken_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ Not found.")
        continue

    with open(input_path) as f:
        original = f.read()

    modified, removed = remove_vlan_lines(original, n=3)

    with open(output_path, 'w') as f:
        f.write(modified)

    print(f"[Router {i}] ✅ Removed VLAN line(s):")
    for line in removed:
        print(line)
