import random
import re
import os

ospf_features = [
    r'^router ospf \d+$',
    r'^area \d+ authentication message-digest$',
    r'^ip ospf message-digest-key \d+ md5 .+$'
]

input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\OSPFError\2OSPF"
os.makedirs(output_dir, exist_ok=True)

def remove_ospf_lines(config_text, n=2):
    lines = config_text.splitlines()
    matches = [i for i, line in enumerate(lines) if any(re.match(p, line.lstrip()) for p in ospf_features)]
    
    if not matches:
        return config_text, []
    
    selected = random.sample(matches, min(n, len(matches)))
    removed = [lines[i] for i in selected]
    kept = [line for idx, line in enumerate(lines) if idx not in selected]

    return "\n".join(kept), removed

for i in range(40, 46):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_broken_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ Not found.")
        continue

    with open(input_path) as f:
        original = f.read()

    modified, removed = remove_ospf_lines(original, n=2)

    with open(output_path, 'w') as f:
        f.write(modified)

    print(f"[Router {i}] ✅ Removed lines:")
    for line in removed:
        print(line)
