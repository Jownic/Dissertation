import random
import re
import os

# List of RIP features to potentially remove
rip_features = [
    r'router rip',
    r'version 2',
    r'no auto-summary',
    r'ip rip authentication mode md5',
    r'ip rip authentication key-chain \S+',
    r'key chain \S+',
    r'key \d+',
    r'key-string \S+',
    r'passive-interface \S+',
    r'^network \d+\.\d+\.\d+\.\d+$',
    r'redistribute \S+',
    r'maximum-paths \d+',
    r'distance \d+',
    r'offset-list \d+ \S+'
]


input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\RIPError\3RIP"
os.makedirs(output_dir, exist_ok=True)

def remove_three_rip_regex_blocks(config_text, n=3):
    lines = config_text.splitlines()
    matching_lines = []

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        for pattern in rip_features:
            if re.match(pattern, stripped_line):
                matching_lines.append(i)
                break

    print(f"[DEBUG] Found {len(matching_lines)} matching RIP lines.")
    if len(matching_lines) == 0:
        return config_text, []

    selected = random.sample(matching_lines, min(n, len(matching_lines)))
    removed_lines = [lines[i] for i in selected]
    kept_lines = [line for idx, line in enumerate(lines) if idx not in selected]

    return "\n".join(kept_lines), removed_lines

# Process router62 through router66
for i in range(62, 67):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_broken_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ File not found: {input_path}")
        continue

    with open(input_path, 'r') as f:
        original = f.read()

    modified, removed = remove_three_rip_regex_blocks(original, n=3)

    with open(output_path, 'w') as f:
        f.write(modified)

    if removed:
        print(f"[Router {i}] ✅ Removed lines:")
        for entry in removed:
            print(entry)
    else:
        print(f"[Router {i}] ⚠️ No matching RIP entries to remove.")