import random
import re
import os

# List of RIP features to potentially mistype
rip_feature_patterns = [
    r'^router rip',
    r'^ version 2',
    r'^ network \d+\.\d+\.\d+\.\d+',
    r'^ no auto-summary',
    r'^ interface \S+',
    r'^ ip rip authentication mode md5',
    r'^ ip rip authentication key-chain \S+',
    r'^ key chain \S+',
    r'^ key \d+',
    r'^  key-string \S+',
    r'^ passive-interface \S+'
]


input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\RIPError\RIPMistype"
os.makedirs(output_dir, exist_ok=True)

def introduce_typo(line, seed):
    """
    Introduce a simple typo into the given line:
    - If the line is longer than 1 character, delete one random character.
    """
    random.seed(seed)
    if len(line) <= 1:
        return line
    idx = random.randrange(len(line))
    return line[:idx] + line[idx+1:]

def mistype_one_rip_feature(config_text, seed):
    """
    Finds all lines matching any RIP feature pattern,
    picks one at random (with the given seed),
    and introduces a typo in that line.
    Returns the modified config text, the original line, and the mistyped line.
    """
    lines = config_text.splitlines()
    # Find all RIP feature lines
    matches = [
        (i, line) for i, line in enumerate(lines)
        if any(re.match(pat, line) for pat in rip_feature_patterns)
    ]
    if not matches:
        return config_text, None, None

    random.seed(seed)
    idx, original_line = random.choice(matches)
    typo_line = introduce_typo(original_line, seed)

    # Replace only that line
    lines[idx] = typo_line
    return "\n".join(lines), original_line, typo_line

# Process router52 through router66
for i in range(52, 67):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_typo_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ File not found: {input_path}")
        continue

    with open(input_path, 'r') as f:
        original = f.read()

    modified, orig_line, typo_line = mistype_one_rip_feature(original, seed=i)

    with open(output_path, 'w') as f:
        f.write(modified)

    if orig_line:
        print(f"[Router {i}] ✅ Introduced typo:")
        print(f"    Original: {orig_line}")
        print(f"    Mistyped: {typo_line}")
    else:
        print(f"[Router {i}] ⚠️ No RIP feature lines found to mistype.")
