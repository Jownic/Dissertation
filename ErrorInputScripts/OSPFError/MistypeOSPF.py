import random
import re
import os

# List of OSPF features to potentially mistype
ospf_feature_patterns = [
    r'^router ospf \d+$',
    r'^area \d+ authentication message-digest$',
    r'^ip ospf message-digest-key \d+ md5 .+$'
]

input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\OSPFError\OSPFMistype"
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

def mistype_one_ospf_feature(config_text, seed):
    """
    Finds all lines matching any OSPF feature pattern,
    picks one at random using seed,
    and introduces a typo in that line using a *different* seed.
    """
    lines = config_text.splitlines()
    
    matches = [
        (i, line) for i, line in enumerate(lines)
        if any(re.match(pat, line) for pat in ospf_feature_patterns)
    ]
    
    if not matches:
        return config_text, None, None

    random.seed(seed)  # Seed for consistent line choice per router
    idx, original_line = random.choice(matches)

    # Use a different seed for typo (e.g. offset seed or use random.randint)
    typo_seed = seed * 1337 + 42
    typo_line = introduce_typo(original_line, typo_seed)

    lines[idx] = typo_line
    return "\n".join(lines), original_line, typo_line


# Process router34 through router51
for i in range(34, 52):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_typo_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ File not found: {input_path}")
        continue

    with open(input_path, 'r') as f:
        original = f.read()

    modified, orig_line, typo_line = mistype_one_ospf_feature(original, seed=i)

    with open(output_path, 'w') as f:
        f.write(modified)

    if orig_line:
        print(f"[Router {i}] ✅ Introduced typo:")
        print(f"    Original: {orig_line}")
        print(f"    Mistyped: {typo_line}")
    else:
        print(f"[Router {i}] ⚠️ No OSPF feature lines found to mistype.")
