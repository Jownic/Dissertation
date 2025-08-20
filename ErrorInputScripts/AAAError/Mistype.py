import random
import re
import os

# List of AAA features to potentially mistype
aaa_feature_patterns = [
    r'^aaa new-model',
    r'^aaa authentication login LOGIN-LIST group tacacs\+ local$',
    r'^aaa authentication enable .*',
    r'^aaa authentication dot1x .*',
    r'^aaa authentication ppp .*',
    r'^aaa authentication arap .*',
    r'^aaa authentication attempts max-fail .*',
    r'^aaa authorization exec .*',
    r'^aaa authorization config-commands.*',
    r'^aaa authorization network .*',
    r'^aaa authorization reverse-access .*',
    r'^aaa accounting exec EXEC-ACC start-stop group tacacs\+$',
    r'^aaa accounting commands 15 .*',
    r'^aaa accounting connection .*',
    r'^aaa accounting network .*',
    r'^aaa accounting system .*',
    r'^aaa accounting vrrs .*',
    r'^aaa accounting delay-start',
    r'^aaa session-id common',
    r'^username .+ secret .+',
    r'^enable secret .+',
    r'^service password-encryption',
    r'^banner (exec|login|motd) [\s\S]+?\^C',
    r'^snmp-server community .+',
    r'^no snmp-server',
    r'^snmp-server host .+',
    r'^snmp-server enable traps snmp',
    r'^snmp-server group .+ v3 priv',
    r'^snmp-server user .+ v3 auth .+ priv aes 128 .+'
]

input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\Mistype"
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

def mistype_one_aaa_feature(config_text, seed):
    """
    Finds all lines matching any AAA feature pattern,
    picks one at random (with the given seed),
    and introduces a typo in that line.
    Returns the modified config text, the original line, and the mistyped line.
    """
    lines = config_text.splitlines()
    # Find all AAA feature lines
    matches = [
        (i, line) for i, line in enumerate(lines)
        if any(re.match(pat, line) for pat in aaa_feature_patterns)
    ]
    if not matches:
        return config_text, None, None

    random.seed(seed)
    idx, original_line = random.choice(matches)
    typo_line = introduce_typo(original_line, seed)

    # Replace only that line
    lines[idx] = typo_line
    return "\n".join(lines), original_line, typo_line

# Process router6 through router10
for i in range(1, 16):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_typo_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ File not found: {input_path}")
        continue

    with open(input_path, 'r') as f:
        original = f.read()

    modified, orig_line, typo_line = mistype_one_aaa_feature(original, seed=i)

    with open(output_path, 'w') as f:
        f.write(modified)

    if orig_line:
        print(f"[Router {i}] ✅ Introduced typo:")
        print(f"    Original: {orig_line}")
        print(f"    Mistyped: {typo_line}")
    else:
        print(f"[Router {i}] ⚠️ No AAA feature lines found to mistype.")
