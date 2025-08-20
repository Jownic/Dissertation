import random
import re
import os

# List of AAA features to potentially remove
aaa_features = [
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
    r'^snmp-server user .+ v3 auth .+ priv aes 128 .+',
    r'^line con 0',
    r'^line tty \d+(?: \d+)?',
    r'^line aux 0',
    r'^line vty \d+(?: \d+)?'
]

input_dir = r"Z:\Thesis\Generation Scripts&Results\merge"
output_dir = r"Z:\Thesis\ErrorInputScripts\AAAError\3AAA"
os.makedirs(output_dir, exist_ok=True)

def remove_three_aaa_regex_blocks(config_text, seed):
    """
    Removes up to three AAA-related blocks from config_text,
    chosen at random (based on seed) from the patterns that actually exist.
    Returns the modified text and a list of the removed blocks.
    """
    random.seed(seed)
    # Find which patterns are present
    existing = [p for p in aaa_features if re.search(p, config_text, re.MULTILINE)]
    # Pick up to three at random
    to_remove = random.sample(existing, 3) if len(existing) >= 3 else existing
    print(f"[DEBUG] Seed {seed} selected patterns to remove: {to_remove}")
    regexes = [re.compile(p, re.MULTILINE) for p in to_remove]

    lines = config_text.splitlines()
    matched_lines = set()
    removed_blocks = []
    i = 0

    # Walk through blocks
    while i < len(lines) and len(removed_blocks) < len(regexes):
        if i in matched_lines:
            i += 1
            continue

        # Collect a block: start at i, include indented or blank lines
        block_start = i
        block_lines = [lines[i]]
        j = i + 1
        while j < len(lines) and (lines[j].startswith(' ') or lines[j].strip() == ''):
            block_lines.append(lines[j])
            j += 1
        block_text = "\n".join(block_lines)

        # Test each regex; remove block on first match
        for rx in regexes:
            if rx.search(block_text):
                removed_blocks.append(block_text.strip())
                matched_lines.update(range(block_start, j))
                break

        i = j

    # Rebuild the config without the removed lines
    kept = [l for idx, l in enumerate(lines) if idx not in matched_lines]
    return "\n".join(kept).strip(), removed_blocks

# Process router11 through router15
for i in range(11, 16):
    input_file = f"router{i}_merged_config.txt"
    output_file = f"router{i}_broken_config.txt"
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"[Router {i}] ❌ File not found: {input_path}")
        continue

    with open(input_path, 'r') as f:
        original = f.read()

    modified, removed = remove_three_aaa_regex_blocks(original, seed=i)

    with open(output_path, 'w') as f:
        f.write(modified)

    if removed:
        print(f"[Router {i}] ✅ Removed blocks:")
        for entry in removed:
            print(entry)
    else:
        print(f"[Router {i}] ⚠️ No matching AAA entries to remove.")
