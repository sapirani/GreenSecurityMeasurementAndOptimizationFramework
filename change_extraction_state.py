import re
import sys

def change_rule_state(filepath, state):
    with open(filepath, 'r+') as f:
        content = f.read()
        if state == 'disable':
            content_new = re.sub(r'^(EXTRACT-EventCode = .*)$', r'#\1', content, flags=re.MULTILINE)
        elif state == 'enable':
            content_new = re.sub(r'^#(EXTRACT-EventCode = .*)$', r'\1', content, flags=re.MULTILINE)
        else:
            print("Invalid argument. Please use 'enable' or 'disable'.")
            sys.exit(1)

        f.seek(0)
        f.write(content_new)
        f.truncate()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <filepath> <state>")
        sys.exit(1)

    filepath = sys.argv[1]
    state = sys.argv[2]
    change_rule_state(filepath, state)
