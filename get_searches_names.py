import re
import sys

def get_saved_search_names(filepath):
    names = []
    with open(filepath, 'r') as f:
        content = f.read()
        search_names = re.findall(r'^\[(.*?)\]', content, flags=re.MULTILINE)
        for name in search_names:
            names.append(name)
    return names
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: script.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    get_saved_search_names(filepath)