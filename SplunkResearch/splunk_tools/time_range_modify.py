import re
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from  splunk_tools.get_searches_names import get_saved_search_names

            
def change_search_time_range(filepath, search_name, earliest_time, latest_time):
    with open(filepath, 'r+') as f:
        content = f.read()
        content_new = re.sub(
            r'^(\[' + re.escape(search_name) + r'\].*?)^(dispatch\.earliest_time = .*?$.*?^dispatch\.latest_time = .*?$)',
            r'\1dispatch.earliest_time = ' + earliest_time + '\ndispatch.latest_time = ' + latest_time,
            content,
            flags=re.MULTILINE | re.DOTALL
        )

        f.seek(0)
        f.write(content_new)
        f.truncate()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(sys.argv)
        print("Usage: win_time_modify.py <filepath> <earliest_time> <latest_time>")
        sys.exit(1)

    filepath = sys.argv[1]
    earliest_time = sys.argv[2]
    latest_time = sys.argv[3]
    searches_names = get_saved_search_names(filepath)
    for search_name in searches_names:
        print(search_name)
        change_search_time_range(filepath, search_name, earliest_time, latest_time)
