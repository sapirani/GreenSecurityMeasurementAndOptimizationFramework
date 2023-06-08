
import re
import sys
# from crontab import CronTab
# import dateparser
# from configparser import ConfigParser
from  get_searches_names import get_saved_search_names
# sudo -E env PATH="$PATH" python cron_update.py /opt/splunk/etc/users/shouei/search/local/savedsearches.conf "*/10 * * * *"

def update_saved_search_schedule(filepath, saved_search_name, new_schedule):
    with open(filepath, 'r') as f:
            contents = f.read()

    # Construct a regular expression to match the cron schedule line for the saved search
    pattern = r'(\[' + re.escape(saved_search_name) + r'\][^\[]*?)cron_schedule = [^\n]*'

    # Replace the cron schedule line with the new schedule
    new_contents = re.sub(pattern, r'\1cron_schedule = ' + new_schedule, contents, flags=re.DOTALL)

    # Write the modified contents back to the file
    with open(filepath, 'w') as f:
        f.write(new_contents)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: cron_update.py <filepath> <new_schedule>")
        sys.exit(1)

    filepath = sys.argv[1]
    new_schedule = sys.argv[2]
    searches_names = get_saved_search_names(filepath)
    for search_name in searches_names:
        update_saved_search_schedule(filepath, search_name, new_schedule)
