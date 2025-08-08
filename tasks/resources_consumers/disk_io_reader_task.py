import argparse

from tasks.utils import read_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only reads files in Disk"
    )

    parser.add_argument("-d", "--directory",
                        type=str,
                        required=True,
                        help="The path to the directory to read its content.")

    args = parser.parse_args()
    read_files(args.directory)
