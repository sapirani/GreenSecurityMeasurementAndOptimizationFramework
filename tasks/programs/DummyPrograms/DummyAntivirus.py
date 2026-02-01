import sys

from tasks.utils import read_files


def main():
    if len(sys.argv) != 2:
        raise Exception("Expecting exactly one argument - scan path")

    read_files(sys.argv[1])


if __name__ == '__main__':
    main()
