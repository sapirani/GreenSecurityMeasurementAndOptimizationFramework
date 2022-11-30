import os
import sys

# directory_path = r"C:\Users\sagib\OneDrive\Desktop\green security\LENOVO Yoga 7 15ITL5 82BJ Windows 10\Power Saver Plan\No Scan\Measurement 1\graphs"


def read_files_from_directory(directory_path):
    import psutil
    p = psutil.Process()
    print(p.io_counters())
    print(psutil.disk_io_counters())
    all_files_and_directories = os.listdir(directory_path)
    all_files = [f for f in all_files_and_directories if os.path.isfile(directory_path+'/'+f)] #Filtering only the files.
    for file in all_files:
        file_path = f"{directory_path}\\{file}"
        with open(file_path, 'rb') as f:
            #print(f.name)
            x = f.read()

    print(p.io_counters())
    print(psutil.disk_io_counters())


def main():
    if len(sys.argv) != 2:
        raise Exception("Expecting exactly one argument - scan path")

    read_files_from_directory(sys.argv[1])


if __name__ == '__main__':
    main()
