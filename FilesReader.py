import os

directory_path = r"C:\Users\sagib\OneDrive\Desktop\green security\LENOVO Yoga 7 15ITL5 82BJ Windows 10\Balanced Plan\No Scan\Measurement 1\graphs"

def read_files_from_directory():
    all_files_and_directories = os.listdir(directory_path)
    all_files = [f for f in all_files_and_directories if os.path.isfile(directory_path+'/'+f)] #Filtering only the files.
    for file in all_files:
        file_path = f"{directory_path}\\{file}"
        with open(file_path, 'rb') as f:
            print(f.name)
            x = f.read()


read_files_from_directory()