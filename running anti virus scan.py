import subprocess


def run(cmd):
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    return completed


if __name__ == '__main__':
    hello_command = "Start-MpScan -ScanType QuickScan"
    hello_info = run(hello_command)
    if hello_info.returncode != 0:
        print("An error occured: %s", hello_info.stderr)
    else:
        print("Hello command executed successfully!")
    
    print("-------------------------")
    
  