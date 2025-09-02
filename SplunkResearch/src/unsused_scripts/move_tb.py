import os
import shutil
from pathlib import Path
import re
from datetime import datetime, time

def copy_files_by_datetime(source_dir, destination_dir, date_pattern=None, start_datetime=None, end_datetime=None):
    """
    Copy single files from subdirectories based on date and time filtering.
    
    Args:
        source_dir (str): Path to the source directory containing subdirectories
        destination_dir (str): Path to the destination directory
        date_pattern (str): Regex pattern to extract datetime from subdir name
        start_datetime (str): Start datetime in YYYYMMDDHHMMSS format (inclusive)
        end_datetime (str): End datetime in YYYYMMDDHHMMSS format (inclusive)
    """
    
    # Convert to Path objects
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Default datetime pattern (looks for YYYYMMDDHHMMSS format)
    if date_pattern is None:
        date_pattern = r'(\d{14})'  # Extract full YYYYMMDDHHMMSS
    
    # Convert datetime strings to datetime objects
    start_dt = datetime.strptime(start_datetime, '%Y%m%d%H%M%S') if start_datetime else None
    end_dt = datetime.strptime(end_datetime, '%Y%m%d%H%M%S') if end_datetime else None
    
    copied_count = 0
    processed_dirs = 0
    
    # Iterate through subdirectories
    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Extract datetime from directory name
        datetime_match = re.search(date_pattern, subdir.name)
        if not datetime_match:
            print(f"Skipping {subdir.name} - no datetime found")
            continue
            
        # Parse the datetime
        try:
            dir_datetime = datetime.strptime(datetime_match.group(1), '%Y%m%d%H%M%S')
        except ValueError:
            print(f"Skipping {subdir.name} - invalid datetime format")
            continue
            
        # Check datetime range
        if start_dt and dir_datetime < start_dt:
            print(f"Skipping {subdir.name} - before start datetime")
            continue
        if end_dt and dir_datetime > end_dt:
            print(f"Skipping {subdir.name} - after end datetime")
            continue
            
        processed_dirs += 1
        print(f"Processing directory: {subdir.name}")
        
        # Find the single file in this subdirectory (ignoring subsubdirs)
        files = [f for f in subdir.iterdir() if f.is_file()]
        
        if len(files) == 0:
            print(f"  No files found in {subdir.name}")
            continue
        elif len(files) > 1:
            print(f"  Multiple files found in {subdir.name}, taking the first one: {files[0].name}")
        
        # Take the first (or only) file
        file_to_copy = files[0]
        
        # Create subdirectory with same name as original
        dest_subdir = dest_path / subdir.name
        dest_subdir.mkdir(exist_ok=True)
        dest_file = dest_subdir / file_to_copy.name
        
        # Handle conflicts by adding counter
        counter = 1
        while dest_file.exists():
            stem = file_to_copy.stem
            suffix = file_to_copy.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_file = dest_subdir / dest_filename
            counter += 1
        
        try:
            shutil.copy2(file_to_copy, dest_file)
            print(f"  Copied: {file_to_copy.name} -> {subdir.name}/{dest_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_to_copy}: {e}")
    
    print(f"\nSummary:")
    print(f"Processed directories: {processed_dirs}")
    print(f"Total files copied: {copied_count}")

def copy_files_by_date(source_dir, destination_dir, date_pattern=None, start_date=None, end_date=None):
    """
    Copy files from subdirectories based on date filtering (ignoring time).
    
    Args:
        source_dir (str): Path to the source directory
        destination_dir (str): Path to the destination directory
        start_date (str): Start date in YYYYMMDD format (inclusive)
        end_date (str): End date in YYYYMMDD format (inclusive)
        date_pattern (str): Regex pattern to extract datetime from subdir name
    """
    
    # Convert date range to datetime range (full day)
    start_datetime = f"{start_date}000000" if start_date else None
    end_datetime = f"{end_date}235959" if end_date else None
    
    copy_files_by_datetime(source_dir, destination_dir, date_pattern, start_datetime, end_datetime)

def copy_files_by_time_range(source_dir, destination_dir, start_time, end_time, date_pattern=None):
    """
    Copy files from subdirectories based on time range (any date).
    
    Args:
        source_dir (str): Path to the source directory
        destination_dir (str): Path to the destination directory
        start_time (str): Start time in HHMMSS format (e.g., "090000" for 9:00 AM)
        end_time (str): End time in HHMMSS format (e.g., "170000" for 5:00 PM)
        date_pattern (str): Regex pattern to extract datetime from subdir name
    """
    
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if date_pattern is None:
        date_pattern = r'(\d{14})'
    
    # Convert time strings to time objects
    start_t = datetime.strptime(start_time, '%H%M%S').time()
    end_t = datetime.strptime(end_time, '%H%M%S').time()
    
    copied_count = 0
    
    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Extract datetime from directory name
        datetime_match = re.search(date_pattern, subdir.name)
        if not datetime_match:
            continue
            
        try:
            dir_datetime = datetime.strptime(datetime_match.group(1), '%Y%m%d%H%M%S')
            dir_time = dir_datetime.time()
        except ValueError:
            continue
            
        # Check time range
        if start_t <= end_t:  # Same day range
            if not (start_t <= dir_time <= end_t):
                continue
        else:  # Overnight range (e.g., 22:00 to 06:00)
            if not (dir_time >= start_t or dir_time <= end_t):
                continue
                
        print(f"Processing directory: {subdir.name}")
        
        # Find the single file
        files = [f for f in subdir.iterdir() if f.is_file()]
        if len(files) == 0:
            continue
            
        file_to_copy = files[0]
        
        # Create subdirectory with same name as original
        dest_subdir = dest_path / subdir.name
        dest_subdir.mkdir(exist_ok=True)
        dest_file = dest_subdir / file_to_copy.name
        
        # Handle conflicts
        counter = 1
        while dest_file.exists():
            stem = file_to_copy.stem
            suffix = file_to_copy.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_file = dest_subdir / dest_filename
            counter += 1
        
        try:
            shutil.copy2(file_to_copy, dest_file)
            print(f"  Copied: {file_to_copy.name} -> {subdir.name}/{dest_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_to_copy}: {e}")
    
    print(f"\nTotal files copied: {copied_count}")

def copy_files_by_datetime_list(source_dir, destination_dir, datetime_list, date_pattern=None):
    """
    Copy files from subdirectories matching specific datetimes.
    
    Args:
        source_dir (str): Path to the source directory
        destination_dir (str): Path to the destination directory
        datetime_list (list): List of datetimes in YYYYMMDDHHMMSS format
        date_pattern (str): Regex pattern to extract datetime from subdir name
    """
    
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if date_pattern is None:
        date_pattern = r'(\d{14})'
    
    copied_count = 0
    
    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Extract datetime from directory name
        datetime_match = re.search(date_pattern, subdir.name)
        if not datetime_match:
            continue
            
        # Check if this datetime is in our target list
        dir_datetime = datetime_match.group(1)
        if dir_datetime not in datetime_list:
            continue
            
        print(f"Processing directory: {subdir.name}")
        
        # Find the single file
        files = [f for f in subdir.iterdir() if f.is_file()]
        if len(files) == 0:
            continue
            
        file_to_copy = files[0]
        
        # Create subdirectory with same name as original
        dest_subdir = dest_path / subdir.name
        dest_subdir.mkdir(exist_ok=True)
        dest_file = dest_subdir / file_to_copy.name
        
        # Handle conflicts
        counter = 1
        while dest_file.exists():
            stem = file_to_copy.stem
            suffix = file_to_copy.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_file = dest_subdir / dest_filename
            counter += 1
        
        try:
            shutil.copy2(file_to_copy, dest_file)
            print(f"  Copied: {file_to_copy.name} -> {subdir.name}/{dest_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_to_copy}: {e}")
    
    print(f"\nTotal files copied: {copied_count}")

def copy_files_by_prefix(source_dir, destination_dir, prefixes, date_pattern=None):
    """
    Copy files from subdirectories matching specific prefixes.
    
    Args:
        source_dir (str): Path to the source directory
        destination_dir (str): Path to the destination directory
        prefixes (list): List of prefixes to match (e.g., ['eval_post_training', 'test_experiment'])
        date_pattern (str): Regex pattern to extract datetime from subdir name
    """
    
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if date_pattern is None:
        date_pattern = r'(\d{14})'
    
    copied_count = 0
    
    for subdir in source_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Check if directory starts with any of the specified prefixes
        if not any(subdir.name.startswith(prefix) for prefix in prefixes):
            continue
            
        print(f"Processing directory: {subdir.name}")
        
        # Find the single file
        files = [f for f in subdir.iterdir() if f.is_file()]
        if len(files) == 0:
            continue
            
        file_to_copy = files[0]
        
        # Create subdirectory with same name as original
        dest_subdir = dest_path / subdir.name
        dest_subdir.mkdir(exist_ok=True)
        dest_file = dest_subdir / file_to_copy.name
        
        # Handle conflicts
        counter = 1
        while dest_file.exists():
            stem = file_to_copy.stem
            suffix = file_to_copy.suffix
            dest_filename = f"{stem}_{counter}{suffix}"
            dest_file = dest_subdir / dest_filename
            counter += 1
        
        try:
            shutil.copy2(file_to_copy, dest_file)
            print(f"  Copied: {file_to_copy.name} -> {subdir.name}/{dest_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_to_copy}: {e}")
    
    print(f"\nTotal files copied: {copied_count}")

# Example usage
if __name__ == "__main__":
    SOURCE_DIR = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/tensorboard"
    DEST_DIR = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/tensorboard/copied_files"
    
    # Method 1: Copy files by exact datetime range
    print("Method 1: DateTime Range")
    copy_files_by_datetime(
        SOURCE_DIR, 
        DEST_DIR,
        start_datetime='20250710205650',  # July 3rd, 9:00 AM
        end_datetime='20250710215650'     # July 8th, 5:00 PM
    )
    