#!/bin/bash

# Function to check if a folder date falls within a date range
# Arguments: Start date, End date, Folder start date, Folder end date
folder_within_range() {
    start_date="$1"
    end_date="$2"
    folder_start_date="$3"
    folder_end_date="$4"

    if (( folder_start_date >= start_date )) && (( folder_end_date <= end_date )); then
        return 0  # Folder date range is within the specified date range
    else
        return 1  # Folder date range is not within the specified date range
    fi
}

# Input start and end dates in timestamp format (YYYYMMDD)
start_date="$1"
end_date="$2"

# Destination directory where relevant folders will be copied
destination_dir="/data/splunk/test1/db"

# Navigate to the directory containing the folders
cd /data/splunk/defaultdb/db || exit

# Get list of folders (assuming they're named with timestamps)
folder_list=$(ls -d db_*)

# Loop through each folder and check if it falls within the date range
for folder_name in $folder_list; do
    # Extract timestamps from folder name
    folder_start_date=$(echo "$folder_name" | awk -F '_' '{print $3}')
    folder_end_date=$(echo "$folder_name" | awk -F '_' '{print $2}')
    
    if folder_within_range "$start_date" "$end_date" "$folder_start_date" "$folder_end_date"; then
        # Copy the folder and its contents to the destination directory
        # cp -r "$folder_name" "$destination_dir"
        echo "Folder $folder_name copied to $destination_dir."
    fi
done
