# import requests
# from time import sleep

# # Define the Splunk URL and login credentials
# url = "https://132.72.81.185:8089/servicesNS/shoueii/splunkawssecuritymon/saved/searches/"
# username = "shoueii"
# password = "sH231294"

# # Define the list of scheduled search names to enable/disable
# searches_to_enable = ['aws_detect_cloudtrail_trail_deleted Clone1', 'aws_detect_cloudwatch_log_group_delete Clone1', 'aws_detect_cloudwatch_log_stream_delete Clone1', 'aws_detect_ec2_instances_run Clone1', 'aws_detect_ec2_vpc_flow_config_deleted Clone1', 'aws_detect_ecr_image_auth_token_get Clone1']
# searches_to_disable = ['aws_detect_iam_user_deleted Clone', 'aws_detect_iam_user_created Clone', 'aws_detect_iam_new_policy_version_assignment Clone', 'aws_detect_iam_group_added_with_user_from_ec2 Clone', 'aws_detect_ecr_new_repo_image_create Clone', 'aws_detect_ec2_ssh_public_key_addition Clone']


# for search_name in searches_to_enable:
#     # Define the headers and payload for the request
#     headers = {"Content-type": "application/json"}
#     payload = {"disabled": 0}

#     # Make the request to update the scheduled search
#     response = requests.post(
#         url + search_name + "/",
#         headers=headers,
#         auth=(username, password),
#         data=payload,
#         verify=False,
#     )
#     sleep(2)
#     # Check the response status code
#     if response.status_code == 200:
#         print("Scheduled search updated successfully")
#     else:
#         print("Error updating scheduled search")
#         print(response.content)
    

# # Loop through the list of searches to disable and disable them
# for search_name in searches_to_disable:
#    # Define the headers and payload for the request
#     headers = {"Content-type": "application/json"}
#     payload = {"disabled": 1}

#     # Make the request to update the scheduled search
#     response = requests.post(
#         url + search_name + "/",
#         headers=headers,
#         auth=(username, password),
#         data=payload,
#         verify=False,
#     )

#     # Check the response status code
#     if response.status_code == 200:
#         print("Scheduled search updated successfully")
#     else:
#         print("Error updating scheduled search")
#         print(response.content)
import sys
import requests
from time import sleep

# Define the Splunk URL and login credentials
url = "https://132.72.81.185:8089/servicesNS/shoueii/splunkawssecuritymon/saved/searches/"
username = "shoueii"
password = "sH231294"

# Define the list of scheduled search names for lite and heavy mode
heavy_searches_list = [
        "aws_detect_cloudtrail_trail_deleted Clone1",
        "aws_detect_cloudwatch_log_group_delete Clone1",
        "aws_detect_cloudwatch_log_stream_delete Clone1",
        "aws_detect_ec2_instances_run Clone1",
        "aws_detect_ec2_vpc_flow_config_deleted Clone1",
        "aws_detect_ecr_image_auth_token_get Clone1"
    ]
lite_searches_list = [
        "aws_detect_iam_user_deleted Clone",
        "aws_detect_iam_user_created Clone",
        "aws_detect_iam_new_policy_version_assignment Clone",
        "aws_detect_iam_group_added_with_user_from_ec2 Clone",
        "aws_detect_ecr_new_repo_image_create Clone",
        "aws_detect_ec2_ssh_public_key_addition Clone"
    ]


# Read the command line argument
mode = sys.argv[1] if len(sys.argv) > 1 else None

# Select the appropriate searches based on the mode
if mode == "lite":
    searches_list = {
        0: lite_searches_list,
        1: heavy_searches_list
    }



elif mode == "heavy":
    searches_list = {
        0: heavy_searches_list,
        1: lite_searches_list
    }
else:
    print("Invalid mode specified. Please specify either 'lite' or 'heavy'.")
    sys.exit(1)
for i in searches_list:
    # Loop through the list of searches to enable and enable them
    for search_name in searches_list[i]:
        # Define the headers and payload for the request
        headers = {"Content-type": "application/json"}
        payload = {"disabled": i}

        # Make the request to update the scheduled search
        response = requests.post(
            url + search_name + "/",
            headers=headers,
            auth=(username, password),
            data=payload,
            verify=False,
        )
        # Check the response status code
        if response.status_code == 200:
            print(f"Enabled scheduled search '{search_name}' successfully")
        else:
            print(f"Error enabling scheduled search '{search_name}'")
            print(response.content)
        

    # Loop through the list of searches to
