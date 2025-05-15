import time
import json
from dotenv import load_dotenv
import requests
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # GPT-J via Hugging Face
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')
# Configuration
MONITORED_FILE = "/var/log/generated_events.log"  # Change as needed
SPLUNK_API_URL = "https://localhost:8089/services/search/jobs"
SPLUNK_USERNAME = os.getenv("SPLUNK_USERNAME")
SPLUNK_PASSWORD = os.getenv("SPLUNK_PASSWORD")
MODEL_NAME = "EleutherAI/gpt-j-6B"  # GPT-J model from Hugging Face

# Initialize GPT-J
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

def generate_log(event_code):
    """Generate an event log based on event code using GPT-J."""
    prompt = f"Generate a log entry for event code {event_code} in text format."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    generated_log = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_log.strip()

def validate_log(log_entry):
    """Validate log format before writing to the file."""
    if "event_code" not in log_entry or "timestamp" not in log_entry:
        return False, "Missing required fields"
    return True, "Valid log"

def write_log_to_file(log_entry):
    """Write the log entry to the monitored file."""
    with open(MONITORED_FILE, "a") as f:
        f.write(log_entry + "\n")

def query_splunk(event_code):
    """Query Splunk to check if the event is indexed and parsed."""
    search_query = f"search event_code={event_code}"
    auth = (SPLUNK_USERNAME, SPLUNK_PASSWORD)
    headers = {"Content-Type": "application/json"}
    data = {"search": search_query, "output_mode": "json"}
    response = requests.post(SPLUNK_API_URL, auth=auth, headers=headers, data=json.dumps(data), verify=False)
    return response.json()

def correct_log(log_entry):
    """Use GPT-J to correct log format if validation fails."""
    prompt = f"Fix the following log entry to match standard format: {log_entry}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    corrected_log = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_log.strip()

def main_loop():
    """Continuously generate logs, validate, correct, and verify indexing in Splunk."""
    event_codes = [4624, 4625, 4634]  # Example event codes, add more as needed
    while True:
        for code in event_codes:
            log_entry = generate_log(code)
            is_valid, reason = validate_log(log_entry)
            
            if not is_valid:
                print(f"Invalid log for event {code}: {reason}. Correcting...")
                log_entry = correct_log(log_entry)
            
            write_log_to_file(log_entry)
            time.sleep(5)  # Give Splunk some time to ingest
            result = query_splunk(code)
            
            if not result.get("results"):
                print(f"Event Code {code} not indexed correctly. Retrying with correction...")
                corrected_log = correct_log(log_entry)
                write_log_to_file(corrected_log)
                time.sleep(5)
                result = query_splunk(code)
            
            print(f"Event Code {code}:", result)
        time.sleep(60)  # Wait before the next cycle

if __name__ == "__main__":
    main_loop()
