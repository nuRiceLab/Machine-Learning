import argparse
import requests
import json
import os
from datetime import datetime
from collections import Counter

def fetch_justIN(workflow_id: int, event_type: str):
    
    url = (
    )
    
    print(f"[INFO] Fetching events from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Fetch justIN event data and calculate event rate.")
    parser.add_argument('--workflow', type=int, required=True, help="Workflow ID from DUNE dashboard.")
    parser.add_argument('--output', type=str, default="test", help="Output file.")
    args = parser.parse_args()
    # Build JSON path with same basename as output (replace extension with .json)
    base, _ = os.path.splitext(args.output)
    json_path = base + ".json"

    event_types = ["JOB_SUBMITTED", "JOB_STARTED", "FILE_ALLOCATED", "JOB_PROCESSING",
                   "FILE_PROCESSED", "JOB_OUTPUTTING", "FILE_CREATED", "JOB_FINISHED"] 


    all_payloads = {}   # raw responses keyed by event type
    all_events = []     # flattened events across types

    for et in event_types:
        data = fetch_justIN(args.workflow, et)
        all_payloads[et] = data

        # Normalize events into a list
        events = data if isinstance(data, list) else data.get("data", [])
        if isinstance(events, dict):  # just in case a single object is returned
            events = [events]
        all_events.extend(events)

    if not all_events:
        print("[!] No events found.")
        # Still save what we got so you can inspect payloads per type
        with open(json_path, "w") as jf:
            json.dump(all_payloads, jf, indent=2, ensure_ascii=False)
        print(f"[✓] JSON (empty events) saved to {json_path}")
        return

    # Save raw JSON payloads per event type
    with open(json_path, "w") as jf:
        json.dump(all_payloads, jf, indent=2, ensure_ascii=False)
    print(f"[✓] JSON saved to {json_path}")

if __name__ == "__main__":
    main()

