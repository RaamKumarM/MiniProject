import requests
import json
import sys
import time

def verify_stream(url="http://127.0.0.1:5000/live-stream"):
    print(f"Connecting to {url}...")
    print("Ensure the Flask app (app.py) is running and a model is trained.\n")
    
    try:
        # We use stream=True to handle the Server-Sent Events (SSE)
        response = requests.get(url, stream=True, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Server returned status {response.status_code}")
            return

        chunk_count = 0
        print(f"{'Time':<10} | {'Src IP':<15} | {'Dst IP':<15} | {'Pred':<8} | {'Conf'}")
        print("-" * 65)

        for line in response.iter_lines():
            if not line:
                continue
            
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                payload_str = decoded_line[6:]
                data = json.loads(payload_str)
                # print(f"DEBUG: {data}") # Debug line
                
                if "error" in data:
                    print(f"Backend Error: {data['error']}")
                    break
                
                # Print a flow row
                pred_label = "Attack" if data.get('prediction') == 1 else "Benign"
                print(f"{data.get('timestamp', 'N/A'):<10} | "
                      f"{data.get('src_ip', 'N/A'):<15} | "
                      f"{data.get('dst_ip', 'N/A'):<15} | "
                      f"{pred_label:<8} | "
                      f"{data.get('confidence', 0):.2%}")
                
                chunk_count += 1
                
                # Stop after 20 flows to verify it's working
                if chunk_count >= 20:
                    print("\nVerification successful: Received 20 flow events from the stream.")
                    break

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is app.py running?")
    except requests.exceptions.Timeout:
        print("Error: Connection timed out. Ensure the stream is active.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check if a specific file should be set for detection first
    # (Optional: call /set-detection-dataset here if needed)
    verify_stream()
