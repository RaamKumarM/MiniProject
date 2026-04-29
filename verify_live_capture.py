import requests
import json
import time

def test_live_capture():
    """Simple verification script for the live packet capture SSE endpoint."""
    url = "http://127.0.0.1:5000/live-packets"
    print(f"[*] Testing live capture at {url}")
    print("[*] Note: Ensure app.py is running and a model is trained.")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        print("[*] Connected to stream. Waiting for window results...")
        
        count = 0
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data = json.loads(decoded_line[6:])
                    print(f"\n[Window {count}] {data['timestamp']}")
                    print(f" - Packets: {data['total_packets']}")
                    print(f" - Flows:   {data.get('total_flows', 0)}")
                    print(f" - Status:  {data['status']}")
                    if data.get("suspicious_ips"):
                        print(f" - Suspicious IPs: {', '.join([x['ip'] for x in data['suspicious_ips']])}")
                    
                    count += 1
                    if count >= 3: # Test for 3 windows
                        break
        
        print("\n[*] Test completed successfully.")

    except Exception as e:
        print(f"[!] Test failed: {e}")

if __name__ == "__main__":
    test_live_capture()
