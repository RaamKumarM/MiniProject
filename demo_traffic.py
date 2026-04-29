import socket
import time
import threading
import random
import requests
import sys

# --- CONFIGURATION ---
def get_local_ip():
    """Attempts to get the primary local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect, just used to find the interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
TARGET_IP = LOCAL_IP  # Target the local machine's IP instead of loopback for better detection
TARGET_PORT = 5000     # Flask app port
DURATION = 600         # Long duration

stop_event = threading.Event()
pause_event = threading.Event()
pause_event.set()

def generate_benign_traffic():
    """Simulates normal user behavior with standard HTTP requests."""
    print(f"[+] Starting Benign Traffic (Target: {TARGET_IP})...")
    end_time = time.time() + DURATION
    
    # Mix of local app and external harmless sites
    test_urls = [f"http://{TARGET_IP}:{TARGET_PORT}/", "http://example.com"] 
    
    while not stop_event.is_set() and time.time() < end_time:
        pause_event.wait()
        try:
            url = random.choice(test_urls)
            # Standard browser-like behavior
            resp = requests.get(url, timeout=2)
            print(f"  [Benign] Success ({resp.status_code}) -> {url}")
            # Real users don't click every millisecond
            time.sleep(random.uniform(2, 5))
        except Exception as e:
            time.sleep(1)

def generate_attack_traffic():
    """Simulates a high-intensity attack with many packets and multiple source IPs."""
    print(f"[!] Starting Intense Attack Simulation (Target: {TARGET_IP})...")
    end_time = time.time() + DURATION
    
    base_ip_parts = LOCAL_IP.split('.')
    
    while not stop_event.is_set() and time.time() < end_time:
        pause_event.wait()
        try:
            # Create connection
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            
            # --- Source IP Spoofing Simulation ---
            # We try to use different IPs in the same subnet
            # Note: This only works if your OS allows binding to these local aliases
            try:
                # Randomize the last octet to simulate different attackers
                fake_src = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{base_ip_parts[2]}.{random.randint(2, 254)}"
                if fake_src != LOCAL_IP:
                    s.bind((fake_src, 0))
            except Exception:
                fake_src = LOCAL_IP # Fallback
            
            s.connect((TARGET_IP, TARGET_PORT))
            
            # --- HIGH VOLUME BEHAVIOR ---
            # To be "Detected" as an attack, we need to deviate from benign patterns
            # We send many small chunks of data in one go
            for _ in range(50):
                if stop_event.is_set(): break
                s.send(b"X" * random.randint(10, 100))
                # No sleep or very tiny sleep to create a "flood" signature
                time.sleep(0.001)
            
            print(f"  [Attack] Flooding from {fake_src}...")
            s.close()
            
            # Brief pause before next wave to not freeze the local network stack
            time.sleep(0.05)
            
        except Exception:
            # If server is overwhelmed (Connection Refused/Timeout), keep trying
            time.sleep(0.1)

def control_thread():
    """Handles keyboard commands for pausing and stopping."""
    while not stop_event.is_set():
        try:
            print("\n" + "-"*30)
            print(" COMMANDS: [s] Stop | [p] Pause/Resume")
            print("-"*30)
            cmd = input("> ").lower().strip()
            if cmd == 's':
                print("[*] Terminating...")
                stop_event.set()
                break
            elif cmd == 'p':
                if pause_event.is_set():
                    pause_event.clear()
                    print("[||] PAUSED")
                else:
                    pause_event.set()
                    print("[>] RESUMED")
        except EOFError:
            break

def run_demo():
    print("\n" + "="*60)
    print("      DDoS GNN Traffic Generator (Enhanced Detection)")
    print("="*60)
    print(f"Targeting: {TARGET_IP}:{TARGET_PORT}")
    print(f"Local IP:  {LOCAL_IP}")
    print("\nHELP:")
    print("1. Ensure 'app.py' is running.")
    print("2. In Dashboard -> Live Detection, set Source to 'NIC Sniffing'.")
    print("3. Ensure 'config.py' LIVE_INTERFACE matches your active network.")
    print("-" * 60)
    
    mode = input("Select mode: [1] Benign, [2] Attack, [3] Mixed (Mixed is best): ")
    
    t_ctrl = threading.Thread(target=control_thread, daemon=True)
    t_ctrl.start()

    threads = []
    if mode in ['1', '3']:
        threads.append(threading.Thread(target=generate_benign_traffic, daemon=True))
    
    if mode in ['2', '3']:
        if mode == '3':
            print("[*] Priming with normal traffic...")
            time.sleep(3)
        threads.append(threading.Thread(target=generate_attack_traffic, daemon=True))

    for t in threads:
        t.start()
    
    try:
        while not stop_event.is_set():
            time.sleep(1)
            if not any(t.is_alive() for t in threads):
                break
    except KeyboardInterrupt:
        stop_event.set()

    print("\n[!] Generator Stopped.")

if __name__ == "__main__":
    run_demo()
