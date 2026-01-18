import socket

# --- CONFIGURATION ---
# If using DART Studio on this same PC, use '127.0.0.1' or 'localhost'
# If using Real Robot, use the Robot's IP (e.g., '172.168.4.1')
SERVER_IP = '172.24.89.89' 
SERVER_PORT = 20002

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print(f"Connecting to {SERVER_IP}:{SERVER_PORT}...")

    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print("Connected! Waiting for robot to send a request...")

        while True:
            # 1. WAIT here until the robot sends data
            # The script effectively pauses here.
            data = client_socket.recv(1024)
            
            if not data:
                print("Connection closed by robot.")
                break
            
            # 2. Print what the robot said
            robot_msg = data.decode('utf-8')
            print(f"\n[Robot Request]: {robot_msg}")

            # 3. NOW prompt the user for input
            user_input = input("Your Reply > ")
            
            if user_input.lower() == 'exit':
                break

            # 4. Send the reply back to the robot
            client_socket.sendall(user_input.encode('utf-8'))
            print("Message sent. Waiting for next request...")

    except ConnectionRefusedError:
        print("Error: Could not connect. Make sure the robot code is running first.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()