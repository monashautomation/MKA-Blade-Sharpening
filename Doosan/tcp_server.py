import socket

# --- CONFIGURATION ---
# '0.0.0.0' tells the PC to listen on ALL network adapters (Wi-Fi, Ethernet, etc.)
HOST_IP = '0.0.0.0'  
HOST_PORT = 20002

def start_server():
    # 1. Setup the Server Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Allow the port to be reused immediately after closing (prevents "Address already in use" errors)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST_IP, HOST_PORT))
        server_socket.listen(1) # Listen for 1 connection
        print(f"Server listening on Port {HOST_PORT}...")
        print(f"Please connect the Robot to this PC's IP address.")

        # 2. Wait for the Robot to connect
        conn, addr = server_socket.accept()
        print(f"\nConnection established with Robot at {addr}")
        
        # 3. Communication Loop
        while True:
            # A. Ask User for Input
            user_msg = input("\n[You] Enter command to send: ")
            
            if user_msg.lower() == 'exit':
                print("Closing connection...")
                break
            
            # B. Send to Robot
            conn.sendall(user_msg.encode('utf-8'))
            print(" -> Sent. Waiting for reply...")
            
            # C. Wait for Reply (Blocking)
            # The script pauses here until the robot sends data back
            data = conn.recv(1024)
            
            if not data:
                print("Robot disconnected.")
                break
            
            # D. Print Reply
            print(f"[Robot] Reply: {data.decode('utf-8')}")
            
            # Loop restarts, asking for input again...

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()
        server_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    start_server()