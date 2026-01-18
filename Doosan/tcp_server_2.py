import socket
import struct  # Library to pack numbers into bytes

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
HOST_PORT = 20002

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST_IP, HOST_PORT))
        server_socket.listen(1)
        print(f"Server listening on Port {HOST_PORT}...")
        print("Waiting for Robot to connect...")

        conn, addr = server_socket.accept()
        print(f"\nConnection established with Robot at {addr}")
        
        while True:
            try:

                user_input_x = input("Enter X Coordinates: ")
                if user_input_x.lower() == 'exit': break
                
                user_input_y = input("Enter Y Coordinates: ")
                
                val_x = int(float(user_input_x) * 100)
                val_y = int(float(user_input_y) * 100)

                cmd_byte = (1).to_bytes(1, 'big')
                x_bytes  = val_x.to_bytes(4, 'big', signed=True)
                y_bytes  = val_y.to_bytes(4, 'big', signed=True)
                
                packet = cmd_byte + x_bytes + y_bytes
                conn.sendall(packet)
                print(f"Sent: Cmd={cmd_byte}, X={val_x}, Y={val_y}")

                robot_reply = conn.recv(1024)
                if robot_reply: print(f"Robot: {robot_reply.decode()}")

            except ValueError:
                print("Error: Please enter valid numbers.")
            except Exception as e:
                print(f"Connection Error: {e}")
                break

    finally:
        if 'conn' in locals(): conn.close()
        server_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    start_server()