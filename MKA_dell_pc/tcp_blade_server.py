# !/usr/bin/env python3
"""
TCP/IP Server Module for Real-Time Blade Detection
Publishes blade detection data to connected clients
"""

import socket
import json
import threading
import time
from datetime import datetime


class BladeDataTCPServer:
    """TCP/IP server for publishing blade detection data"""

    def __init__(self, host='172.24.9.15', port=5000, max_clients=5):
        """
        Initialize TCP server

        Args:
            host: Server IP address (0.0.0.0 = all interfaces)
            port: Server port number
            max_clients: Maximum number of concurrent clients
        """
        self.host = host
        self.port = port
        self.max_clients = max_clients

        self.server_socket = None
        self.clients = []
        self.clients_lock = threading.Lock()

        self.running = False
        self.latest_data = None
        self.data_lock = threading.Lock()

    def start(self):
        """Start the TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_clients)
            self.server_socket.settimeout(1.0)  # Non-blocking accept

            self.running = True

            # Start accept thread
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()

            print(f"✓ TCP Server started on {self.host}:{self.port}")
            print(f"  Waiting for clients... (max {self.max_clients})")

            return True

        except Exception as e:
            print(f"✗ Failed to start TCP server: {e}")
            return False

    def _accept_clients(self):
        """Accept incoming client connections"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()


                with self.clients_lock:
                    if len(self.clients) < self.max_clients:
                        self.clients.append({
                            'socket': client_socket,
                            'address': client_address,
                            'connected': True
                        })

                        print(f"✓ Client {len(self.clients)} connected: {client_address}")

                        # Start thread for this client
                        client_thread = threading.Thread(
                            target=self._handle_client,
                            args=(client_socket, client_address),
                            daemon=True
                        )
                        client_thread.start()
                    else:
                        # Too many clients
                        client_socket.close()
                        print(f"✗ Rejected client (max clients reached): {client_address}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")

    def _handle_client(self, client_socket, client_address):
        """Handle communication with a specific client"""
        try:
            while self.running:
                # Check if we have new data to send
                # with self.data_lock:
                #     if self.latest_data is not None:
                #         data_to_send = self.latest_data.copy()
                #     else:
                #         data_to_send = None

                if self.latest_data is not None:
                    try:
                        # Send data as JSON with newline delimiter
                        # json_data = json.dumps(self.latest_data) + '\n'
                        client_socket.sendall(self.latest_data)
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"✗ Client disconnected: {client_address}")
                        break
                    except Exception as e:
                        print(f"Send error to {client_address}: {e}")
                        break

                time.sleep(0.1)  # Send rate limiting

        finally:
            # Remove client from list
            with self.clients_lock:
                self.clients = [c for c in self.clients if c['address'] != client_address]

            try:
                client_socket.close()
            except:
                pass

    def read_client(self, client_number):
        client_socket = self.clients[client_number]['socket']
        client_address = self.clients[client_number]['address']
        data = client_socket.recv(4096).decode('utf-8')

        if data is not None:
            print(data)
        else:
            print("No data received")

    def publish_data(self, blade_results):
        """
        Publish blade detection data to all connected clients
        Optimized: Only sends closest blade tip with positive Y offset

        Args:
            blade_results: Detection results dictionary
        """
        if not self.running:
            return

        try:
            # Find closest blade tip with positive Y offset
            closest_blade = None
            min_distance = float('inf')

            if blade_results.get('grinding_coordinates'):
                for coord in blade_results['grinding_coordinates']:
                    # Check if Y offset is positive (move_y_mm > 0)
                    if coord['move_y_mm'] > 0:
                        distance = coord['distance_to_grinder_mm']
                        if distance < min_distance:
                            min_distance = distance
                            closest_blade = coord

            # Create minimal data packet
            if closest_blade:
                cmd_byte = (1).to_bytes(1, 'big')
                x_byte = int(float(closest_blade['move_y_mm'])*100).to_bytes(4, 'big', signed=True)
                y_byte = int(float(closest_blade['move_x_mm'] - 2)*100).to_bytes(4, 'big', signed=True)
                data = cmd_byte + x_byte + y_byte
                # data = {
                #     'x': float(closest_blade['move_x_mm']),
                #     'y': float(closest_blade['move_y_mm']),
                #     'd': float(closest_blade['distance_to_grinder_mm']),
                # }
            else:
                # No valid blade tip found
                data = (2).to_bytes(1, 'big')

            # Update latest data
            with self.data_lock:
                self.latest_data = data

        except Exception as e:
            print(f"Publish error: {e}")

    def get_client_count(self):
        """Get number of connected clients"""
        with self.clients_lock:
            return len(self.clients)

    def stop(self):
        """Stop the TCP server"""
        print("Stopping TCP server...")
        self.running = False

        # Close all client connections
        with self.clients_lock:
            for client in self.clients:
                try:
                    client['socket'].close()
                except:
                    pass
            self.clients.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("✓ TCP server stopped")


# Example client code for testing
class BladeDataTCPClient:
    """Simple TCP client for receiving blade data"""

    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def connect(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.running = True
            print(f"✓ Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def receive_data(self, callback=None):
        """
        Receive data from server

        Args:
            callback: Function to call with received data (optional)
        """
        buffer = ""

        while self.running:
            try:
                # Receive data
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    print("✗ Connection closed by server")
                    break

                buffer += data

                # Process complete JSON messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        try:
                            json_data = json.loads(line)

                            if callback:
                                callback(json_data)
                            else:
                                print(f"Received: {json_data}")

                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")

            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break

    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("✓ Disconnected")


if __name__ == "__main__":
    # Test server
    print("Starting test TCP server...")
    server = BladeDataTCPServer(host='172.24.9.15', port=5000)

    if server.start():
        print("\nServer running. Press Ctrl+C to stop.")
        print("Connect with: telnet localhost 5000")
        print("\nSending MINIMAL data: closest blade with Y+ offset only")

        # Simulate publishing data
        try:
            count = 0
            while True:
                # Create test data with multiple blades
                test_data = {
                    'timestamp': datetime.now(),
                    'num_grooves': 3,
                    'grinder_tip': (500, 300),
                    'grinding_coordinates': [
                        {
                            'tooth_id': 1,
                            'groove_position_x_px': 450,
                            'groove_position_y_px': 250,
                            'move_x_mm': 2.5,
                            'move_y_mm': 3.2,  # Positive Y
                            'distance_to_grinder_mm': 15.3,
                        },
                        {
                            'tooth_id': 2,
                            'groove_position_x_px': 470,
                            'groove_position_y_px': 280,
                            'move_x_mm': 1.8,
                            'move_y_mm': 2.5,  # Positive Y - CLOSEST
                            'distance_to_grinder_mm': 12.1,
                        },
                        {
                            'tooth_id': 3,
                            'groove_position_x_px': 490,
                            'groove_position_y_px': 310,
                            'move_x_mm': 0.5,
                            'move_y_mm': -1.0,  # Negative Y - ignored
                            'distance_to_grinder_mm': 10.5,
                        },
                    ]
                }

                server.publish_data(test_data)
                count += 1

                print(f"\rPublished {count} messages | Clients: {server.get_client_count()}", end='')
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            server.stop()