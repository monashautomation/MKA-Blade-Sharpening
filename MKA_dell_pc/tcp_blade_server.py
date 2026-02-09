"""
Fixed TCP/IP Server for Tooth Point Detection
Ensures clean binary data packets with no corruption
Compatible with both tooth point and groove detection
"""

import socket
import threading
import time
from datetime import datetime


class BladeDataTCPServer:
    """TCP/IP server for publishing blade/tooth detection data"""

    def __init__(self, host='172.24.9.15', port=5000, max_clients=5):
        """
        Initialize TCP server

        Args:
            host: Server IP address
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

        # Track last sent data to avoid duplicates
        self.last_sent_data = {}

    def start(self):
        """Start the TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Disable Nagle's algorithm for immediate sending
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_clients)
            self.server_socket.settimeout(1.0)

            self.running = True

            # Start accept thread
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()

            print(f"✓ TCP Server started on {self.host}:{self.port}")
            return True

        except Exception as e:
            print(f"✗ Failed to start TCP server: {e}")
            return False

    def _accept_clients(self):
        """Accept incoming client connections"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()

                # Configure client socket
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)

                with self.clients_lock:
                    if len(self.clients) < self.max_clients:
                        client_id = f"{client_address[0]}:{client_address[1]}"

                        self.clients.append({
                            'socket': client_socket,
                            'address': client_address,
                            'id': client_id,
                            'connected': True
                        })

                        print(f"✓ Client connected: {client_address}")

                        # Start thread for this client
                        client_thread = threading.Thread(
                            target=self._handle_client,
                            args=(client_socket, client_address, client_id),
                            daemon=True
                        )
                        client_thread.start()
                    else:
                        client_socket.close()
                        print(f"✗ Rejected client (max reached): {client_address}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")

    def _handle_client(self, client_socket, client_address, client_id):
        """Handle communication with a specific client"""
        try:
            while self.running:
                # Get latest data
                with self.data_lock:
                    data_to_send = self.latest_data

                if data_to_send is not None:
                    # Check if this is new data for this client
                    if client_id not in self.last_sent_data or self.last_sent_data[client_id] != data_to_send:
                        try:
                            # Send exactly the binary data packet
                            client_socket.sendall(data_to_send)

                            # Track what we sent
                            self.last_sent_data[client_id] = data_to_send

                        except (BrokenPipeError, ConnectionResetError, OSError):
                            print(f"✗ Client disconnected: {client_address}")
                            break
                        except Exception as e:
                            print(f"Send error to {client_address}: {e}")
                            break

                time.sleep(0.05)  # 20Hz send rate

        finally:
            # Remove client from list
            with self.clients_lock:
                self.clients = [c for c in self.clients if c['id'] != client_id]

            # Clean up tracking
            if client_id in self.last_sent_data:
                del self.last_sent_data[client_id]

            try:
                client_socket.close()
            except:
                pass

    def read_client(self, client_number):
        """Read data from a specific client"""
        try:
            with self.clients_lock:
                if client_number >= len(self.clients):
                    return None

                client_socket = self.clients[client_number]['socket']
                client_address = self.clients[client_number]['address']

            # Set non-blocking
            client_socket.setblocking(False)

            try:
                data = client_socket.recv(4096)
                if data:
                    print(f"Received from {client_address}: {data}")
                    return data
            except BlockingIOError:
                # No data available
                pass
            finally:
                # Restore blocking
                client_socket.setblocking(True)

        except Exception as e:
            print(f"Read error: {e}")

        return None

    def publish_data(self, blade_results):
        """
        Publish blade/tooth detection data to all connected clients
        Sends binary packet: [cmd(1)] + [x(4)] + [y(4)]

        Sends the MIDDLE POINT (valley/groove) between teeth that is:
        - Above the grinder (positive Y offset)
        - Closest to the grinder

        Args:
            blade_results: Detection results dictionary
        """
        if not self.running:
            return

        try:
            # Find tooth profiles to calculate middle points (valleys)
            teeth_profiles = blade_results.get('teeth_profiles', [])
            # teeth_profiles = blade_results.get('grinding_coordinates',[])
            grinder_tip = blade_results.get('grinder_tip')
            pixels_per_mm = 86.96  # Default calibration

            closest_valley = None
            min_distance = float('inf')

            if teeth_profiles and grinder_tip:
                for i in range(len(teeth_profiles) - 1):
                    current_tooth = teeth_profiles[i]
                    next_tooth = teeth_profiles[i + 1]

                    # Calculate middle point between current tooth tip and next tooth tip
                    # This is the valley/groove between the teeth
                    valley_x = (current_tooth.grinding_point[0] + next_tooth.grinding_point[0]) / 2
                    valley_y = (current_tooth.grinding_point[1] + next_tooth.grinding_point[1]) / 2
                    # Calculate movement to grinder from valley
                    move_x_px = grinder_tip[0] - valley_x
                    move_y_px = grinder_tip[1] - valley_y
                    #
                    move_x_mm = move_x_px / pixels_per_mm
                    move_y_mm = move_y_px / pixels_per_mm

                    # Check if valley is above grinder (positive Y offset)
                    if move_y_mm > 1:
                        distance_mm = ((move_x_mm ** 2 + move_y_mm ** 2) ** 0.5)

                        if distance_mm < min_distance:
                            min_distance = distance_mm
                            closest_valley = {
                                'valley_x': valley_x,
                                'valley_y': valley_y,
                                'move_x_mm': move_x_mm,
                                'move_y_mm': move_y_mm
                                # 'between_teeth': f"{current_tooth.tooth_id}-{next_tooth.tooth_id}"
                            }

            # Create binary data packet
            if closest_valley:
                # CMD = 1: Valley (middle point) detected
                cmd_byte = (1).to_bytes(1, 'big')

                # X coordinate (convert mm to 0.01mm units, 4 bytes signed)
                x_value = int(float(closest_valley['move_y_mm']) * 100)
                x_bytes = x_value.to_bytes(4, 'big', signed=True)

                # Y coordinate (apply offset and convert)
                y_value = int((float(closest_valley['move_x_mm'])) * 100)
                y_bytes = y_value.to_bytes(4, 'big', signed=True)

                # Combine into single packet (9 bytes total)
                data = cmd_byte + x_bytes + y_bytes

                # Optional: print debug info
                # print(f"Valley between teeth {closest_valley['between_teeth']}: ({closest_valley['move_x_mm']:.2f}, {closest_valley['move_y_mm']:.2f})mm")
            else:
                # CMD = 2: No valid valley detected
                data = (2).to_bytes(1, 'big')

            # Update latest data (atomic operation)
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
        time.sleep(0.2)

        # Close all client connections
        with self.clients_lock:
            for client in self.clients:
                try:
                    client['socket'].close()
                except:
                    pass
            self.clients.clear()

        # Clear tracking
        self.last_sent_data.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("✓ TCP server stopped")


# Test client
class BladeDataTCPClient:
    """Simple TCP client for receiving blade/tooth data"""

    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def connect(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.host, self.port))
            self.running = True
            print(f"✓ Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def receive_data(self, callback=None):
        """
        Receive binary data from server

        Args:
            callback: Function to call with received data (optional)
        """
        while self.running:
            try:
                # Read command byte first
                cmd_data = self.socket.recv(1)
                if not cmd_data:
                    print("✗ Connection closed by server")
                    break

                cmd = int.from_bytes(cmd_data, 'big')

                if cmd == 1:
                    # Tooth/blade detected - read 8 more bytes (x and y)
                    coord_data = self.socket.recv(8)
                    if len(coord_data) != 8:
                        print("✗ Incomplete data packet")
                        continue

                    x_value = int.from_bytes(coord_data[0:4], 'big', signed=True) / 100.0
                    y_value = int.from_bytes(coord_data[4:8], 'big', signed=True) / 100.0

                    data = {
                        'cmd': 1,
                        'x_mm': x_value,
                        'y_mm': y_value
                    }

                    if callback:
                        callback(data)
                    else:
                        print(f"Tooth: X={x_value:.2f}mm, Y={y_value:.2f}mm")

                elif cmd == 2:
                    # No tooth/blade detected
                    data = {'cmd': 2, 'status': 'no_detection'}

                    if callback:
                        callback(data)
                    else:
                        print("No tooth detected")

                else:
                    print(f"Unknown command: {cmd}")

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
        print("Binary protocol: [CMD(1)] + [X(4)] + [Y(4)]")
        print("Detects closest tooth point with positive Y offset")

        try:
            count = 0
            while True:
                # Create test data - tooth points
                test_data = {
                    'timestamp': datetime.now(),
                    'num_teeth': 3,
                    'num_grooves': 3,  # Backward compatibility
                    'grinder_tip': (500, 300),
                    'grinding_coordinates': [
                        {
                            'tooth_id': 1,
                            'move_x_mm': 2.5,
                            'move_y_mm': 3.2,
                            'distance_to_grinder_mm': 15.3,
                        },
                        {
                            'tooth_id': 2,
                            'move_x_mm': 1.8,
                            'move_y_mm': 2.5,  # CLOSEST with positive Y
                            'distance_to_grinder_mm': 12.1,
                        },
                        {
                            'tooth_id': 3,
                            'move_x_mm': 0.5,
                            'move_y_mm': -1.0,  # Negative Y - ignored
                            'distance_to_grinder_mm': 10.5,
                        },
                    ]
                }

                server.publish_data(test_data)
                count += 1

                print(f"\rPublished {count} packets | Clients: {server.get_client_count()}", end='')
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            server.stop()