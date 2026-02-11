"""
Restructured TCP/IP Server for Blade Detection
Protocol:
1. Server sends configuration packet first
2. Server waits for client request (CMD=3)
3. Server responds with compact detection data
"""

import socket
import threading
import time
import struct
from datetime import datetime


class BladeDataTCPServerV2:
    """TCP/IP server with request-response protocol"""

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
        
        # Configuration data (set by user before starting)
        self.config_data = None
        self.config_sent = {}  # Track which clients received config

    def set_configuration(self, bay_id, grinder_id, angle, depth, length):
        """
        Set configuration data to send to clients
        
        Args:
            bay_id: Bay ID (1-10)
            grinder_id: Grinder ID (1-3)
            angle: Angle in degrees (00.0 format)
            depth: Depth (0.00 format)
            length: Length (000 format)
        """
        # Validate inputs
        if not (1 <= bay_id <= 10):
            raise ValueError("BAY ID must be 1-10")
        if not (1 <= grinder_id <= 3):
            raise ValueError("GRINDER ID must be 1-3")
        
        # Create compact binary packet
        # Format: [CMD=0][BAY_ID][GRINDER_ID][ANGLE*10][DEPTH*100][LENGTH]
        # Total: 1 + 1 + 1 + 2 + 2 + 2 = 9 bytes
        
        cmd_byte = (0).to_bytes(1, 'big')  # CMD=0: Configuration
        bay_byte = bay_id.to_bytes(1, 'big')
        grinder_byte = grinder_id.to_bytes(1, 'big')
        
        # Angle: multiply by 10 to get 1 decimal place (e.g., 45.5° -> 455)
        angle_value = int(angle * 10)
        angle_bytes = angle_value.to_bytes(2, 'big', signed=True)
        
        # Depth: multiply by 100 to get 2 decimal places (e.g., 1.25 -> 125)
        depth_value = int(depth * 100)
        depth_bytes = depth_value.to_bytes(2, 'big', signed=True)
        
        # Length: integer value (0-999)
        length_value = int(length)
        length_bytes = length_value.to_bytes(2, 'big', signed=False)
        
        self.config_data = cmd_byte + bay_byte + grinder_byte + angle_bytes + depth_bytes + length_bytes
        
        print(f"✓ Configuration set: Bay={bay_id}, Grinder={grinder_id}, " +
              f"Angle={angle:.1f}°, Depth={depth:.2f}, Length={length}")

    def broadcast_new_configuration(self):
        """
        Broadcast new configuration to all connected clients
        This allows updating configuration mid-operation
        """
        if self.config_data is None:
            print("⚠ No configuration data to broadcast")
            return
        
        broadcast_count = 0
        failed_count = 0
        
        with self.clients_lock:
            for client in self.clients:
                try:
                    client['socket'].sendall(self.config_data)
                    # Mark that this client has received new config
                    self.config_sent[client['id']] = True
                    broadcast_count += 1
                except Exception as e:
                    print(f"✗ Failed to send config to {client['address']}: {e}")
                    failed_count += 1
        
        if broadcast_count > 0:
            print(f"✓ Configuration broadcasted to {broadcast_count} client(s)")
        if failed_count > 0:
            print(f"⚠ Failed to send to {failed_count} client(s)")

    def start(self):
        """Start the TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512)

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
        """
        Handle communication with a specific client
        Protocol:
        1. Send configuration packet immediately
        2. Wait for client request (CMD=3)
        3. Respond with detection data
        """
        try:
            # Step 1: Send configuration packet first
            if self.config_data is not None:
                try:
                    client_socket.sendall(self.config_data)
                    self.config_sent[client_id] = True
                    print(f"→ Sent configuration to {client_address}")
                except Exception as e:
                    print(f"✗ Failed to send config to {client_address}: {e}")
                    return
            else:
                print(f"⚠ No configuration data set for {client_address}")
                return

            # Step 2 & 3: Wait for requests and respond
            while self.running:
                try:
                    # Set a short timeout for recv
                    client_socket.settimeout(0.1)
                    
                    # Try to receive request from client
                    try:
                        request = client_socket.recv(1)
                        
                        if not request:
                            # Client disconnected
                            print(f"✗ Client disconnected: {client_address}")
                            break
                        
                        cmd = int.from_bytes(request, 'big')
                        
                        if cmd == 3:  # CMD=3: Request detection data
                            # Send latest detection data
                            with self.data_lock:
                                data_to_send = self.latest_data
                            
                            if data_to_send is not None:
                                client_socket.sendall(data_to_send)
                                # Optional: print for debugging
                                # print(f"→ Sent detection data to {client_address}")
                            else:
                                # No data available, send CMD=2 (no detection)
                                no_data = (2).to_bytes(1, 'big')
                                client_socket.sendall(no_data)
                        
                    except socket.timeout:
                        # No request received, continue waiting
                        continue
                    
                except (BrokenPipeError, ConnectionResetError, OSError):
                    print(f"✗ Client disconnected: {client_address}")
                    break
                except Exception as e:
                    print(f"Communication error with {client_address}: {e}")
                    break

        finally:
            # Remove client from list
            with self.clients_lock:
                self.clients = [c for c in self.clients if c['id'] != client_id]

            # Clean up tracking
            if client_id in self.config_sent:
                del self.config_sent[client_id]

            try:
                client_socket.close()
            except:
                pass

    def publish_data(self, blade_results):
        """
        Update detection data (called by detection thread)
        Creates COMPACT binary packet: [CMD=1][X(2)][Y(2)] = 5 bytes total
        
        X and Y are in 0.1mm units (signed 16-bit integers)
        Range: -3276.8mm to +3276.7mm (sufficient for most applications)

        Args:
            blade_results: Detection results dictionary
        """
        if not self.running:
            return

        try:
            teeth_profiles = blade_results.get('teeth_profiles', [])
            grinder_tip = blade_results.get('grinder_tip')
            pixels_per_mm = 86.96  # Default calibration

            closest_valley = None
            min_distance = float('inf')

            if teeth_profiles and grinder_tip:
                for i in range(len(teeth_profiles) - 1):
                    current_tooth = teeth_profiles[i]
                    next_tooth = teeth_profiles[i + 1]

                    # Calculate valley (middle point between teeth)
                    valley_x = (current_tooth.grinding_point[0] + next_tooth.grinding_point[0]) / 2
                    valley_y = (current_tooth.grinding_point[1] + next_tooth.grinding_point[1]) / 2
                    
                    # Calculate movement to grinder from valley
                    move_x_px = grinder_tip[0] - valley_x
                    move_y_px = grinder_tip[1] - valley_y
                    
                    move_x_mm = move_x_px / pixels_per_mm
                    move_y_mm = move_y_px / pixels_per_mm

                    # Check if valley is above grinder (positive Y offset)
                    if move_y_mm > 1:
                        distance_mm = ((move_x_mm ** 2 + move_y_mm ** 2) ** 0.5)

                        if distance_mm < min_distance:
                            min_distance = distance_mm
                            closest_valley = {
                                'move_x_mm': move_x_mm,
                                'move_y_mm': move_y_mm
                            }

            # Create COMPACT binary data packet
            if closest_valley:
                # CMD = 1: Valley detected
                cmd_byte = (1).to_bytes(1, 'big')

                # Convert to 0.1mm units and clamp to 16-bit range
                # X coordinate (0.1mm units)
                x_value = int(closest_valley['move_y_mm'] * 10)
                x_value = max(-32768, min(32767, x_value))
                x_bytes = x_value.to_bytes(2, 'big', signed=True)

                # Y coordinate (0.1mm units)
                y_value = int(closest_valley['move_x_mm'] * 10)
                y_value = max(-32768, min(32767, y_value))
                y_bytes = y_value.to_bytes(2, 'big', signed=True)

                # Total packet: 5 bytes
                data = cmd_byte + x_bytes + y_bytes
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
        self.config_sent.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("✓ TCP server stopped")


# Test client for verification
class BladeDataTCPClientV2:
    """Simple TCP client for testing the new protocol"""

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

    def receive_configuration(self):
        """Receive initial configuration packet"""
        try:
            # Receive 9 bytes: [CMD][BAY][GRINDER][ANGLE(2)][DEPTH(2)][LENGTH(2)]
            config_data = self.socket.recv(9)
            
            if len(config_data) != 9:
                print(f"✗ Incomplete config packet: {len(config_data)} bytes")
                return None
            
            cmd = config_data[0]
            if cmd != 0:
                print(f"✗ Expected CMD=0, got CMD={cmd}")
                return None
            
            bay_id = config_data[1]
            grinder_id = config_data[2]
            angle = int.from_bytes(config_data[3:5], 'big', signed=True) / 10.0
            depth = int.from_bytes(config_data[5:7], 'big', signed=True) / 100.0
            length = int.from_bytes(config_data[7:9], 'big', signed=False)
            
            config = {
                'bay_id': bay_id,
                'grinder_id': grinder_id,
                'angle': angle,
                'depth': depth,
                'length': length
            }
            
            print(f"✓ Received configuration: {config}")
            return config
            
        except Exception as e:
            print(f"✗ Config receive error: {e}")
            return None

    def request_detection_data(self):
        """Request detection data from server"""
        try:
            # Send CMD=3: Request data
            request = (3).to_bytes(1, 'big')
            self.socket.sendall(request)
            
            # Receive response
            cmd_data = self.socket.recv(1)
            if not cmd_data:
                print("✗ No response from server")
                return None
            
            cmd = cmd_data[0]
            
            if cmd == 1:
                # Detection data: read 4 more bytes [X(2)][Y(2)]
                coord_data = self.socket.recv(4)
                if len(coord_data) != 4:
                    print(f"✗ Incomplete data: {len(coord_data)} bytes")
                    return None
                
                x_value = int.from_bytes(coord_data[0:2], 'big', signed=True) / 10.0
                y_value = int.from_bytes(coord_data[2:4], 'big', signed=True) / 10.0
                
                return {
                    'cmd': 1,
                    'x_mm': x_value,
                    'y_mm': y_value
                }
            
            elif cmd == 2:
                # No detection
                return {'cmd': 2, 'status': 'no_detection'}
            
            else:
                print(f"✗ Unknown command: {cmd}")
                return None
                
        except Exception as e:
            print(f"✗ Request error: {e}")
            return None

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
    server = BladeDataTCPServerV2(host='172.24.9.15', port=5000)
    
    # Set configuration
    server.set_configuration(
        bay_id=5,
        grinder_id=2,
        angle=45.5,
        depth=1.25,
        length=150
    )

    if server.start():
        print("\nServer running. Press Ctrl+C to stop.")
        print("Protocol:")
        print("  1. Server sends config: [CMD=0][BAY][GRINDER][ANGLE][DEPTH][LENGTH] (9 bytes)")
        print("  2. Client requests data: [CMD=3] (1 byte)")
        print("  3. Server responds: [CMD=1][X(2)][Y(2)] (5 bytes) or [CMD=2] (1 byte)")

        try:
            count = 0
            while True:
                # Simulate detection data
                test_data = {
                    'grinder_tip': (500, 300),
                    'teeth_profiles': [
                        type('Tooth', (), {'grinding_point': (400, 200)})(),
                        type('Tooth', (), {'grinding_point': (450, 220)})(),
                        type('Tooth', (), {'grinding_point': (500, 240)})(),
                    ]
                }

                server.publish_data(test_data)
                count += 1

                print(f"\rReady for requests | Clients: {server.get_client_count()}", end='')
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            server.stop()