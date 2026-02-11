"""
Test Client for Blade Detection System
Demonstrates the request-response protocol:
1. Receives configuration packet
2. Requests detection data when ready
3. Processes compact data packets
"""

import socket
import time
import struct


class BladeTestClient:
    """Test client for the restructured blade detection system"""

    def __init__(self, host='172.24.9.15', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.config = None

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
        """
        Receive initial configuration packet from server
        Packet format: [CMD=0][BAY_ID][GRINDER_ID][ANGLE(2)][DEPTH(2)][LENGTH(2)]
        Total: 9 bytes
        """
        try:
            print("\nWaiting for configuration packet...")
            config_data = self.socket.recv(9)
            
            if len(config_data) != 9:
                print(f"✗ Incomplete config packet: {len(config_data)} bytes")
                return False
            
            # Parse configuration
            cmd = config_data[0]
            if cmd != 0:
                print(f"✗ Expected CMD=0, got CMD={cmd}")
                return False
            
            bay_id = config_data[1]
            grinder_id = config_data[2]
            angle = int.from_bytes(config_data[3:5], 'big', signed=True) / 10.0
            depth = int.from_bytes(config_data[5:7], 'big', signed=True) / 100.0
            length = int.from_bytes(config_data[7:9], 'big', signed=False)
            
            self.config = {
                'bay_id': bay_id,
                'grinder_id': grinder_id,
                'angle': angle,
                'depth': depth,
                'length': length
            }
            
            print("✓ Configuration received:")
            print(f"  Bay ID:      {self.config['bay_id']}")
            print(f"  Grinder ID:  {self.config['grinder_id']}")
            print(f"  Angle:       {self.config['angle']:.1f}°")
            print(f"  Depth:       {self.config['depth']:.2f}")
            print(f"  Length:      {self.config['length']}")
            
            return True
            
        except Exception as e:
            print(f"✗ Config receive error: {e}")
            return False

    def check_for_new_configuration(self, timeout=0.01):
        """
        Check if server has sent a new configuration (non-blocking)
        Returns True if new config received, False otherwise
        
        Args:
            timeout: Socket timeout in seconds
        """
        try:
            # Set short timeout for non-blocking check
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(timeout)
            
            # Try to peek at incoming data
            data = self.socket.recv(1, socket.MSG_PEEK)
            
            if data and data[0] == 0:  # CMD=0 means configuration
                # Actually receive the full config packet
                config_data = self.socket.recv(9)
                
                if len(config_data) == 9:
                    bay_id = config_data[1]
                    grinder_id = config_data[2]
                    angle = int.from_bytes(config_data[3:5], 'big', signed=True) / 10.0
                    depth = int.from_bytes(config_data[5:7], 'big', signed=True) / 100.0
                    length = int.from_bytes(config_data[7:9], 'big', signed=False)
                    
                    self.config = {
                        'bay_id': bay_id,
                        'grinder_id': grinder_id,
                        'angle': angle,
                        'depth': depth,
                        'length': length
                    }
                    
                    print("\n" + "="*70)
                    print("🔄 NEW CONFIGURATION RECEIVED FROM SERVER")
                    print("="*70)
                    print(f"  Bay ID:      {self.config['bay_id']}")
                    print(f"  Grinder ID:  {self.config['grinder_id']}")
                    print(f"  Angle:       {self.config['angle']:.1f}°")
                    print(f"  Depth:       {self.config['depth']:.2f}")
                    print(f"  Length:      {self.config['length']}")
                    print("="*70 + "\n")
                    
                    # Restore timeout
                    self.socket.settimeout(original_timeout)
                    return True
            
            # Restore timeout
            self.socket.settimeout(original_timeout)
            return False
            
        except socket.timeout:
            return False
        except Exception as e:
            return False

    def request_detection_data(self):
        """
        Request detection data from server
        Send: [CMD=3]
        Receive: [CMD=1][X(2)][Y(2)] or [CMD=2]
        """
        try:
            # Send request
            request = (3).to_bytes(1, 'big')
            self.socket.sendall(request)
            
            # Receive response
            cmd_data = self.socket.recv(1)
            if not cmd_data:
                print("✗ No response from server")
                return None
            
            cmd = cmd_data[0]
            
            if cmd == 1:
                # Detection data available: read 4 bytes [X(2)][Y(2)]
                coord_data = self.socket.recv(4)
                if len(coord_data) != 4:
                    print(f"✗ Incomplete data: {len(coord_data)} bytes")
                    return None
                
                # Parse coordinates (in 0.1mm units)
                x_value = int.from_bytes(coord_data[0:2], 'big', signed=True) / 10.0
                y_value = int.from_bytes(coord_data[2:4], 'big', signed=True) / 10.0
                
                return {
                    'cmd': 1,
                    'x_mm': x_value,
                    'y_mm': y_value,
                    'status': 'detected'
                }
            
            elif cmd == 2:
                # No detection
                return {
                    'cmd': 2,
                    'status': 'no_detection'
                }
            
            else:
                print(f"✗ Unknown command: {cmd}")
                return None
                
        except Exception as e:
            print(f"✗ Request error: {e}")
            return None

    def run_continuous_mode(self, request_rate_hz=5.0):
        """
        Run in continuous mode - request data at specified rate
        Also monitors for new configuration updates from server
        
        Args:
            request_rate_hz: How many times per second to request data
        """
        if not self.config:
            print("✗ No configuration received. Cannot start continuous mode.")
            return
        
        print(f"\n{'='*70}")
        print(f"Starting continuous data requests at {request_rate_hz} Hz")
        print("Server can send new configurations at any time")
        print("Press Ctrl+C to stop")
        print(f"{'='*70}\n")
        
        request_interval = 1.0 / request_rate_hz
        request_count = 0
        detection_count = 0
        config_updates = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                # Check for new configuration before requesting data
                if self.check_for_new_configuration():
                    config_updates += 1
                    print(f"📋 Total configuration updates received: {config_updates}")
                
                # Request detection data
                data = self.request_detection_data()
                
                if data:
                    request_count += 1
                    
                    if data['cmd'] == 1:
                        detection_count += 1
                        config_str = f"[Bay:{self.config['bay_id']} Grind:{self.config['grinder_id']}]"
                        print(f"[{request_count:04d}] {config_str} ✓ Detection: X={data['x_mm']:+7.1f}mm, Y={data['y_mm']:+7.1f}mm")
                    elif data['cmd'] == 2:
                        print(f"[{request_count:04d}] ○ No detection")
                else:
                    print(f"[{request_count:04d}] ✗ Request failed")
                
                # Maintain request rate
                elapsed = time.time() - start_time
                sleep_time = max(0, request_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Statistics:")
            print(f"  Total requests:       {request_count}")
            print(f"  Detections:           {detection_count}")
            if request_count > 0:
                print(f"  Detection rate:       {(detection_count/request_count)*100:.1f}%")
            print(f"  Config updates:       {config_updates}")
            print("\nFinal Configuration:")
            print(f"  Bay ID:      {self.config['bay_id']}")
            print(f"  Grinder ID:  {self.config['grinder_id']}")
            print(f"  Angle:       {self.config['angle']:.1f}°")
            print(f"  Depth:       {self.config['depth']:.2f}")
            print(f"  Length:      {self.config['length']}")
            print("="*70)

    def disconnect(self):
        """Disconnect from server"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("✓ Disconnected")


def main():
    """Main test client"""
    print("\n" + "="*70)
    print("BLADE DETECTION TEST CLIENT")
    print("="*70)
    
    # Configuration
    HOST = '172.24.9.15'  # Change to your server IP
    PORT = 5000
    REQUEST_RATE = 5.0  # Requests per second
    
    client = BladeTestClient(host=HOST, port=PORT)
    
    # Connect to server
    if not client.connect():
        return
    
    # Receive configuration
    if not client.receive_configuration():
        client.disconnect()
        return
    
    # Wait for user to be ready
    print("\nConfiguration received successfully!")
    input("Press Enter when ready to start requesting detection data...")
    
    # Run continuous data requests
    try:
        client.run_continuous_mode(request_rate_hz=REQUEST_RATE)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()