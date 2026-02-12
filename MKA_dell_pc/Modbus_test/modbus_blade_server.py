"""
Modbus Server for Blade Detection
Robot reads Modbus holding registers to get blade data
- Configuration stored in registers 128-133
- Detection data stored in registers 134-136
- Status/command register at 137
"""

from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.device import ModbusDeviceIdentification
import threading
import time
from datetime import datetime


class BladeDataModbusServer:
    """Modbus TCP server for blade detection data"""
    
    # Register mapping
    REG_BAY_ID = 128          # Bay ID (1-10)
    REG_GRINDER_ID = 129      # Grinder ID (1-3)
    REG_ANGLE = 130           # Angle * 10 (e.g., 45.5° = 455)
    REG_DEPTH = 131           # Depth * 100 (e.g., 1.25 = 125)
    REG_LENGTH = 132          # Length (0-999)
    REG_CONFIG_VERSION = 133  # Increments when config changes
    
    REG_DETECTION_X = 134     # X coordinate * 10 (in 0.1mm units)
    REG_DETECTION_Y = 135     # Y coordinate * 10 (in 0.1mm units)
    REG_STATUS = 136          # Status: 0=no_data, 1=detection_valid, 2=no_detection
    REG_COMMAND = 137         # Command from robot: 0=idle, 10=read_config, 11=read_detection
    
    def __init__(self, host='0.0.0.0', port=502):
        """
        Initialize Modbus server
        
        Args:
            host: Server IP address (0.0.0.0 = listen on all interfaces)
            port: Modbus TCP port (default 502)
        """
        self.host = host
        self.port = port
        self.running = False
        
        # Current configuration
        self.bay_id = 0
        self.grinder_id = 0
        self.angle = 0.0
        self.depth = 0.0
        self.length = 0
        self.config_version = 0
        
        # Current detection data
        self.detection_x = 0
        self.detection_y = 0
        self.detection_status = 0  # 0=no data, 1=valid, 2=no detection
        
        # Modbus data store (holding registers)
        # Initialize with 1000 registers starting at address 0
        self.store = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0] * 1000)
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)
        
        # Server thread
        self.server_thread = None
        
        print(f"Modbus server initialized on {host}:{port}")

    def set_configuration(self, bay_id, grinder_id, angle, depth, length):
        """
        Set blade configuration in Modbus registers
        
        Args:
            bay_id: Bay ID (1-10)
            grinder_id: Grinder ID (1-3)
            angle: Angle in degrees
            depth: Depth
            length: Length (0-999)
        """
        # Validate inputs
        if not (1 <= bay_id <= 10):
            raise ValueError("BAY ID must be 1-10")
        if not (1 <= grinder_id <= 3):
            raise ValueError("GRINDER ID must be 1-3")
        
        # Store configuration
        self.bay_id = bay_id
        self.grinder_id = grinder_id
        self.angle = angle
        self.depth = depth
        self.length = length
        self.config_version += 1  # Increment version
        
        # Write to Modbus registers
        self._write_configuration()
        
        print(f"✓ Configuration updated (v{self.config_version}): Bay={bay_id}, Grinder={grinder_id}, " +
              f"Angle={angle:.1f}°, Depth={depth:.2f}, Length={length}")

    def _write_configuration(self):
        """Write configuration to Modbus registers"""
        values = [
            self.bay_id,                      # REG_BAY_ID (128)
            self.grinder_id,                  # REG_GRINDER_ID (129)
            int(self.angle * 10),             # REG_ANGLE (130)
            int(self.depth * 100),            # REG_DEPTH (131)
            self.length,                      # REG_LENGTH (132)
            self.config_version,              # REG_CONFIG_VERSION (133)
        ]
        
        self.context[0].setValues(3, self.REG_BAY_ID, values)

    def publish_data(self, blade_results):
        """
        Publish blade/tooth detection data to Modbus registers
        
        Args:
            blade_results: Detection results dictionary
        """
        if not self.running:
            return

        try:
            teeth_profiles = blade_results.get('teeth_profiles', [])
            grinder_tip = blade_results.get('grinder_tip')
            pixels_per_mm = 86.96

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

            # Update Modbus registers
            if closest_valley:
                # Convert to 0.1mm units and clamp to 16-bit range
                x_value = int(closest_valley['move_y_mm'] * 10)
                x_value = max(-32768, min(32767, x_value))
                
                y_value = int(closest_valley['move_x_mm'] * 10)
                y_value = max(-32768, min(32767, y_value))
                
                self.detection_x = x_value
                self.detection_y = y_value
                self.detection_status = 1  # Valid detection
            else:
                self.detection_status = 2  # No detection
            
            # Write detection data to registers
            self._write_detection()

        except Exception as e:
            print(f"Publish error: {e}")

    def _write_detection(self):
        """Write detection data to Modbus registers"""
        # Convert signed values to unsigned for Modbus
        x_unsigned = self.detection_x if self.detection_x >= 0 else 65536 + self.detection_x
        y_unsigned = self.detection_y if self.detection_y >= 0 else 65536 + self.detection_y
        
        values = [
            x_unsigned,           # REG_DETECTION_X (134)
            y_unsigned,           # REG_DETECTION_Y (135)
            self.detection_status # REG_STATUS (136)
        ]
        
        self.context[0].setValues(3, self.REG_DETECTION_X, values)

    def start(self):
        """Start the Modbus TCP server"""
        if self.running:
            print("Server already running")
            return True
        
        self.running = True
        
        # Start server in separate thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        
        time.sleep(0.5)  # Give server time to start
        print(f"✓ Modbus server started on {self.host}:{self.port}")
        return True

    def _run_server(self):
        """Run the Modbus server (called in thread)"""
        try:
            # Set device identification (optional)
            identity = ModbusDeviceIdentification()
            identity.VendorName = 'Blade Detection System'
            identity.ProductCode = 'BDS-001'
            identity.VendorUrl = 'http://localhost'
            identity.ProductName = 'Blade Detection Modbus Server'
            identity.ModelName = 'BDS-MODBUS-V1'
            identity.MajorMinorRevision = '1.0.0'
            
            # Start the server
            StartTcpServer(
                context=self.context,
                identity=identity,
                address=(self.host, self.port),
                allow_reuse_address=True
            )
        except Exception as e:
            print(f"Server error: {e}")
            self.running = False

    def stop(self):
        """Stop the Modbus server"""
        print("Stopping Modbus server...")
        self.running = False
        # Note: pymodbus doesn't have a clean way to stop the server
        # The server thread will exit when the main program exits
        print("✓ Modbus server stopped")

    def get_register_map(self):
        """Return register map as string for documentation"""
        return f"""
Modbus Register Map:
{'='*60}
Configuration Registers (Read by Robot):
  {self.REG_BAY_ID:3d} - Bay ID (1-10)
  {self.REG_GRINDER_ID:3d} - Grinder ID (1-3)
  {self.REG_ANGLE:3d} - Angle * 10 (e.g., 455 = 45.5°)
  {self.REG_DEPTH:3d} - Depth * 100 (e.g., 125 = 1.25)
  {self.REG_LENGTH:3d} - Length (0-999)
  {self.REG_CONFIG_VERSION:3d} - Config version (increments on update)

Detection Data Registers (Read by Robot):
  {self.REG_DETECTION_X:3d} - X coordinate * 10 (0.1mm units, signed)
  {self.REG_DETECTION_Y:3d} - Y coordinate * 10 (0.1mm units, signed)
  {self.REG_STATUS:3d} - Status (0=no_data, 1=valid, 2=no_detection)

Command Register (Written by Robot):
  {self.REG_COMMAND:3d} - Command (0=idle, 10=read_config, 11=read_detection)
{'='*60}
"""


def test_modbus_server():
    """Test function to demonstrate Modbus server"""
    from pymodbus.client import ModbusTcpClient
    
    print("\n" + "="*70)
    print("MODBUS SERVER TEST")
    print("="*70)
    
    # Start server
    server = BladeDataModbusServer(host='0.0.0.0', port=502)
    
    # Set configuration
    server.set_configuration(
        bay_id=5,
        grinder_id=2,
        angle=45.5,
        depth=1.25,
        length=150
    )
    
    if server.start():
        print("\nServer running. Starting test client...")
        print(server.get_register_map())
        
        time.sleep(1)
        
        # Test with client
        print("\nTesting with Modbus client...")
        client = ModbusTcpClient('localhost', port=502)
        
        if client.connect():
            print("✓ Client connected")
            
            # Read configuration
            print("\nReading configuration registers...")
            result = client.read_holding_registers(address=128, count=6)
            if not result.isError():
                bay_id = result.registers[0]
                grinder_id = result.registers[1]
                angle = result.registers[2] / 10.0
                depth = result.registers[3] / 100.0
                length = result.registers[4]
                version = result.registers[5]
                
                print(f"  Bay ID:      {bay_id}")
                print(f"  Grinder ID:  {grinder_id}")
                print(f"  Angle:       {angle:.1f}°")
                print(f"  Depth:       {depth:.2f}")
                print(f"  Length:      {length}")
                print(f"  Version:     {version}")
            
            # Simulate detection data
            print("\nSimulating detection data...")
            test_data = {
                'grinder_tip': (500, 300),
                'teeth_profiles': [
                    type('Tooth', (), {'grinding_point': (400, 200)})(),
                    type('Tooth', (), {'grinding_point': (450, 220)})(),
                    type('Tooth', (), {'grinding_point': (500, 240)})(),
                ]
            }
            server.publish_data(test_data)
            
            time.sleep(0.5)
            
            # Read detection data
            print("\nReading detection registers...")
            result = client.read_holding_registers(address=134, count=3)
            if not result.isError():
                x_unsigned = result.registers[0]
                y_unsigned = result.registers[1]
                status = result.registers[2]
                
                # Convert unsigned to signed
                x_mm = (x_unsigned if x_unsigned < 32768 else x_unsigned - 65536) / 10.0
                y_mm = (y_unsigned if y_unsigned < 32768 else y_unsigned - 65536) / 10.0
                
                status_str = {0: "No Data", 1: "Valid Detection", 2: "No Detection"}
                print(f"  X:      {x_mm:+.1f}mm")
                print(f"  Y:      {y_mm:+.1f}mm")
                print(f"  Status: {status_str.get(status, 'Unknown')}")
            
            client.close()
            print("\n✓ Test complete")
        else:
            print("✗ Client connection failed")
        
        print("\nServer will continue running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping server...")
            server.stop()


if __name__ == "__main__":
    test_modbus_server()
