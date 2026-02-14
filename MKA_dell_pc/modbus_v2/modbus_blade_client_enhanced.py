from pymodbus.client import ModbusTcpClient

class BladeDataModbusClient:
    """Modbus TCP client to write blade data to robot"""

    # Robot register mapping
    REG_BAY_ID = 128
    REG_GRINDER_ID = 129
    REG_ANGLE = 130
    REG_DEPTH = 131
    REG_LENGTH = 132
    REG_CONFIG_VERSION = 133

    REG_DETECTION_X = 134
    REG_DETECTION_Y = 135
    REG_STATUS = 136
    REG_COMMAND = 137
    REG_START = 138  # NEW: Start register to trigger robot operation

    # Command codes
    CMD_READ_DETECTION = 11
    CMD_START_GRINDING = 20
    CMD_STOP = 21
    CMD_RESET = 22

    def __init__(self, host='172.24.89.89', port=502, unit=1):
        """
        Initialize Modbus client
        Args:
            host: Robot IP address
            port: Modbus TCP port (usually 502)
            unit: Modbus slave ID
        """
        self.host = host
        self.port = port
        self.unit = unit
        self.client = ModbusTcpClient(host, port=port)
        self.connected = False
        
    def connect(self):
        """Connect to the robot"""
        if self.client.connect():
            self.connected = True
            print(f"✓ Connected to robot at {self.host}:{self.port}")
            return True
        else:
            self.connected = False
            print(f"✗ Could not connect to robot at {self.host}:{self.port}")
            return False

    def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
        """
        Write blade configuration to robot's Modbus server
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        values = [
            int(bay_id),
            int(grinder_id),
            int(angle * 10),
            int(depth * 100),
            int(length),
            int(config_version)
        ]

        result = self.client.write_registers(address=self.REG_BAY_ID, values=values)
        
        if not result.isError():
            print(f"✓ Configuration written successfully")
            print(f"  Bay ID: {bay_id}, Grinder ID: {grinder_id}")
            print(f"  Angle: {angle}°, Depth: {depth}mm, Length: {length}mm")
        else:
            print(f"✗ Failed to write configuration: {result}")
            
        return result

    def write_detection(self, x_mm, y_mm, status):
        """
        Write detection results to robot's Modbus server
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        # Convert signed mm values to 0.1mm units (0.1mm = multiply by 10)
        x_val = int(x_mm * 10)
        y_val = int(y_mm * 10)

        # Convert signed to unsigned 16-bit (0-65535)
        x_unsigned = x_val if x_val >= 0 else 65536 + x_val
        y_unsigned = y_val if y_val >= 0 else 65536 + y_val
        status_int = int(status)

        values = [x_unsigned, y_unsigned, status_int]

        result = self.client.write_registers(address=self.REG_DETECTION_X, values=values)
        
        if not result.isError():
            print(f"✓ Detection data written: X={x_mm:.2f}mm, Y={y_mm:.2f}mm, Status={status}")
        else:
            print(f"✗ Failed to write detection data: {result}")
            
        return result

    def write_command(self, command):
        """Write command register to robot"""
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        result = self.client.write_register(
            address=self.REG_COMMAND,
            value=command
        )
        if result.isError():
            print(f"✗ Failed to write command: {result}")
        else:
            cmd_name = self._get_command_name(command)
            print(f"✓ Command written: {cmd_name} ({command})")
        return result

    def start_robot(self):
        """
        Send START command to robot
        This triggers the robot to begin the grinding operation
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        print("\n🚀 Starting robot operation...")
        result = self.client.write_register(
            address=self.REG_START,
            value=1
        )
        
        if result.isError():
            print(f"✗ Failed to start robot: {result}")
        else:
            print(f"✓ Robot started successfully!")
            
        return result

    def stop_robot(self):
        """Stop the robot operation"""
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        print("\n🛑 Stopping robot...")
        result = self.client.write_register(
            address=self.REG_START,
            value=0
        )
        
        if not result.isError():
            print(f"✓ Robot stopped")
        else:
            print(f"✗ Failed to stop robot: {result}")
            
        return result

    def reset_robot(self):
        """Reset the robot to initial state"""
        if not self.connected:
            print("✗ Not connected to robot")
            return None
            
        print("\n🔄 Resetting robot...")
        result = self.write_command(self.CMD_RESET)
        return result

    def _get_command_name(self, command):
        """Get human-readable command name"""
        cmd_names = {
            11: "READ_DETECTION",
            20: "START_GRINDING",
            21: "STOP",
            22: "RESET"
        }
        return cmd_names.get(command, f"UNKNOWN_{command}")

    def close(self):
        """Close Modbus connection"""
        self.client.close()
        self.connected = False
        print("✓ Modbus client connection closed")


# Example usage
if __name__ == "__main__":
    # Create client
    client = BladeDataModbusClient()
    
    # Connect to robot
    if not client.connect():
        exit(1)

    # Write configuration
    client.write_configuration(
        bay_id=5,
        grinder_id=2,
        angle=45.5,
        depth=1.25,
        length=150,
        config_version=1
    )

    # Simulate detection data
    client.write_detection(
        x_mm=2.5,
        y_mm=1.8,
        status=1  # 1=valid detection
    )

    # Send command to robot to read detection
    client.write_command(client.CMD_READ_DETECTION)

    # Start the robot operation
    client.start_robot()

    # Wait for user input to stop
    input("\nPress Enter to stop robot...")
    
    # Stop robot
    client.stop_robot()

    # Close connection
    client.close()
