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
        if self.client.connect():
            print(f"✓ Connected to robot at {host}:{port}")
        else:
            raise ConnectionError(f"✗ Could not connect to robot at {host}:{port}")

    def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
        """
        Write blade configuration to robot's Modbus server
        """
        values = [
            int(bay_id),
            int(grinder_id),
            int(angle * 10),
            int(depth * 100),
            int(length),
            int(config_version)
        ]

        # Remove 'unit' argument for pymodbus 3.x
        result = self.client.write_registers(address=128, values=values)  # 128 = REG_BAY_ID
        return result

    def write_detection(self, x_mm, y_mm, status):
        """
        Write detection results to robot's Modbus server
        """
        # Convert signed mm values to 0.1mm units (0.1mm = multiply by 10)
        x_val = int(x_mm * 10)
        y_val = int(y_mm * 10)

        # Convert signed to unsigned 16-bit (0-65535)
        x_unsigned = x_val if x_val >= 0 else 65536 + x_val
        y_unsigned = y_val if y_val >= 0 else 65536 + y_val
        status_int = int(status)

        values = [x_unsigned, y_unsigned, status_int]

        # Write to robot (no 'unit' needed unless your robot requires it)
        result = self.client.write_registers(address=134, values=values)
        return result

    def write_command(self, command):
        """Write command register to robot"""
        result = self.client.write_register(
            address=self.REG_COMMAND,
            value=command
        )
        if result.isError():
            print("✗ Failed to write command:", result)
        else:
            print(f"✓ Command written: {command}")

    def close(self):
        """Close Modbus connection"""
        self.client.close()
        print("✓ Modbus client connection closed")


# Example usage
if __name__ == "__main__":
    client = BladeDataModbusClient()

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
    client.write_command(11)  # 11 = read_detection

    client.close()
