from pymodbus.client import ModbusTcpClient

class BladeDataModbusClient:
    """
    Modbus TCP client for blade grinder robot communication.

    Loop structure:
    ──────────────────────────────────────────────────────
    PC writes detection X/Y + status=1 (teeth detected)
    PC writes START=1          → robot begins moving to tooth valley
    Robot moves, then sets     GRINDER_READY=1
    PC reads GRINDER_READY     → triggers camera on / re-detection
    PC writes new X/Y + GRIND_START=1 → robot starts grinding
    Robot grinds, resets       GRIND_START=0 (ready for next cut)
    Repeat from GRIND_START step for each tooth
    ──────────────────────────────────────────────────────
    """

    # ── Register Map ─────────────────────────────────────
    REG_DETECTION_X    = 134   # PC → Robot  | Detection X offset (×10, signed as uint16)
    REG_DETECTION_Y    = 135   # PC → Robot  | Detection Y offset (×10, signed as uint16)
    REG_STATUS         = 136   # PC → Robot  | 0=no teeth, 1=teeth detected, 2=error
    REG_START          = 137   # PC → Robot  | 1=start full loop, 0=idle
    REG_GRINDER_READY  = 138   # Robot → PC  | 1=robot reached grinder position (triggers camera)
    REG_GRIND_START    = 139   # PC → Robot  | 1=start grinding; robot resets to 0 after each move

    # ── Status codes ──────────────────────────────────────
    STATUS_NO_TEETH    = 0
    STATUS_TEETH_OK    = 1
    STATUS_ERROR       = 2

    def __init__(self, host='172.24.89.89', port=502, unit=1):
        """
        Args:
            host: Robot IP address
            port: Modbus TCP port (default 502)
            unit: Modbus slave ID
        """
        self.host = host
        self.port = port
        self.unit = unit
        self.client = ModbusTcpClient(host, port=port)
        self.connected = False

    # ── Connection ────────────────────────────────────────

    def connect(self):
        """Connect to the robot Modbus server."""
        if self.client.connect():
            self.connected = True
            print(f"✓ Connected to robot at {self.host}:{self.port}")
            return True
        self.connected = False
        print(f"✗ Could not connect to robot at {self.host}:{self.port}")
        return False

    def close(self):
        """Close Modbus connection."""
        self.client.close()
        self.connected = False
        print("✓ Modbus connection closed")

    # ── Detection data ────────────────────────────────────

    def write_detection(self, x_mm, y_mm, status):
        """
        Write detection X/Y offsets and status to robot.

        Args:
            x_mm:   X offset in mm (signed, converted to 0.1mm units)
            y_mm:   Y offset in mm (signed, converted to 0.1mm units)
            status: STATUS_NO_TEETH=0, STATUS_TEETH_OK=1, STATUS_ERROR=2
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        # Scale to 0.1 mm resolution, encode signed as unsigned 16-bit
        x_val = int(x_mm * 10)
        y_val = int(y_mm * 10)
        x_u16 = x_val if x_val >= 0 else 65536 + x_val
        y_u16 = y_val if y_val >= 0 else 65536 + y_val

        values = [x_u16, y_u16, int(status)]
        result = self.client.write_registers(address=self.REG_DETECTION_X, values=values)

        if not result.isError():
            status_name = {0: 'NO_TEETH', 1: 'TEETH_OK', 2: 'ERROR'}.get(status, str(status))
            print(f"✓ Detection written: X={x_mm:.2f}mm, Y={y_mm:.2f}mm, Status={status_name}")
        else:
            print(f"✗ Failed to write detection data: {result}")

        return result

    # ── Loop control ──────────────────────────────────────

    def start_loop(self):
        """
        Signal robot to start the full grinding loop.
        PC should have already written valid detection data before calling this.
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        print("🚀 Starting grinding loop (START=1)...")
        result = self.client.write_register(address=self.REG_START, value=1)

        if not result.isError():
            print("✓ Loop started")
        else:
            print(f"✗ Failed to start loop: {result}")

        return result

    def stop_loop(self):
        """Stop / abort the grinding loop (START=0)."""
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        print("🛑 Stopping loop (START=0)...")
        result = self.client.write_register(address=self.REG_START, value=0)

        if not result.isError():
            print("✓ Loop stopped")
        else:
            print(f"✗ Failed to stop loop: {result}")

        return result

    def send_grind_start(self):
        """
        Tell robot to begin grinding the current tooth position.
        Called by PC after GRINDER_READY=1 is detected and new detection data
        has been written.
        Robot resets GRIND_START back to 0 after completing the move,
        signalling it is ready for the next tooth.
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        print("⚙️  Sending GRIND_START=1...")
        result = self.client.write_register(address=self.REG_GRIND_START, value=1)

        if not result.isError():
            print("✓ Grind start sent")
        else:
            print(f"✗ Failed to send grind start: {result}")

        return result

    # ── Read robot state ──────────────────────────────────

    def read_grinder_ready(self):
        """
        Read GRINDER_READY register from robot.
        Returns True when robot has finished moving to grinder position
        and is waiting for GRIND_START.
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        result = self.client.read_holding_registers(address=self.REG_GRINDER_READY, count=1)

        if result.isError():
            print(f"✗ Failed to read GRINDER_READY: {result}")
            return None

        value = result.registers[0]
        return bool(value)

    def read_grind_start(self):
        """
        Read GRIND_START register.
        Returns False (0) when robot has reset it after completing a cut —
        signals PC that the next GRIND_START can be sent.
        """
        if not self.connected:
            print("✗ Not connected to robot")
            return None

        result = self.client.read_holding_registers(address=self.REG_GRIND_START, count=1)

        if result.isError():
            print(f"✗ Failed to read GRIND_START: {result}")
            return None

        return bool(result.registers[0])

    def read_all_status(self):
        """Read all loop-related registers in one shot for monitoring."""
        if not self.connected:
            return None

        # Read registers 134–139 (6 registers)
        result = self.client.read_holding_registers(
            address=self.REG_DETECTION_X, count=6
        )

        if result.isError():
            print(f"✗ Failed to read status registers: {result}")
            return None

        regs = result.registers
        # Decode signed 16-bit
        def to_signed(v):
            return v if v < 32768 else v - 65536

        return {
            'detection_x_mm': to_signed(regs[0]) / 10.0,
            'detection_y_mm': to_signed(regs[1]) / 10.0,
            'status':         regs[2],
            'start':          regs[3],
            'grinder_ready':  regs[4],
            'grind_start':    regs[5],
        }


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    client = BladeDataModbusClient()

    if not client.connect():
        exit(1)

    # --- Step 1: Write initial detection data ---
    client.write_detection(x_mm=5.2, y_mm=2.1, status=BladeDataModbusClient.STATUS_TEETH_OK)

    # --- Step 2: Start the full loop ---
    client.start_loop()

    # --- Step 3: Poll for GRINDER_READY, then send each grind command ---
    print("\nWaiting for robot to reach grinder position...")
    timeout = 30  # seconds
    t0 = time.time()

    while time.time() - t0 < timeout:
        ready = client.read_grinder_ready()
        if ready:
            print("✓ Grinder is ready! Writing updated detection & sending GRIND_START...")

            # (Re-run camera detection here in real usage)
            client.write_detection(x_mm=5.2, y_mm=2.1, status=BladeDataModbusClient.STATUS_TEETH_OK)
            client.send_grind_start()

            # Wait for robot to reset GRIND_START (signals cut is done)
            print("  Waiting for robot to complete cut...")
            while client.read_grind_start():
                time.sleep(0.1)
            print("  ✓ Cut complete, ready for next tooth")
            break

        time.sleep(0.2)
    else:
        print("✗ Timeout waiting for GRINDER_READY")

    # --- Stop ---
    client.stop_loop()
    client.close()