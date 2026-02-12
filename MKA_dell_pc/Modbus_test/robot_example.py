"""
Robot Example: Reading Blade Detection Data via Modbus
This script demonstrates how a robot controller reads configuration
and detection data from the Modbus server
"""

from pymodbus.client import ModbusTcpClient
import time

# Server configuration
VISION_SYSTEM_IP = "172.24.9.15"  # IP of the vision computer
MODBUS_PORT = 502

# Register addresses (must match server)
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


def read_configuration(client):
    """
    Read blade configuration from Modbus registers
    Returns dict with configuration or None on error
    """
    try:
        result = client.read_holding_registers(address=REG_BAY_ID, count=6)
        
        if result.isError():
            print("Error reading configuration")
            return None
        
        config = {
            'bay_id': result.registers[0],
            'grinder_id': result.registers[1],
            'angle': result.registers[2] / 10.0,  # Divide by 10
            'depth': result.registers[3] / 100.0,  # Divide by 100
            'length': result.registers[4],
            'version': result.registers[5]
        }
        
        return config
    
    except Exception as e:
        print(f"Exception reading configuration: {e}")
        return None


def read_detection_data(client):
    """
    Read detection data from Modbus registers
    Returns dict with x, y, status or None on error
    """
    try:
        result = client.read_holding_registers(address=REG_DETECTION_X, count=3)
        
        if result.isError():
            print("Error reading detection data")
            return None
        
        # Convert unsigned to signed (Modbus uses unsigned 16-bit)
        x_unsigned = result.registers[0]
        y_unsigned = result.registers[1]
        status = result.registers[2]
        
        # Convert to signed values
        x_mm = (x_unsigned if x_unsigned < 32768 else x_unsigned - 65536) / 10.0
        y_mm = (y_unsigned if y_unsigned < 32768 else y_unsigned - 65536) / 10.0
        
        detection = {
            'x_mm': x_mm,
            'y_mm': y_mm,
            'status': status,  # 0=no_data, 1=valid, 2=no_detection
            'status_text': {0: "No Data", 1: "Valid Detection", 2: "No Detection"}.get(status, "Unknown")
        }
        
        return detection
    
    except Exception as e:
        print(f"Exception reading detection: {e}")
        return None


def robot_workflow_example():
    """
    Example workflow showing how robot reads data in a loop
    """
    print("\n" + "="*70)
    print("ROBOT WORKFLOW EXAMPLE")
    print("="*70)
    
    # Connect to Modbus server
    client = ModbusTcpClient(VISION_SYSTEM_IP, port=MODBUS_PORT)
    
    if not client.connect():
        print(f"Failed to connect to {VISION_SYSTEM_IP}:{MODBUS_PORT}")
        return
    
    print(f"✓ Connected to vision system at {VISION_SYSTEM_IP}:{MODBUS_PORT}")
    
    try:
        # Step 1: Read configuration
        print("\n[STEP 1] Reading blade configuration...")
        config = read_configuration(client)
        
        if config:
            print(f"✓ Configuration loaded:")
            print(f"  Bay ID:      {config['bay_id']}")
            print(f"  Grinder ID:  {config['grinder_id']}")
            print(f"  Angle:       {config['angle']:.1f}°")
            print(f"  Depth:       {config['depth']:.2f}")
            print(f"  Length:      {config['length']}")
            print(f"  Version:     {config['version']}")
        else:
            print("✗ Failed to read configuration")
            return
        
        # Step 2: Continuous detection reading loop
        print("\n[STEP 2] Starting detection loop...")
        print("Reading detection data every 200ms. Press Ctrl+C to stop.\n")
        
        last_version = config['version']
        detection_count = 0
        
        while True:
            # Check if configuration has changed
            current_config = read_configuration(client)
            if current_config and current_config['version'] != last_version:
                print("\n" + "="*70)
                print("🔄 NEW CONFIGURATION DETECTED!")
                print("="*70)
                print(f"  Bay ID:      {current_config['bay_id']}")
                print(f"  Grinder ID:  {current_config['grinder_id']}")
                print(f"  Angle:       {current_config['angle']:.1f}°")
                print(f"  Depth:       {current_config['depth']:.2f}")
                print(f"  Length:      {current_config['length']}")
                print(f"  Version:     {current_config['version']}")
                print("="*70 + "\n")
                last_version = current_config['version']
                config = current_config
            
            # Read detection data
            detection = read_detection_data(client)
            
            if detection:
                detection_count += 1
                
                if detection['status'] == 1:  # Valid detection
                    print(f"[{detection_count:04d}] Bay:{config['bay_id']} Grind:{config['grinder_id']} | " +
                          f"✓ Detection: X={detection['x_mm']:+7.1f}mm, Y={detection['y_mm']:+7.1f}mm")
                    
                    # HERE IS WHERE YOUR ROBOT WOULD USE THE DATA
                    # Example: Move grinder to detected position
                    # movel(grinder_position + [detection['x_mm'], detection['y_mm'], 0], speed, accel)
                    
                elif detection['status'] == 2:  # No detection
                    print(f"[{detection_count:04d}] ○ No detection")
                else:  # No data yet
                    print(f"[{detection_count:04d}] ⏳ Waiting for data...")
            
            time.sleep(0.2)  # 5 Hz read rate
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        client.close()
        print("✓ Disconnected from vision system")


def simple_read_example():
    """
    Simple example - just read once and print
    """
    print("\n" + "="*70)
    print("SIMPLE READ EXAMPLE")
    print("="*70)
    
    client = ModbusTcpClient(VISION_SYSTEM_IP, port=MODBUS_PORT)
    
    if client.connect():
        print(f"✓ Connected to {VISION_SYSTEM_IP}:{MODBUS_PORT}\n")
        
        # Read configuration
        config = read_configuration(client)
        if config:
            print("Configuration:")
            for key, value in config.items():
                print(f"  {key:12s}: {value}")
        
        print()
        
        # Read detection
        detection = read_detection_data(client)
        if detection:
            print("Detection:")
            for key, value in detection.items():
                print(f"  {key:12s}: {value}")
        
        client.close()
    else:
        print(f"✗ Connection failed to {VISION_SYSTEM_IP}:{MODBUS_PORT}")


def robot_control_pattern():
    """
    Example pattern showing typical robot control logic
    """
    print("\n" + "="*70)
    print("ROBOT CONTROL PATTERN EXAMPLE")
    print("="*70)
    
    client = ModbusTcpClient(VISION_SYSTEM_IP, port=MODBUS_PORT)
    
    if not client.connect():
        print(f"✗ Failed to connect")
        return
    
    print(f"✓ Connected\n")
    
    try:
        # Read configuration to know which bay/grinder
        config = read_configuration(client)
        print(f"Processing Bay {config['bay_id']}, Grinder {config['grinder_id']}")
        
        # Main control loop
        print("\nStarting grinding cycle...\n")
        
        for cycle in range(10):  # Process 10 teeth
            print(f"Cycle {cycle + 1}/10:")
            
            # 1. Move to inspection position
            print("  → Moving to inspection position")
            # movel(inspection_pos, speed=100, accel=20)
            time.sleep(0.5)  # Simulate movement
            
            # 2. Read detection data
            print("  → Reading vision data")
            detection = read_detection_data(client)
            
            if detection and detection['status'] == 1:
                x_offset = detection['x_mm']
                y_offset = detection['y_mm']
                print(f"  ✓ Tooth detected at offset: X={x_offset:+.1f}mm, Y={y_offset:+.1f}mm")
                
                # 3. Calculate grinding position
                print(f"  → Calculating grinding position")
                # grind_pos = base_position + [x_offset, y_offset, 0, 0, 0, 0]
                
                # 4. Move to grinding position
                print(f"  → Moving to grind position")
                # movel(grind_pos, speed=50, accel=20)
                time.sleep(0.5)
                
                # 5. Execute grinding
                print(f"  → Grinding (depth={config['depth']:.2f}, angle={config['angle']:.1f}°)")
                # grind_operation(config['depth'], config['angle'])
                time.sleep(1.0)
                
                # 6. Retract
                print(f"  → Retracting")
                # movel(retract_pos, speed=100, accel=20)
                time.sleep(0.5)
                
            else:
                print("  ✗ No valid detection, skipping")
            
            print()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        client.close()
        print("\n✓ Complete")


if __name__ == "__main__":
    # Choose which example to run:
    
    # Option 1: Simple single read
    # simple_read_example()
    
    # Option 2: Continuous reading workflow
    robot_workflow_example()
    
    # Option 3: Full robot control pattern
    # robot_control_pattern()
