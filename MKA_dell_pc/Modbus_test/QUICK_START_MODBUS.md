# Quick Start Guide - Modbus Implementation

## What Changed?

**OLD**: Custom TCP socket protocol with request/response  
**NEW**: Standard Modbus TCP - robot reads registers anytime

---

## Installation

### Required Python Packages

```bash
pip install pymodbus --break-system-packages
```

---

## Running the System

### 1. Start Vision System

```bash
python test_modbus.py
```

**You will be prompted for configuration**:
```
Enter BAY ID (1-10): 5
Enter GRINDER ID (1-3): 2
Enter ANGLE (degrees, e.g., 45.5): 45.5
Enter DEPTH (e.g., 1.25): 1.25
Enter LENGTH (0-999): 150
```

**System will display**:
- Modbus register map
- Camera feed with detections
- Current configuration in overlay

### 2. Test with Robot Example

```bash
python robot_example.py
```

This demonstrates reading data from the Modbus server.

---

## Register Map (Quick Reference)

### Configuration (Read these once at startup)

| Reg | Name | Description | How to Read |
|-----|------|-------------|-------------|
| 128 | Bay ID | Which bay (1-10) | Direct value |
| 129 | Grinder ID | Which grinder (1-3) | Direct value |
| 130 | Angle | Blade angle | Value ÷ 10 = degrees |
| 131 | Depth | Grind depth | Value ÷ 100 = actual |
| 132 | Length | Blade length | Direct value |
| 133 | Version | Config version | Increments on change |

### Detection (Read these continuously)

| Reg | Name | Description | How to Read |
|-----|------|-------------|-------------|
| 134 | X Position | X offset | Convert to signed, ÷ 10 = mm |
| 135 | Y Position | Y offset | Convert to signed, ÷ 10 = mm |
| 136 | Status | Detection status | 0=no_data, 1=valid, 2=none |

---

## Robot Code Template

### Minimal Example

```python
from pymodbus.client import ModbusTcpClient

VISION_IP = "172.24.9.15"

# Connect
client = ModbusTcpClient(VISION_IP, port=502)
client.connect()

# Read configuration (once)
config = client.read_holding_registers(128, 6)
bay_id = config.registers[0]
grinder_id = config.registers[1]

# Read detection (in loop)
while True:
    detection = client.read_holding_registers(134, 3)
    
    x_unsigned = detection.registers[0]
    y_unsigned = detection.registers[1]
    status = detection.registers[2]
    
    # Convert to signed and scale
    x_mm = (x_unsigned if x_unsigned < 32768 else x_unsigned - 65536) / 10.0
    y_mm = (y_unsigned if y_unsigned < 32768 else y_unsigned - 65536) / 10.0
    
    if status == 1:  # Valid detection
        print(f"Grind at offset: X={x_mm:+.1f}mm, Y={y_mm:+.1f}mm")
        # YOUR ROBOT CODE HERE
        # movel(base_pos + [x_mm/1000, y_mm/1000, 0], speed, accel)
    
    time.sleep(0.2)  # 5 Hz
```

---

## Converting Values

### From Modbus Register to Real Value

```python
# Angle (register 130)
angle_degrees = register_value / 10.0
# Example: 455 → 45.5°

# Depth (register 131)
depth = register_value / 100.0
# Example: 125 → 1.25

# X/Y Position (registers 134, 135)
# Step 1: Convert unsigned to signed
if register_value > 32767:
    signed = register_value - 65536
else:
    signed = register_value

# Step 2: Scale
position_mm = signed / 10.0
# Example: 123 → +12.3mm
# Example: 65413 → -12.3mm (65413 - 65536 = -123, ÷10 = -12.3)
```

---

## During Operation

### Keyboard Commands (Vision System)

- **'c'** - Update configuration (prompts for new values)
- **'g'** - Update grinder position reference
- **'s'** - Save screenshot
- **'r'** - Reset detection
- **'q'** - Quit

### Changing Configuration

1. Press **'c'** on vision system
2. Enter new blade parameters
3. Configuration registers update automatically
4. Register 133 (version) increments
5. Robot detects change and re-reads config

---

## Workflow Example

### Processing Multiple Blades

```
┌─────────────────────────────────────────────────────────────┐
│ VISION SYSTEM                                               │
│                                                             │
│ 1. Start with Blade 1 config (Bay 1, Grinder 1)           │
│    → Modbus registers updated                              │
│                                                             │
│ 2. Robot connects and reads config                         │
│    → Starts grinding operations                            │
│                                                             │
│ 3. Operator finishes Blade 1, presses 'c'                 │
│    → Enters Blade 2 config (Bay 2, Grinder 1)             │
│    → Version increments: 1 → 2                             │
│                                                             │
│ 4. Robot detects version change                            │
│    → Re-reads configuration                                 │
│    → Continues with new parameters                          │
│                                                             │
│ 5. Repeat for Blade 3, 4, 5...                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `test_modbus.py` | Main vision system program |
| `modbus_blade_server.py` | Modbus server implementation |
| `robot_example.py` | Example robot code (Python) |
| `MODBUS_GUIDE.md` | Full documentation with all robot languages |
| `serrated_blade_detector.py` | Detection algorithm (unchanged) |

---

## Troubleshooting

### Problem: Cannot connect to Modbus server

**Check**:
1. Vision system is running
2. IP address is correct (default: 0.0.0.0 means all interfaces)
3. Port 502 is not blocked by firewall
4. Network connection is working

**Test**:
```bash
ping 172.24.9.15
```

### Problem: Reading zeros or strange values

**Check**:
1. Vision system has started detection (wait 5-10 seconds after startup)
2. Camera is connected and working
3. Blade is visible in camera view
4. Check register addresses (128-136)

**Test**:
```bash
python robot_example.py
```

### Problem: Status always 2 (no detection)

**Reasons**:
- Blade not in camera view
- Lighting conditions poor
- Grinder reference not set

**Fix**:
- Check camera feed on vision system
- Press 'g' to update grinder position
- Adjust blade position

---

## Network Setup

### Vision System IP

Default configuration listens on **all network interfaces** (0.0.0.0).

To set specific IP, edit `test_modbus.py`:

```python
system_config = {
    # ...
    'modbus_host': '172.24.9.15',  # Your specific IP
    'modbus_port': 502,
}
```

### Robot Network Configuration

Ensure robot and vision system are on same network or can route to each other.

**Example network**:
- Vision System: 172.24.9.15
- Robot: 172.24.9.89
- Subnet: 255.255.255.0

---

## Performance

### Recommended Read Rate

**5 Hz (every 200ms)** is recommended:
- Fast enough for real-time control
- Doesn't overload network
- Matches typical robot control cycle

```python
while True:
    detection = read_detection()
    # ... process
    time.sleep(0.2)  # 5 Hz
```

### Detection Update Rate

Vision system updates registers at **2 Hz** (every 500ms):
- Enough time for accurate detection
- Matches camera frame rate
- Robot reads faster than updates (no missed data)

---

## Summary

✅ **Connect**: `ModbusTcpClient(VISION_IP, 502)`  
✅ **Read Config**: Registers 128-133 (once at startup)  
✅ **Read Detection**: Registers 134-136 (continuously at 5 Hz)  
✅ **Convert Values**: Signed conversion + scaling  
✅ **Check Status**: Register 136 (1 = valid detection)  
✅ **Use Data**: Apply X/Y offsets to grinder position  

That's it! Standard Modbus, no custom protocol needed.
