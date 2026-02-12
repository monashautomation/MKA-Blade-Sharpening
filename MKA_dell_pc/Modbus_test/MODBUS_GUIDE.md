# Modbus Implementation Guide

## Overview

The system has been updated to use **Modbus TCP** protocol instead of custom TCP sockets. This is more standard and robot-friendly, allowing direct integration with industrial robot controllers.

---

## Why Modbus?

✅ **Industry Standard** - Widely supported by robot controllers  
✅ **Simpler Integration** - No custom protocol implementation needed  
✅ **Register-Based** - Robot reads data from registers like reading variables  
✅ **No Request/Response** - Robot reads whenever it wants  
✅ **Multiple Clients** - Multiple robots can read simultaneously  

---

## Architecture

```
┌─────────────────────┐
│  Vision Computer    │
│  (Blade Detection)  │
│                     │
│  ┌───────────────┐  │
│  │ Camera System │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼────────┐ │
│  │ Detection      │ │
│  │ Algorithm      │ │
│  └───────┬────────┘ │
│          │          │
│  ┌───────▼────────┐ │
│  │ Modbus Server  │ │◄─────── Robot reads registers
│  │ (Port 502)     │ │         anytime it wants
│  └────────────────┘ │
└─────────────────────┘
```

---

## Modbus Register Map

### Configuration Registers (Read by Robot)

| Register | Name | Description | Format | Example |
|----------|------|-------------|---------|---------|
| 128 | BAY_ID | Bay identifier | 1-10 | 5 |
| 129 | GRINDER_ID | Grinder identifier | 1-3 | 2 |
| 130 | ANGLE | Blade angle × 10 | Signed 16-bit | 455 = 45.5° |
| 131 | DEPTH | Grinding depth × 100 | Signed 16-bit | 125 = 1.25 |
| 132 | LENGTH | Blade length | Unsigned 16-bit | 150 |
| 133 | CONFIG_VERSION | Configuration version | Increments on update | 1, 2, 3... |

### Detection Data Registers (Read by Robot)

| Register | Name | Description | Format | Range |
|----------|------|-------------|---------|-------|
| 134 | DETECTION_X | X coordinate × 10 | Signed 16-bit (0.1mm) | ±3276.7mm |
| 135 | DETECTION_Y | Y coordinate × 10 | Signed 16-bit (0.1mm) | ±3276.7mm |
| 136 | STATUS | Detection status | 0/1/2 | See below |

**Status Values**:
- **0** = No data yet (system starting up)
- **1** = Valid detection available
- **2** = No detection in current frame

### Command Register (Written by Robot - Optional)

| Register | Name | Description | Values |
|----------|------|-------------|--------|
| 137 | COMMAND | Robot command | 0=idle, 10=read_config, 11=read_detection |

*Note: Command register is optional - robot can read anytime without sending commands*

---

## Reading Data from Robot

### Python Example

```python
from pymodbus.client import ModbusTcpClient

VISION_SYSTEM_IP = "172.24.9.15"

# Connect to Modbus server
client = ModbusTcpClient(VISION_SYSTEM_IP, port=502)
client.connect()

# Read configuration (6 registers starting at 128)
config_result = client.read_holding_registers(address=128, count=6)
bay_id = config_result.registers[0]
grinder_id = config_result.registers[1]
angle = config_result.registers[2] / 10.0
depth = config_result.registers[3] / 100.0
length = config_result.registers[4]
version = config_result.registers[5]

# Read detection data (3 registers starting at 134)
detection_result = client.read_holding_registers(address=134, count=3)
x_unsigned = detection_result.registers[0]
y_unsigned = detection_result.registers[1]
status = detection_result.registers[2]

# Convert unsigned to signed
x_mm = (x_unsigned if x_unsigned < 32768 else x_unsigned - 65536) / 10.0
y_mm = (y_unsigned if y_unsigned < 32768 else y_unsigned - 65536) / 10.0

print(f"Bay {bay_id}, Grinder {grinder_id}")
print(f"Detection: X={x_mm:+.1f}mm, Y={y_mm:+.1f}mm, Status={status}")

client.close()
```

### Universal Robots (UR) Script Example

```python
# UR Script for reading Modbus registers
def read_blade_detection():
    # Read configuration
    config = modbus_get_signal_status("bay_id", "128")  # Register 128
    bay_id = config[1]
    
    grinder_id = modbus_get_signal_status("grinder_id", "129")[1]
    angle_raw = modbus_get_signal_status("angle", "130")[1]
    angle = angle_raw / 10.0
    
    # Read detection data
    x_raw = modbus_get_signal_status("x_pos", "134")[1]
    y_raw = modbus_get_signal_status("y_pos", "135")[1]
    status = modbus_get_signal_status("status", "136")[1]
    
    # Convert to signed
    if x_raw > 32767:
        x_raw = x_raw - 65536
    end
    if y_raw > 32767:
        y_raw = y_raw - 65536
    end
    
    x_mm = x_raw / 10.0
    y_mm = y_raw / 10.0
    
    return [bay_id, grinder_id, angle, x_mm, y_mm, status]
end

# Main loop
while True:
    data = read_blade_detection()
    
    if data[5] == 1:  # Status = valid detection
        textmsg("Detection: X=", data[3], " Y=", data[4])
        # Move grinder to detected position
        # movel(p[base_x + data[3]/1000, base_y + data[4]/1000, base_z, 0, 0, 0], a=0.5, v=0.1)
    end
    
    sleep(0.2)  # Read at 5 Hz
end
```

### ABB RAPID Example

```rapid
MODULE BladeDetection
    
    ! Modbus configuration
    VAR num vision_ip{4} := [172, 24, 9, 15];
    CONST num modbus_port := 502;
    
    ! Detection data
    VAR num bay_id;
    VAR num grinder_id;
    VAR num angle;
    VAR num x_mm;
    VAR num y_mm;
    VAR num status;
    
    PROC ReadBladeConfig()
        VAR num registers{6};
        
        ! Read 6 registers starting at 128
        ModbusReadHoldingRegisters vision_ip, modbus_port, 128, 6, registers;
        
        bay_id := registers{1};
        grinder_id := registers{2};
        angle := registers{3} / 10.0;
        
        TPWrite "Config: Bay=" + NumToStr(bay_id,0) + " Grinder=" + NumToStr(grinder_id,0);
    ENDPROC
    
    PROC ReadBladeDetection()
        VAR num registers{3};
        
        ! Read 3 registers starting at 134
        ModbusReadHoldingRegisters vision_ip, modbus_port, 134, 3, registers;
        
        ! Convert to signed
        IF registers{1} > 32767 THEN
            x_mm := (registers{1} - 65536) / 10.0;
        ELSE
            x_mm := registers{1} / 10.0;
        ENDIF
        
        IF registers{2} > 32767 THEN
            y_mm := (registers{2} - 65536) / 10.0;
        ELSE
            y_mm := registers{2} / 10.0;
        ENDIF
        
        status := registers{3};
    ENDPROC
    
    PROC MainGrindingLoop()
        ReadBladeConfig;
        
        WHILE TRUE DO
            ReadBladeDetection;
            
            IF status = 1 THEN
                ! Valid detection - move to grind
                TPWrite "Detection: X=" + NumToStr(x_mm,1) + " Y=" + NumToStr(y_mm,1);
                ! MoveL grind_pos + Offs(x_mm, y_mm, 0), v100, fine, tool0;
            ENDIF
            
            WaitTime 0.2;  ! 5 Hz read rate
        ENDWHILE
    ENDPROC
    
ENDMODULE
```

### FANUC Karel Example

```pascal
PROGRAM BladeDetection
VAR
    bay_id : INTEGER
    grinder_id : INTEGER
    angle : REAL
    x_mm : REAL
    y_mm : REAL
    status : INTEGER
    registers : ARRAY[6] OF INTEGER
    
BEGIN
    -- Connect to Modbus server
    MB_CONNECT('172.24.9.15', 502)
    
    -- Read configuration (registers 128-133)
    MB_READ_REGS(128, 6, registers)
    bay_id = registers[1]
    grinder_id = registers[2]
    angle = registers[3] / 10.0
    
    -- Main loop
    WHILE TRUE DO
        -- Read detection data (registers 134-136)
        MB_READ_REGS(134, 3, registers)
        
        -- Convert to signed and scale
        IF registers[1] > 32767 THEN
            x_mm = (registers[1] - 65536) / 10.0
        ELSE
            x_mm = registers[1] / 10.0
        ENDIF
        
        IF registers[2] > 32767 THEN
            y_mm = (registers[2] - 65536) / 10.0
        ELSE
            y_mm = registers[2] / 10.0
        ENDIF
        
        status = registers[3]
        
        -- Process detection
        IF status = 1 THEN
            -- Valid detection
            WRITE('Detection: X=', x_mm, ' Y=', y_mm)
            -- L P[grind_pos] OFFSET,PR[offset] 100mm/sec FINE
        ENDIF
        
        DELAY 200  -- 5 Hz read rate
    ENDDO
    
    MB_DISCONNECT
END BladeDetection
```

---

## Workflow

### Robot Control Loop

```
1. START SYSTEM
   ├─ Connect to Modbus server (172.24.9.15:502)
   └─ Read configuration from registers 128-133

2. MAIN LOOP
   ├─ Check if config version changed (register 133)
   │  └─ If changed, re-read configuration
   │
   ├─ Read detection data from registers 134-136
   │
   ├─ Check status register (136)
   │  ├─ If status = 1 (valid detection)
   │  │  ├─ Read X and Y offsets
   │  │  ├─ Calculate grinder position
   │  │  ├─ Move to position
   │  │  └─ Execute grinding
   │  │
   │  ├─ If status = 2 (no detection)
   │  │  └─ Wait or retry
   │  │
   │  └─ If status = 0 (no data)
   │     └─ System starting, wait
   │
   └─ Repeat (recommended 5 Hz = 200ms interval)

3. SHUTDOWN
   └─ Disconnect from Modbus server
```

---

## Configuration Updates

### How It Works

When operator presses **'c'** key on vision system:
1. System prompts for new configuration
2. Configuration registers (128-133) are updated
3. **CONFIG_VERSION register (133) increments**
4. Robot detects version change
5. Robot re-reads configuration
6. Robot continues with new parameters

### Robot Code to Detect Updates

```python
last_version = 0

while True:
    # Read version register
    result = client.read_holding_registers(address=133, count=1)
    current_version = result.registers[0]
    
    if current_version != last_version:
        print("Configuration changed! Re-reading...")
        config = read_configuration(client)
        last_version = current_version
    
    # Continue normal operation
    detection = read_detection_data(client)
    # ... process detection
```

---

## Data Conversion Reference

### Signed 16-bit Conversion

Modbus uses unsigned 16-bit integers. For signed values:

```python
# Convert unsigned to signed
if value > 32767:
    signed_value = value - 65536
else:
    signed_value = value

# Or using bitwise operations
signed_value = value if value < 32768 else value - 65536

# Or compact
signed_value = (value ^ 0x8000) - 0x8000  # Flip sign bit
```

### Scaling Values

```python
# Read from Modbus
angle_raw = registers[2]      # e.g., 455
angle_degrees = angle_raw / 10.0   # = 45.5°

depth_raw = registers[3]      # e.g., 125
depth = depth_raw / 100.0     # = 1.25

x_raw = registers[0]          # e.g., 123
x_mm = x_raw / 10.0           # = 12.3mm
```

---

## Testing

### Test with Python Client

See `robot_example.py` for full examples:

```bash
python robot_example.py
```

This will:
1. Connect to Modbus server
2. Read configuration
3. Continuously read detection data
4. Detect configuration changes automatically

### Test with Modbus Simulator

```bash
# Terminal 1: Start vision system
python test_v2.py

# Terminal 2: Run robot example
python robot_example.py
```

---

## Network Configuration

### Default Settings

- **IP Address**: 0.0.0.0 (listens on all interfaces)
- **Port**: 502 (standard Modbus TCP port)
- **Protocol**: Modbus TCP

### Firewall Rules

Make sure port 502 is open:

```bash
# Linux
sudo ufw allow 502/tcp

# Windows
netsh advfirewall firewall add rule name="Modbus TCP" dir=in action=allow protocol=TCP localport=502
```

---

## Troubleshooting

### Connection Failed

**Problem**: Robot cannot connect to Modbus server

**Solutions**:
1. Check vision system IP address
2. Verify port 502 is not blocked by firewall
3. Ensure Modbus server is running (check vision system display)
4. Test with `ping 172.24.9.15`

### Reading Wrong Values

**Problem**: Values don't make sense

**Solutions**:
1. Check register addresses match (128-136)
2. Verify signed/unsigned conversion for X/Y
3. Ensure proper scaling (÷10 for angle/position, ÷100 for depth)
4. Check byte order (Modbus uses big-endian)

### Status Always 0

**Problem**: Status register shows 0 (no data)

**Solutions**:
1. Vision system may be starting up - wait 5-10 seconds
2. Check camera is connected and working
3. Verify blade detection is running (check vision system display)

---

## Advantages Over TCP Sockets

| Feature | Custom TCP | Modbus TCP |
|---------|-----------|------------|
| **Standard** | Custom protocol | Industry standard |
| **Robot Support** | Need custom code | Built-in support |
| **Complexity** | Request/response needed | Read anytime |
| **Multiple Readers** | Requires management | Native support |
| **Tools** | Custom debugging | Standard Modbus tools |
| **Documentation** | Need to write | Standard reference |

---

## Summary

✅ **Simple Integration** - Use standard Modbus client in robot  
✅ **Read Anytime** - No request/response protocol  
✅ **Configuration Updates** - Detected via version register  
✅ **Multiple Robots** - Can read simultaneously  
✅ **Standard Tools** - Use existing Modbus utilities  

The vision system continuously updates registers. Robot reads whenever it needs data. No complex protocol to implement!
