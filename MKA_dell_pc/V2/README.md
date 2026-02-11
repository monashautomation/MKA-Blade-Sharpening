# Restructured Blade Detection System

## Overview

This is a restructured version of the blade detection system with a new request-response protocol. The key changes are:

1. **CLI Configuration Input**: User enters blade parameters before starting
2. **Configuration Packet First**: Server sends config to client immediately on connection
3. **Request-Response Protocol**: Client requests data when ready (instead of continuous streaming)
4. **Compact Data Packets**: Reduced packet sizes for limited client storage

---

## Protocol Specification

### Connection Flow

```
1. Client connects to server
2. Server immediately sends CONFIGURATION PACKET (9 bytes)
3. Client processes configuration
4. Client sends REQUEST when ready (1 byte)
5. Server responds with DETECTION DATA (5 bytes) or NO DETECTION (1 byte)
6. Repeat steps 4-5 as needed
7. [NEW] Server can send NEW CONFIGURATION at any time (step 2 repeated)
8. Client receives and processes new config, continues requesting data
```

---

## Packet Formats

### 1. Configuration Packet (CMD=0)
**Direction**: Server → Client  
**Size**: 9 bytes  
**Sent**: Once, immediately on connection

```
Byte 0:     CMD = 0 (Configuration)
Byte 1:     BAY_ID (1-10)
Byte 2:     GRINDER_ID (1-3)
Bytes 3-4:  ANGLE (signed 16-bit, value*10, e.g., 45.5° = 455)
Bytes 5-6:  DEPTH (signed 16-bit, value*100, e.g., 1.25 = 125)
Bytes 7-8:  LENGTH (unsigned 16-bit, 0-999)
```

**Example**:
```python
# Bay=5, Grinder=2, Angle=45.5°, Depth=1.25, Length=150
[0x00, 0x05, 0x02, 0x01, 0xC7, 0x00, 0x7D, 0x00, 0x96]
```

### 2. Data Request (CMD=3)
**Direction**: Client → Server  
**Size**: 1 byte  
**Sent**: Whenever client needs detection data

```
Byte 0:     CMD = 3 (Request data)
```

### 3. Detection Data (CMD=1)
**Direction**: Server → Client  
**Size**: 5 bytes  
**Sent**: In response to CMD=3 request, when detection available

```
Byte 0:     CMD = 1 (Detection available)
Bytes 1-2:  X coordinate (signed 16-bit, in 0.1mm units)
Bytes 3-4:  Y coordinate (signed 16-bit, in 0.1mm units)
```

**Range**: ±3276.7mm per axis  
**Resolution**: 0.1mm

**Example**:
```python
# X=12.5mm, Y=-8.3mm
# X_value = 125 (12.5 * 10)
# Y_value = -83 (-8.3 * 10)
[0x01, 0x00, 0x7D, 0xFF, 0xAD]
```

### 4. No Detection (CMD=2)
**Direction**: Server → Client  
**Size**: 1 byte  
**Sent**: In response to CMD=3 request, when no detection available

```
Byte 0:     CMD = 2 (No detection)
```

---

## File Structure

### New Files

1. **tcp_blade_server_v2.py**
   - New TCP server with request-response protocol
   - Configuration packet support
   - Compact 5-byte detection packets (vs. old 9-byte)

2. **test_v2.py**
   - Restructured main detection program
   - CLI configuration input on startup
   - Uses new server protocol

3. **test_client.py**
   - Example client implementation
   - Demonstrates full protocol flow
   - Continuous request mode

### Original Files (Unchanged)

- **serrated_blade_detector.py**: Core detection algorithm
- **tcp_blade_server.py**: Original continuous streaming server
- **test.py**: Original main program

---

## Usage

### 1. Start the Server

```bash
python test_v2.py
```

**You will be prompted for configuration**:
```
Enter BAY ID (1-10): 5
Enter GRINDER ID (1-3): 2
Enter ANGLE (degrees, e.g., 45.5): 45.5
Enter DEPTH (e.g., 1.25): 1.25
Enter LENGTH (0-999): 150
```

**The system will**:
- Load camera
- Start TCP server on 172.24.9.15:5000
- Wait for client connections
- Send configuration to connected clients
- Process detection requests

**During operation, press 'c' to update configuration**:
- System will prompt for new blade parameters
- New configuration is broadcasted to all connected clients
- Clients automatically receive and process the update
- No reconnection needed!

### 2. Connect Client

```bash
python test_client.py
```

**The client will**:
1. Connect to server
2. Receive configuration packet
3. Display configuration
4. Wait for user to press Enter
5. Start requesting detection data at 5 Hz

---

## Key Improvements

### 1. Reduced Packet Size
- **Old system**: 9 bytes per detection (CMD + X(4) + Y(4))
- **New system**: 5 bytes per detection (CMD + X(2) + Y(2))
- **Savings**: 44% reduction in data size

### 2. Request-Response Model
- Client controls when data is sent
- No continuous streaming overhead
- Server doesn't waste bandwidth on unused data
- Client buffer management is easier

### 3. Immediate Configuration
- Client receives blade parameters before requesting data
- Can validate/store configuration before processing
- No need to request configuration separately

### 4. Better Resolution/Range Tradeoff
- **Old**: 0.01mm resolution, ±21474mm range (excessive)
- **New**: 0.1mm resolution, ±3276mm range (practical)
- **Benefit**: Half the bytes, still more than enough precision

### 5. Dynamic Configuration Updates
- **NEW**: Server can broadcast new configurations mid-operation
- Client automatically receives and processes updates
- No need to disconnect/reconnect
- Perfect for multi-blade workflows (finish blade 1, move to blade 2)
- Press 'c' key on server to send new configuration to all clients

---

## Data Format Decoding

### Python Example (Client Side)

```python
import socket

# Receive configuration
config_data = sock.recv(9)
cmd = config_data[0]  # Should be 0
bay_id = config_data[1]
grinder_id = config_data[2]
angle = int.from_bytes(config_data[3:5], 'big', signed=True) / 10.0
depth = int.from_bytes(config_data[5:7], 'big', signed=True) / 100.0
length = int.from_bytes(config_data[7:9], 'big', signed=False)

# Request detection data
sock.sendall(b'\x03')  # CMD=3

# Receive detection
response = sock.recv(1)
cmd = response[0]

if cmd == 1:  # Detection available
    coords = sock.recv(4)
    x_mm = int.from_bytes(coords[0:2], 'big', signed=True) / 10.0
    y_mm = int.from_bytes(coords[2:4], 'big', signed=True) / 10.0
    print(f"Position: X={x_mm}mm, Y={y_mm}mm")
elif cmd == 2:  # No detection
    print("No detection available")
```

### Arduino/Microcontroller Example

```cpp
// Receive configuration (9 bytes)
uint8_t config[9];
Serial.readBytes(config, 9);

uint8_t cmd = config[0];        // Should be 0
uint8_t bay_id = config[1];     // 1-10
uint8_t grinder_id = config[2]; // 1-3

int16_t angle_raw = (config[3] << 8) | config[4];
float angle = angle_raw / 10.0;

int16_t depth_raw = (config[5] << 8) | config[6];
float depth = depth_raw / 100.0;

uint16_t length = (config[7] << 8) | config[8];

// Request data
Serial.write(0x03);  // CMD=3

// Read response
uint8_t response_cmd;
Serial.readBytes(&response_cmd, 1);

if (response_cmd == 1) {  // Detection
    uint8_t coords[4];
    Serial.readBytes(coords, 4);
    
    int16_t x_raw = (coords[0] << 8) | coords[1];
    int16_t y_raw = (coords[2] << 8) | coords[3];
    
    float x_mm = x_raw / 10.0;
    float y_mm = y_raw / 10.0;
}
```

---

## Configuration Parameters

### Bay ID (1-10)
Identifies which grinding bay this blade is in.

### Grinder ID (1-3)
Identifies which grinder head within the bay.

### Angle (degrees)
Blade angle in degrees (e.g., 45.5°)
- Resolution: 0.1°
- Range: -3276.8° to +3276.7°

### Depth
Grinding depth setting
- Resolution: 0.01
- Range: -327.68 to +327.67

### Length (0-999)
Blade length parameter
- Integer value 0-999

---

## Error Handling

### Client Side
```python
try:
    # Request data
    sock.sendall(b'\x03')
    
    # Set timeout
    sock.settimeout(1.0)
    
    response = sock.recv(1)
    if len(response) == 0:
        print("Connection lost")
        # Reconnect
        
except socket.timeout:
    print("Server not responding")
    # Retry or reconnect
```

### Server Side
- Validates all configuration inputs (range checks)
- Handles disconnections gracefully
- Continues serving other clients if one fails

---

## Comparison: Old vs New

| Feature | Old System | New System |
|---------|-----------|-----------|
| **Packet Size** | 9 bytes | 5 bytes |
| **Protocol** | Continuous push | Request-response |
| **Configuration** | Hardcoded | User input (CLI) |
| **Config Transfer** | None | 9-byte packet |
| **Data Resolution** | 0.01mm | 0.1mm |
| **Client Control** | No | Yes (requests data) |
| **Buffer Friendly** | No | Yes (small packets) |

---

## Multiple Configuration Workflow

### Typical Use Case: Processing Multiple Blades

1. **Start System** with first blade configuration:
   ```
   Bay=1, Grinder=1, Angle=45.0°, Depth=1.50, Length=100
   ```

2. **Client Connects** and receives initial config

3. **Process First Blade** - Client requests detection data repeatedly

4. **Blade Finished** - Operator presses 'c' on server

5. **Enter New Configuration**:
   ```
   Bay=2, Grinder=1, Angle=30.0°, Depth=1.25, Length=120
   ```

6. **Client Automatically Receives** new config (no reconnection!)

7. **Continue Processing** - Client now uses new parameters

8. **Repeat** as needed for additional blades

### Server-Side Commands

While running, you can press:
- **'c'** - Update configuration (prompts for new values, broadcasts to all clients)
- **'g'** - Update grinder position
- **'s'** - Save current frame
- **'r'** - Reset detection
- **'q'** - Quit

### Client-Side Behavior

The client automatically:
- Detects new configuration packets
- Displays update notification
- Continues operating with new parameters
- Tracks number of configuration updates received

---

## Testing

### Test Without Hardware

Run the server in test mode:
```bash
python tcp_blade_server_v2.py
```

This starts a standalone server with test data.

### Full System Test

1. **Terminal 1**: Start detection system
   ```bash
   python test_v2.py
   ```

2. **Terminal 2**: Start test client
   ```bash
   python test_client.py
   ```

3. Monitor both terminals for:
   - Configuration transmission
   - Data requests
   - Detection responses

---

## Troubleshooting

### Client Not Receiving Configuration
- Check server started successfully
- Verify IP address (172.24.9.15)
- Ensure port 5000 is not blocked
- Check network connectivity

### No Detection Data
- Verify camera is working (check OpenCV window)
- Press 'g' to update grinder position
- Check blade is in frame
- Verify grinder_position.json exists

### Data Corruption
- Ensure TCP_NODELAY is set
- Check both ends use same byte order (big-endian)
- Verify packet sizes match specification

---

## Future Enhancements

Possible improvements:
1. **CRC/Checksum**: Add error detection to packets
2. **Multiple Detections**: Send array of all teeth positions
3. **Timestamp**: Include detection timestamp
4. **Quality Metric**: Send confidence score with detection
5. **Compression**: Further reduce packet size if needed

---

## Network Configuration

- **Default Host**: 172.24.9.15
- **Default Port**: 5000
- **Protocol**: TCP/IP
- **Byte Order**: Big-endian
- **No Nagle**: TCP_NODELAY enabled for immediate transmission

To change network settings, edit `test_v2.py`:
```python
system_config = {
    'tcp_host': '192.168.1.100',  # Your IP
    'tcp_port': 5000,              # Your port
    # ...
}
```

---

## License

Same license as original blade detection system.