# 🤖 Blade Grinder Robot Control System

A modern, industrial-themed web dashboard for controlling robotic blade grinding operations via Modbus TCP protocol.

## 🌟 Features

- **FLIR Camera Integration** - Professional PySpin SDK for high-quality image capture
- **Real-time Modbus TCP Communication** - Direct control of robot via Modbus registers
- **Beautiful Industrial UI** - Cyberpunk-inspired dashboard with real-time status updates
- **Automatic Blade Detection** - AI-powered tooth detection from live camera feed
- **Configuration Management** - Send blade configurations (bay ID, grinder ID, angle, depth, length)
- **Detection Data** - Automatic tooth coordinate extraction and transmission
- **Robot Control** - Start, stop, reset, and command the robot
- **Live System Log** - Real-time monitoring of all operations
- **Connection Management** - Easy connect/disconnect with status indicators

## 📋 System Architecture

```
┌─────────────────────────┐
│   Web Dashboard         │
│   (HTML/CSS/JS)         │
└───────────┬─────────────┘
            │ HTTP/REST API
            ▼
┌─────────────────────────┐
│   Flask Backend         │
│   (Python)              │
└───────────┬─────────────┘
            │ Modbus TCP
            ▼
┌─────────────────────────┐
│   Robot Controller      │
│   (Modbus Server)       │
└─────────────────────────┘
```

## 🔧 Modbus Register Map

| Register | Name            | Description                        | Data Type      |
|----------|-----------------|------------------------------------|-----------------|
| 128      | BAY_ID          | Bay identification number          | INT16          |
| 129      | GRINDER_ID      | Grinder identification number      | INT16          |
| 130      | ANGLE           | Grinding angle (× 10)              | INT16          |
| 131      | DEPTH           | Grinding depth (× 100)             | INT16          |
| 132      | LENGTH          | Blade length (mm)                  | INT16          |
| 133      | CONFIG_VERSION  | Configuration version              | INT16          |
| 134      | DETECTION_X     | X coordinate (× 10, unsigned)      | UINT16         |
| 135      | DETECTION_Y     | Y coordinate (× 10, unsigned)      | UINT16         |
| 136      | STATUS          | Detection status (0=invalid, 1=valid) | INT16       |
| 137      | COMMAND         | Command register                   | INT16          |
| 138      | START           | Start/Stop register (1=start, 0=stop) | INT16       |

## 🎮 Command Codes

| Code | Command          | Description                    |
|------|------------------|--------------------------------|
| 11   | READ_DETECTION   | Read detection data            |
| 20   | START_GRINDING   | Start grinding operation       |
| 21   | STOP             | Stop operation                 |
| 22   | RESET            | Reset robot to initial state   |

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
python --version
```

### Install PySpin (FLIR Spinnaker SDK)

**IMPORTANT:** PySpin must be installed before other dependencies.

1. **Download Spinnaker SDK** from FLIR's website:
   - Visit: https://www.flir.com/products/spinnaker-sdk/
   - Download the appropriate version for your OS
   - Install the Spinnaker SDK first

2. **Install PySpin Python module:**
   ```bash
   # The PySpin wheel is included in the Spinnaker SDK installation
   # Location varies by OS:
   # Linux: /opt/spinnaker/lib/
   # Windows: C:\Program Files\FLIR Systems\Spinnaker\
   
   # Install the .whl file:
   pip install /path/to/spinnaker_python-*.whl
   ```

### Install Other Dependencies

```bash
pip install pymodbus flask flask-cors numpy opencv-python scipy
```

### Project Structure

```
blade-grinder-control/
├── modbus_blade_client_enhanced.py  # Enhanced Modbus client with start register
├── robot_control_backend.py          # Flask backend server (requires HTML in same dir)
├── robot_control_standalone.py       # Standalone server (all-in-one, recommended)
├── robot_control_dashboard.html      # Web dashboard UI
├── serrated_blade_detector.py        # Blade detection system
└── README.md                         # This file
```

### Important: File Placement

**Make sure both files are in the same directory:**
- `robot_control_standalone.py` (or `robot_control_backend.py`)
- `robot_control_dashboard.html`

The server looks for the HTML file in the same folder where you run the Python script.

## 📖 Usage

### 1. Start the Flask Backend

**Option A: Standalone Server (Recommended)**
```bash
python robot_control_standalone.py
```

**Option B: Modular Backend**
```bash
python robot_control_backend.py
```

You should see:
```
================================================================================
🤖 BLADE GRINDER CONTROL SYSTEM - BACKEND SERVER
================================================================================

✓ Starting Flask server...
✓ Dashboard will be available at: http://localhost:5000
✓ API endpoints ready

================================================================================
```

**Important:** The HTML file must be in the same directory as the Python script!

### 2. Open the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

### 3. Connect to Robot

1. Enter the robot's IP address (default: `172.24.89.89`)
2. Enter the Modbus port (default: `502`)
3. Enter the unit ID (default: `1`)
4. Click **CONNECT TO ROBOT**

### 4. Send Configuration

1. Fill in the blade configuration:
   - Bay ID
   - Grinder ID
   - Angle (degrees)
   - Depth (mm)
   - Length (mm)
   - Config Version
2. Click **SEND CONFIGURATION**

### 5. Send Detection Data

1. Enter the detected tooth coordinates:
   - X Position (mm)
   - Y Position (mm)
   - Detection Status (0=invalid, 1=valid)
2. Click **SEND DETECTION DATA**

### 6. Control Robot

- **READ** - Read detection data from robot
- **RESET** - Reset robot to initial state
- **STOP** - Emergency stop
- **START ROBOT OPERATION** - Begin grinding operation

## 🔌 Standalone Modbus Client Usage

You can also use the enhanced Modbus client directly:

```python
from modbus_blade_client_enhanced import BladeDataModbusClient

# Create and connect client
client = BladeDataModbusClient(host='172.24.89.89', port=502, unit=1)
client.connect()

# Send configuration
client.write_configuration(
    bay_id=5,
    grinder_id=2,
    angle=45.5,
    depth=1.25,
    length=150,
    config_version=1
)

# Send detection data
client.write_detection(x_mm=2.5, y_mm=1.8, status=1)

# Start robot
client.start_robot()

# Stop robot
client.stop_robot()

# Close connection
client.close()
```

## 🎨 Dashboard Features

### Status Bar
- **Connection Status** - Shows if connected to robot
- **Robot State** - Current operation state (IDLE, RUNNING, PAUSED)
- **Last Command** - Most recent command sent

### Control Panels
1. **Connection Settings** - Configure robot IP, port, unit ID
2. **Blade Configuration** - Set blade parameters
3. **Detection Data** - Send tooth coordinates
4. **Robot Control** - Start/stop/command robot
5. **System Log** - Real-time activity monitoring

### Visual Indicators
- 🟢 Green - Connected/Running
- 🔴 Red - Disconnected/Error
- 🔵 Blue - Active operation
- 🟡 Yellow - Warning/Paused

## 🔐 Safety Features

- Connection status validation before any operation
- Error handling and logging
- Emergency stop button
- Reset functionality
- Clear visual feedback for all operations

## 🛠️ API Endpoints

| Method | Endpoint           | Description              |
|--------|-------------------|--------------------------|
| POST   | /api/connect      | Connect to robot         |
| POST   | /api/disconnect   | Disconnect from robot    |
| GET    | /api/status       | Get connection status    |
| POST   | /api/configuration| Send blade config        |
| POST   | /api/detection    | Send detection data      |
| POST   | /api/command      | Send command             |
| POST   | /api/start        | Start robot              |
| POST   | /api/stop         | Stop robot               |

## 📊 Data Conversion

### Angle
- Input: Decimal degrees (e.g., 45.5°)
- Modbus: Integer × 10 (e.g., 455)

### Depth
- Input: Millimeters (e.g., 1.25mm)
- Modbus: Integer × 100 (e.g., 125)

### Coordinates (X, Y)
- Input: Millimeters with sign (e.g., -2.5mm)
- Modbus: Unsigned 16-bit × 10 (e.g., 65511)
- Negative values converted: `65536 + (value × 10)`

## 🐛 Troubleshooting

### TemplateNotFound Error
**Error:** `jinja2.exceptions.TemplateNotFound: robot_control_dashboard.html`

**Solution:** 
1. Make sure `robot_control_dashboard.html` is in the **same directory** as the Python server file
2. Use `robot_control_standalone.py` instead of `robot_control_backend.py`
3. Check the server output - it shows the current working directory
4. Run the server from the directory containing the HTML file:
   ```bash
   cd /path/to/your/files
   python robot_control_standalone.py
   ```

### Cannot Connect to Robot
1. Verify robot IP address is correct
2. Check network connectivity
3. Ensure Modbus port 502 is open
4. Verify robot Modbus server is running

### Commands Not Working
1. Check connection status indicator
2. Review system log for errors
3. Verify register addresses match robot configuration
4. Ensure data types and conversions are correct

### Dashboard Not Loading
1. Verify Flask backend is running
2. Check browser console for errors
3. Ensure port 5000 is not blocked
4. Try accessing from localhost first

## 📝 Notes

- The dashboard uses async/await for non-blocking API calls
- All Modbus operations include error checking
- Connection state is maintained between operations
- System log provides detailed operation history

## 🔮 Future Enhancements

- [ ] Real-time register monitoring
- [ ] Batch detection data upload
- [ ] Configuration presets/templates
- [ ] Data visualization (charts/graphs)
- [ ] Multi-robot support
- [ ] Historical operation logging
- [ ] User authentication
- [ ] Mobile-responsive design

## 📄 License

This project is part of the Blade Grinder Control System.

## 👨‍💻 Author

Created for industrial blade grinding automation.

---

**⚠️ Safety Warning**: Always follow proper safety procedures when operating industrial robotics equipment. Use emergency stop procedures when needed.
