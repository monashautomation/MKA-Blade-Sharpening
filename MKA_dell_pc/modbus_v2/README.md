# 🤖 Modbus v2 - Robot Control System

A web-based dashboard for controlling robotic blade grinding operations via Modbus TCP protocol.

## 🌟 Features

- **Real-time Modbus TCP Communication** - Direct control of robot via Modbus registers
- **Web Dashboard** - User-friendly interface for robot control and monitoring
- **Configuration Management** - Send blade configurations (bay ID, grinder ID, angle, depth, length)
- **Detection Data Handling** - Send tooth coordinate data
- **Robot Control** - Start, stop, reset, and command the robot
- **Database Integration** - Store and retrieve data
- **Grinder Position Configuration** - JSON-based position settings

## 📋 System Architecture

```
┌─────────────────────────┐
│   Web Dashboard         │
│   (HTML)                │
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
| 20   | START_GRINDING   | Start grinding operation       |
| 21   | STOP             | Stop operation                 |
| 22   | RESET            | Reset robot to initial state   |

## 🚀 Installation

### Prerequisites

- Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
modbus_v2/
├── database.py                     # Database operations
├── grinder_position.json           # Grinder position configuration
├── index.html                      # Main dashboard page
├── modbus_blade_client_enhanced.py # Enhanced Modbus client
├── requirements.txt                # Python dependencies
├── robot_control_backend.py        # Flask backend server
├── robot_control_dashboard.html    # Robot control dashboard
└── README.md                       # This file
```

## 📖 Usage

### 1. Start the Flask Backend

```bash
python robot_control_backend.py
```

The server will start and be available at `http://localhost:5000`.

### 2. Open the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

Use the dashboard to connect to the robot, send configurations, and control operations.

### 3. Using the Modbus Client

You can also use the Modbus client directly:

```python
from modbus_blade_client_enhanced import BladeDataModbusClient

client = BladeDataModbusClient(host='172.24.89.89', port=502, unit=1)
client.connect()
# Use client methods to send data
```
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
