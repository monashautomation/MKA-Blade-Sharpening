#!/usr/bin/env python3
"""
Real-Time Serrated Blade Detection System with OPC UA Integration
Uses FLIR USB Camera with PySpin driver
Writes shortest distance groove X,Y offsets to OPC UA server INT1, INT2
Python 3.10
"""

import cv2
import numpy as np
import PySpin
import time
from datetime import datetime
from threading import Thread, Lock
from serrated_blade_detector import SerratedBladeAnalyzer
import sys

# OPC UA imports
try:
    from opcua import Client, ua

    OPCUA_AVAILABLE = True
except ImportError:
    print("WARNING: opcua library not installed. OPC UA features disabled.")
    print("Install with: pip install opcua")
    OPCUA_AVAILABLE = False


class RealtimeBladeDetector:
    """Real-time blade detection using FLIR camera with OPC UA integration"""

    def __init__(self, config):
        """
        Initialize the real-time detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.frame = None
        self.frame_lock = Lock()
        self.latest_results = None
        self.results_lock = Lock()

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Camera system
        self.system = None
        self.cam = None
        self.cam_list = None

        # OPC UA client
        self.opcua_client = None
        self.opcua_connected = False
        self.int1_node = None
        self.int2_node = None
        self.opcua_lock = Lock()
        self.last_x_offset = None
        self.last_y_offset = None

    def initialize_opcua(self):
        """Initialize OPC UA client connection"""
        if not OPCUA_AVAILABLE:
            print("OPC UA library not available. Skipping OPC UA initialization.")
            return False

        if 'opcua_server_url' not in self.config:
            print("OPC UA server URL not configured. Skipping OPC UA initialization.")
            return False

        print("Initializing OPC UA connection...")

        try:
            # Create OPC UA client
            server_url = self.config['opcua_server_url']
            self.opcua_client = Client(server_url)

            # Set timeout
            self.opcua_client.session_timeout = 30000

            # Connect
            self.opcua_client.connect()
            print(f"✓ Connected to OPC UA server: {server_url}")

            # Get node IDs for INT1 and INT2
            int1_node_id = self.config.get('opcua_int1_node', 'ns=21;s=int1')
            int2_node_id = self.config.get('opcua_int2_node', 'ns=21;s=int2')

            # Get node references
            self.int1_node = self.opcua_client.get_node(int1_node_id)
            self.int2_node = self.opcua_client.get_node(int2_node_id)

            # Verify nodes are accessible
            self.int1_node.get_value()
            self.int2_node.get_value()

            print(f"✓ INT1 node connected: {int1_node_id}")
            print(f"✓ INT2 node connected: {int2_node_id}")

            self.opcua_connected = True
            return True

        except Exception as e:
            print(f"ERROR: Could not connect to OPC UA server: {e}")
            print(f"  Server URL: {self.config.get('opcua_server_url', 'Not configured')}")
            print(f"  Make sure your OPC UA server is running and accessible.")
            self.opcua_connected = False
            return False

    def write_to_opcua(self, x_offset, y_offset):
        """
        Write X and Y offsets to OPC UA server

        Args:
            x_offset: X offset in mm (will be converted to int)
            y_offset: Y offset in mm (will be converted to int)
        """
        if not self.opcua_connected or self.opcua_client is None:
            return

        try:
            with self.opcua_lock:
                # Convert to integers (round to nearest mm)
                x_int = np.int32(round(x_offset*10))

                y_int = np.int32(round(y_offset*10))


                # Write to OPC UA server
                self.int1_node.set_value(x_int,ua.VariantType.Int32)
                self.int2_node.set_value(y_int,ua.VariantType.Int32)


                # Store last written values
                self.last_x_offset = x_int
                self.last_y_offset = y_int

                # Debug output (optional - comment out if too verbose)
                # print(f"OPC UA: Written INT1={x_int}, INT2={y_int}")

        except Exception as e:
            print(f"ERROR writing to OPC UA: {e}")
            # Try to reconnect
            self.opcua_connected = False

    def initialize_camera(self):
        """Initialize FLIR camera using PySpin"""
        print("Initializing FLIR camera...")

        try:
            # Retrieve singleton reference to system object
            self.system = PySpin.System.GetInstance()

            # Retrieve list of cameras from the system
            self.cam_list = self.system.GetCameras()

            num_cameras = self.cam_list.GetSize()
            print(f"Number of cameras detected: {num_cameras}")

            if num_cameras == 0:
                print("ERROR: No cameras detected!")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                return False

            # Use first camera
            self.cam = self.cam_list[0]

            # Initialize camera
            self.cam.Init()

            # Configure camera settings
            self._configure_camera()

            # Begin acquiring images
            self.cam.BeginAcquisition()

            print("✓ Camera initialized successfully!")
            return True

        except PySpin.SpinnakerException as ex:
            print(f"ERROR: {ex}")
            return False

    def _configure_camera(self):
        """Configure camera settings for optimal detection"""
        print("Configuring camera settings...")

        try:
            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(
                self.cam.GetNodeMap().GetNode('AcquisitionMode')
            )
            if PySpin.IsAvailable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode):
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if PySpin.IsAvailable(node_acquisition_mode_continuous) and PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                    print("  ✓ Acquisition mode set to continuous")

            # Set frame rate if specified
            if 'frame_rate' in self.config:
                try:
                    # Enable manual frame rate control
                    node_acquisition_framerate_enable = PySpin.CBooleanPtr(
                        self.cam.GetNodeMap().GetNode('AcquisitionFrameRateEnable')
                    )
                    if PySpin.IsAvailable(node_acquisition_framerate_enable) and PySpin.IsWritable(
                            node_acquisition_framerate_enable):
                        node_acquisition_framerate_enable.SetValue(True)

                    # Set frame rate
                    node_acquisition_framerate = PySpin.CFloatPtr(
                        self.cam.GetNodeMap().GetNode('AcquisitionFrameRate')
                    )
                    if PySpin.IsAvailable(node_acquisition_framerate) and PySpin.IsWritable(node_acquisition_framerate):
                        framerate_to_set = min(
                            node_acquisition_framerate.GetMax(),
                            self.config['frame_rate']
                        )
                        node_acquisition_framerate.SetValue(framerate_to_set)
                        print(f"  ✓ Frame rate set to {framerate_to_set} fps")
                except:
                    print("  ⚠ Could not set frame rate (camera may not support it)")

            # Set exposure time if specified
            if 'exposure_time' in self.config:
                try:
                    node_exposure_auto = PySpin.CEnumerationPtr(
                        self.cam.GetNodeMap().GetNode('ExposureAuto')
                    )
                    if PySpin.IsAvailable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
                        node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
                        if PySpin.IsAvailable(node_exposure_auto_off) and PySpin.IsReadable(node_exposure_auto_off):
                            node_exposure_auto.SetIntValue(node_exposure_auto_off.GetValue())

                    node_exposure_time = PySpin.CFloatPtr(
                        self.cam.GetNodeMap().GetNode('ExposureTime')
                    )
                    if PySpin.IsAvailable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
                        exposure_time_to_set = min(
                            node_exposure_time.GetMax(),
                            self.config['exposure_time']
                        )
                        node_exposure_time.SetValue(exposure_time_to_set)
                        print(f"  ✓ Exposure time set to {exposure_time_to_set} μs")
                except:
                    print("  ⚠ Could not set exposure time")

            # Set gain if specified
            if 'gain' in self.config:
                try:
                    node_gain_auto = PySpin.CEnumerationPtr(
                        self.cam.GetNodeMap().GetNode('GainAuto')
                    )
                    if PySpin.IsAvailable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
                        node_gain_auto_off = node_gain_auto.GetEntryByName('Off')
                        if PySpin.IsAvailable(node_gain_auto_off) and PySpin.IsReadable(node_gain_auto_off):
                            node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())

                    node_gain = PySpin.CFloatPtr(
                        self.cam.GetNodeMap().GetNode('Gain')
                    )
                    if PySpin.IsAvailable(node_gain) and PySpin.IsWritable(node_gain):
                        gain_to_set = min(node_gain.GetMax(), self.config['gain'])
                        node_gain.SetValue(gain_to_set)
                        print(f"  ✓ Gain set to {gain_to_set} dB")
                except:
                    print("  ⚠ Could not set gain")

        except PySpin.SpinnakerException as ex:
            print(f"ERROR configuring camera: {ex}")

    def capture_thread(self):
        """Thread for continuously capturing frames"""
        print("Starting capture thread...")

        while self.running:
            try:
                # Retrieve next received image
                image_result = self.cam.GetNextImage(1000)  # 1000ms timeout

                if image_result.IsIncomplete():
                    print(f"Image incomplete with status {image_result.GetImageStatus()}")
                else:
                    # Convert to numpy array
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()

                    # Get pixel format
                    pixel_format = image_result.GetPixelFormat()

                    # Convert based on pixel format
                    if pixel_format == PySpin.PixelFormat_Mono8:
                        # Grayscale image - convert directly
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                    elif pixel_format == PySpin.PixelFormat_BayerRG8:
                        # Bayer format - convert to BGR
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2BGR)
                    elif pixel_format == PySpin.PixelFormat_BayerGR8:
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GR2BGR)
                    elif pixel_format == PySpin.PixelFormat_BayerGB8:
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GB2BGR)
                    elif pixel_format == PySpin.PixelFormat_BayerBG8:
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)
                    elif pixel_format == PySpin.PixelFormat_RGB8:
                        # RGB - convert to BGR
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    elif pixel_format == PySpin.PixelFormat_BGR8:
                        # Already BGR
                        frame = image_result.GetNDArray()
                    else:
                        # Try to use processor for conversion
                        try:
                            processor = PySpin.ImageProcessor()
                            processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
                            image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                            frame = image_converted.GetNDArray()
                        except:
                            # Fallback: just get raw data and convert to grayscale then BGR
                            print(f"Warning: Unknown pixel format {pixel_format}, using grayscale conversion")
                            image_data = image_result.GetNDArray().reshape((height, width))
                            frame = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # Update frame
                    with self.frame_lock:
                        self.frame = frame.copy()
                        self.frame_count += 1

                # Release image
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print(f"Capture error: {ex}")
                time.sleep(0.1)
            except Exception as ex:
                print(f"Unexpected capture error: {ex}")
                time.sleep(0.1)

    def detection_thread(self):
        """Thread for running blade detection"""
        print("Starting detection thread...")

        detection_interval = 1.0 / self.config['detection_fps']
        last_detection_time = 0

        while self.running:
            current_time = time.time()

            # Check if it's time for detection
            if current_time - last_detection_time >= detection_interval:
                with self.frame_lock:
                    if self.frame is not None:
                        frame_to_process = self.frame.copy()
                    else:
                        time.sleep(0.01)
                        continue

                # Run detection
                try:
                    # Save frame temporarily
                    temp_path = '/tmp/temp_blade_frame.png'
                    cv2.imwrite(temp_path, frame_to_process)

                    # Create analyzer and run detection
                    analyzer = SerratedBladeAnalyzer(temp_path)
                    analyzer.preprocess_image()
                    analyzer.detect_edges()
                    analyzer.find_blade_contours()
                    analyzer.detect_blade_and_grinder()

                    if hasattr(analyzer, "blade_edge_points") and analyzer.blade_edge_points is not None and len(
                            analyzer.blade_edge_points) > 2:
                        pts = np.array(analyzer.blade_edge_points)

                        # Fit a straight line (least squares)
                        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

                        # Compute angle from Y-axis (in degrees)
                        angle_from_x = np.arctan2(vy, vx)
                        angle_from_y = np.pi / 2 - angle_from_x
                        angle_deg = np.degrees(angle_from_y)

                        blade_angle_info = {
                            'vx': float(vx),
                            'vy': float(vy),
                            'x0': float(x0),
                            'y0': float(y0),
                            'angle_from_y_deg': float(angle_deg)
                        }
                    else:
                        blade_angle_info = None

                    analyzer.teeth_profiles = analyzer.extract_tooth_profiles(
                        window_size=self.config['window_size'],
                        min_depth_px=self.config['min_depth_px']
                    )

                    # Calculate coordinates
                    grinding_coords = analyzer.generate_grinding_coordinates(
                        pixels_per_mm=self.config['pixels_per_mm']
                    )

                    # Find groove with shortest distance and write to OPC UA
                    if grinding_coords and self.opcua_connected:
                        valid_coords = [g for g in grinding_coords if g['move_y_mm'] > 0.2]
                        if valid_coords:
                            shortest_groove = min(valid_coords, key=lambda g: g['distance_to_grinder_mm'])

                            x_coord_mm = shortest_groove['move_x_mm']
                            y_coord_mm = shortest_groove['move_y_mm']

                            # Example: print or write Y to OPC UA
                            print(f"Shortest groove Y (mm): {y_coord_mm}")

                            # Write both coordinates to OPC UA
                            self.write_to_opcua(x_coord_mm, y_coord_mm)
                        else:
                            print("No valid grooves found above 0.2 mm")
                    # Store results
                    results = {
                        'timestamp': datetime.now(),
                        'num_grooves': len(analyzer.teeth_profiles),
                        'grinding_coordinates': grinding_coords,
                        'grinder_tip': analyzer.grinder_tip,
                        'blade_edge_points': analyzer.blade_edge_points,
                        'teeth_profiles': analyzer.teeth_profiles,
                        'blade_angle': blade_angle_info
                    }

                    with self.results_lock:
                        self.latest_results = results

                except Exception as e:
                    print(f"Detection error: {e}")

                last_detection_time = current_time
            else:
                time.sleep(0.01)

    def draw_overlay(self, frame):
        """Draw detection overlay on frame"""
        overlay = frame.copy()

        with self.results_lock:
            if self.latest_results is None:
                return overlay

            results = self.latest_results

        try:
            # Draw grinder tip
            if results['grinder_tip'] is not None:
                tip = results['grinder_tip']
                cv2.circle(overlay, tip, 12, (0, 255, 255), 3)
                cv2.circle(overlay, tip, 15, (0, 0, 0), 2)
                cv2.putText(overlay, "GRINDER TIP",
                            (tip[0] + 20, tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw fitted line & display angle
            if results.get('blade_angle') is not None:
                ang_info = results['blade_angle']
                vx, vy, x0, y0 = ang_info['vx'], ang_info['vy'], ang_info['x0'], ang_info['y0']
                angle_deg = ang_info['angle_from_y_deg']

                # Draw line along the blade
                pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
                pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

                # Show angle
                cv2.putText(overlay, f"Blade Angle from Y-axis: {angle_deg:.2f}°",
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Find shortest distance groove for highlighting
            shortest_groove_id = None
            if results['grinding_coordinates']:
                shortest_groove = min(results['grinding_coordinates'],
                                      key=lambda x: x['distance_to_grinder_mm'])
                shortest_groove_id = shortest_groove['tooth_id']

            # Draw grooves with their offset values
            for i, coord in enumerate(results['grinding_coordinates']):
                groove_pos = (coord['groove_position_x_px'], coord['groove_position_y_px'])

                is_shortest = (coord['tooth_id'] == shortest_groove_id)

                # Draw groove circle (highlight shortest with different color)
                if is_shortest:
                    cv2.circle(overlay, groove_pos, 10, (0, 255, 0), -1)  # Green for shortest
                    cv2.circle(overlay, groove_pos, 12, (255, 255, 255), 3)
                else:
                    cv2.circle(overlay, groove_pos, 10, (255, 0, 255), -1)  # Magenta for others
                    cv2.circle(overlay, groove_pos, 12, (255, 255, 255), 2)

                # Draw label with groove ID
                label = f"#{coord['tooth_id']}"
                if is_shortest:
                    label += " SHORTEST"
                cv2.putText(overlay, label,
                            (groove_pos[0] - 30, groove_pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Draw offset values next to groove
                offset_text = f"({coord['move_x_mm']:+.1f}, {coord['move_y_mm']:+.1f})mm"

                # Background box for offset text
                text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                box_x = groove_pos[0] - 35
                box_y = groove_pos[1] - 40

                box_color = (0, 255, 0) if is_shortest else (0, 255, 255)

                cv2.rectangle(overlay,
                              (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5),
                              (0, 0, 0), -1)
                cv2.rectangle(overlay,
                              (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5),
                              box_color, 2)

                # Draw offset text
                cv2.putText(overlay, offset_text,
                            (box_x, box_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Draw arrow to grinder
                if results['grinder_tip'] is not None:
                    arrow_color = (0, 255, 0) if is_shortest else (0, 255, 255)
                    arrow_thickness = 3 if is_shortest else 2
                    cv2.arrowedLine(overlay, groove_pos, results['grinder_tip'],
                                    arrow_color, arrow_thickness, tipLength=0.02)

            # Draw info panel (top-left)
            num_grooves = results['num_grooves']
            panel_height = 30 + (num_grooves * 30) + 90  # Extra space for OPC UA status

            cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (450, panel_height), (255, 255, 255), 2)

            info_y = 30

            # FPS
            cv2.putText(overlay, f"FPS: {self.fps:.1f}",
                        (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 30

            # Number of grooves
            cv2.putText(overlay, f"Grooves Detected: {num_grooves}",
                        (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30

            # OPC UA status
            if self.opcua_connected:
                cv2.putText(overlay, "OPC UA: CONNECTED",
                            (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                info_y += 25

                if self.last_x_offset is not None and self.last_y_offset is not None:
                    cv2.putText(overlay, f"INT1 (X): {self.last_x_offset} mm",
                                (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    info_y += 20
                    cv2.putText(overlay, f"INT2 (Y): {self.last_y_offset} mm",
                                (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    info_y += 25
            else:
                cv2.putText(overlay, "OPC UA: DISCONNECTED",
                            (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                info_y += 30

            # List all groove offsets
            if results['grinding_coordinates']:
                cv2.putText(overlay, "Blade Offsets:",
                            (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                info_y += 25

                for coord in results['grinding_coordinates']:
                    is_shortest = (coord['tooth_id'] == shortest_groove_id)
                    marker = " ★ SHORTEST" if is_shortest else ""
                    offset_line = f"  #{coord['tooth_id']}: ({coord['move_x_mm']:+6.1f}, {coord['move_y_mm']:+6.1f})mm  [{coord['distance_to_grinder_mm']:5.1f}mm]{marker}"
                    color = (0, 255, 0) if is_shortest else (0, 255, 255)
                    cv2.putText(overlay, offset_line,
                                (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    info_y += 22

        except Exception as e:
            print(f"Overlay error: {e}")

        return overlay

    def run(self):
        """Main run loop"""
        print("\n" + "=" * 70)
        print("REAL-TIME BLADE DETECTION SYSTEM with OPC UA")
        print("=" * 70)

        # Initialize OPC UA connection
        self.initialize_opcua()

        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera!")
            return

        self.running = True

        # Start capture thread
        capture_thread = Thread(target=self.capture_thread, daemon=True)
        capture_thread.start()

        # Start detection thread
        detection_thread = Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()

        print("\n✓ System running!")
        print("  Press 'q' to quit")
        print("  Press 's' to save current frame")
        print("  Press 'r' to reset detection\n")

        if self.opcua_connected:
            print("✓ OPC UA: Writing shortest distance groove to INT1 & INT2\n")
        else:
            print("⚠ OPC UA: Not connected - detection only mode\n")

        # Main display loop
        try:
            while self.running:
                with self.frame_lock:
                    if self.frame is not None:
                        display_frame = self.frame.copy()
                    else:
                        time.sleep(0.01)
                        continue

                # Draw overlay
                display_frame = self.draw_overlay(display_frame)

                # Calculate FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                # Resize for display if too large
                height, width = display_frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = 1280
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))

                # Show frame
                cv2.namedWindow('Real-Time Blade Detection', cv2.WINDOW_NORMAL)
                cv2.imshow('Real-Time Blade Detection', display_frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nShutting down...")
                    self.running = False
                    break
                elif key == ord('s'):
                    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Saved: {filename}")
                elif key == ord('r'):
                    with self.results_lock:
                        self.latest_results = None
                    print("✓ Detection reset")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.running = False

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")

        self.running = False
        time.sleep(0.5)

        try:
            # Disconnect OPC UA
            if self.opcua_client is not None and self.opcua_connected:
                self.opcua_client.disconnect()
                print("✓ OPC UA disconnected")

            # Clean up camera
            if self.cam is not None:
                self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam

            if self.cam_list is not None:
                self.cam_list.Clear()

            if self.system is not None:
                self.system.ReleaseInstance()

        except Exception as e:
            print(f"Cleanup error: {e}")

        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """Main entry point"""

    # Configuration
    config = {
        # Camera settings
        'pixels_per_mm': 86.96,  # Calibration factor
        'window_size': 50,  # Detection window size
        'min_depth_px': 200,  # Minimum groove depth
        'detection_fps': 2.0,  # Detection rate (Hz)
        'frame_rate': 30.0,  # Camera frame rate
        'exposure_time': 5000,  # Exposure time (μs)
        'gain': 0.0,  # Gain (dB)

        # OPC UA settings
        'opcua_server_url': 'opc.tcp://172.24.200.1:4840',  # YOUR OPC UA SERVER URL
        'opcua_int1_node': 'ns=21;s=R1c_int1',  # Node ID for INT1 (X offset)
        'opcua_int2_node': 'ns=21;s=R1c_int2',  # Node ID for INT2 (Y offset)
    }

    print("\n" + "=" * 70)
    print("CONFIGURATION:")
    print("=" * 70)
    print(f"OPC UA Server: {config['opcua_server_url']}")
    print(f"INT1 Node: {config['opcua_int1_node']} (X offset)")
    print(f"INT2 Node: {config['opcua_int2_node']} (Y offset)")
    print("=" * 70 + "\n")

    # Create and run detector
    detector = RealtimeBladeDetector(config)
    detector.run()


if __name__ == "__main__":
    main()