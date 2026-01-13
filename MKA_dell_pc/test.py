# # #!/usr/bin/env python3
# # """
# # Real-Time Serrated Blade Detection System
# # Uses FLIR USB Camera with PySpin driver
# # Python 3.10
# #
# # Updated:
# # - Blade detection (grooves/teeth) runs CONTINUOUSLY
# # - Grinder position only updates when 'g' key is pressed
# # - Offset calculations always use stored grinder position
# # """
# #
# import cv2
# import numpy as np
# import PySpin
# import time
# from datetime import datetime
# from threading import Thread, Lock
# from serrated_blade_detector import SerratedBladeAnalyzer
# import sys
# import json
# import os
#
#
# class RealtimeBladeDetector:
#     """Real-time blade detection using FLIR camera"""
#
#     def __init__(self, config):
#         """
#         Initialize the real-time detector
#
#         Args:
#             config: Configuration dictionary
#         """
#         self.config = config
#         self.running = False
#         self.frame = None
#         self.frame_lock = Lock()
#         self.latest_results = None
#         self.results_lock = Lock()
#
#         # Performance tracking
#         self.fps = 0
#         self.frame_count = 0
#         self.start_time = time.time()
#
#         # Camera system
#         self.system = None
#         self.cam = None
#         self.cam_list = None
#
#         # Grinder tip control
#         self.grinder_tip_stored = None
#         self.grinder_position_file = config.get('grinder_position_file', 'grinder_position.json')
#         self.detect_grinder_flag = False  # Flag to UPDATE grinder position
#         self.grinder_lock = Lock()
#
#         # Load stored grinder position on startup
#         self.load_grinder_position()
#
#     def load_grinder_position(self):
#         """Load grinder position from JSON file"""
#         if os.path.exists(self.grinder_position_file):
#             try:
#                 with open(self.grinder_position_file, 'r') as f:
#                     data = json.load(f)
#                     self.grinder_tip_stored = tuple(data['grinder_tip'])
#                     print(f"✓ Loaded grinder position from file: {self.grinder_tip_stored}")
#             except Exception as e:
#                 print(f"⚠ Could not load grinder position: {e}")
#                 self.grinder_tip_stored = None
#         else:
#             print("ℹ No saved grinder position found")
#             self.grinder_tip_stored = None
#
#     def save_grinder_position(self, grinder_tip):
#         """Save grinder position to JSON file"""
#         try:
#             # Convert numpy types to native Python types for JSON serialization
#             data = {
#                 'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
#                 'timestamp': datetime.now().isoformat()
#             }
#             with open(self.grinder_position_file, 'w') as f:
#                 json.dump(data, f, indent=4)
#             print(f"✓ Saved grinder position to file: ({data['grinder_tip'][0]}, {data['grinder_tip'][1]})")
#         except Exception as e:
#             print(f"⚠ Could not save grinder position: {e}")
#
#     def initialize_camera(self):
#         """Initialize FLIR camera using PySpin"""
#         print("Initializing FLIR camera...")
#
#         try:
#             # Retrieve singleton reference to system object
#             self.system = PySpin.System.GetInstance()
#
#             # Retrieve list of cameras from the system
#             self.cam_list = self.system.GetCameras()
#
#             num_cameras = self.cam_list.GetSize()
#             print(f"Number of cameras detected: {num_cameras}")
#
#             if num_cameras == 0:
#                 print("ERROR: No cameras detected!")
#                 self.cam_list.Clear()
#                 self.system.ReleaseInstance()
#                 return False
#
#             # Use first camera
#             self.cam = self.cam_list[0]
#
#             # Initialize camera
#             self.cam.Init()
#
#             # Configure camera settings
#             self._configure_camera()
#
#             # Begin acquiring images
#             self.cam.BeginAcquisition()
#
#             print("✓ Camera initialized successfully!")
#             return True
#
#         except PySpin.SpinnakerException as ex:
#             print(f"ERROR: {ex}")
#             return False
#
#     def _configure_camera(self):
#         """Configure camera settings for optimal detection"""
#         print("Configuring camera settings...")
#
#         try:
#             # Set acquisition mode to continuous
#             node_acquisition_mode = PySpin.CEnumerationPtr(
#                 self.cam.GetNodeMap().GetNode('AcquisitionMode')
#             )
#             if PySpin.IsAvailable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode):
#                 node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
#                 if PySpin.IsAvailable(node_acquisition_mode_continuous) and PySpin.IsReadable(
#                         node_acquisition_mode_continuous):
#                     acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
#                     node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
#                     print("  ✓ Acquisition mode set to continuous")
#
#             # Set frame rate if specified
#             if 'frame_rate' in self.config:
#                 try:
#                     # Enable manual frame rate control
#                     node_acquisition_framerate_enable = PySpin.CBooleanPtr(
#                         self.cam.GetNodeMap().GetNode('AcquisitionFrameRateEnable')
#                     )
#                     if PySpin.IsAvailable(node_acquisition_framerate_enable) and PySpin.IsWritable(
#                             node_acquisition_framerate_enable):
#                         node_acquisition_framerate_enable.SetValue(True)
#
#                     # Set frame rate
#                     node_acquisition_framerate = PySpin.CFloatPtr(
#                         self.cam.GetNodeMap().GetNode('AcquisitionFrameRate')
#                     )
#                     if PySpin.IsAvailable(node_acquisition_framerate) and PySpin.IsWritable(node_acquisition_framerate):
#                         framerate_to_set = min(
#                             node_acquisition_framerate.GetMax(),
#                             self.config['frame_rate']
#                         )
#                         node_acquisition_framerate.SetValue(framerate_to_set)
#                         print(f"  ✓ Frame rate set to {framerate_to_set} fps")
#                 except:
#                     print("  ⚠ Could not set frame rate (camera may not support it)")
#
#             # Set exposure time if specified
#             if 'exposure_time' in self.config:
#                 try:
#                     node_exposure_auto = PySpin.CEnumerationPtr(
#                         self.cam.GetNodeMap().GetNode('ExposureAuto')
#                     )
#                     if PySpin.IsAvailable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
#                         node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
#                         if PySpin.IsAvailable(node_exposure_auto_off) and PySpin.IsReadable(node_exposure_auto_off):
#                             node_exposure_auto.SetIntValue(node_exposure_auto_off.GetValue())
#
#                     node_exposure_time = PySpin.CFloatPtr(
#                         self.cam.GetNodeMap().GetNode('ExposureTime')
#                     )
#                     if PySpin.IsAvailable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
#                         exposure_time_to_set = min(
#                             node_exposure_time.GetMax(),
#                             self.config['exposure_time']
#                         )
#                         node_exposure_time.SetValue(exposure_time_to_set)
#                         print(f"  ✓ Exposure time set to {exposure_time_to_set} μs")
#                 except:
#                     print("  ⚠ Could not set exposure time")
#
#             # Set gain if specified
#             if 'gain' in self.config:
#                 try:
#                     node_gain_auto = PySpin.CEnumerationPtr(
#                         self.cam.GetNodeMap().GetNode('GainAuto')
#                     )
#                     if PySpin.IsAvailable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
#                         node_gain_auto_off = node_gain_auto.GetEntryByName('Off')
#                         if PySpin.IsAvailable(node_gain_auto_off) and PySpin.IsReadable(node_gain_auto_off):
#                             node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())
#
#                     node_gain = PySpin.CFloatPtr(
#                         self.cam.GetNodeMap().GetNode('Gain')
#                     )
#                     if PySpin.IsAvailable(node_gain) and PySpin.IsWritable(node_gain):
#                         gain_to_set = min(node_gain.GetMax(), self.config['gain'])
#                         node_gain.SetValue(gain_to_set)
#                         print(f"  ✓ Gain set to {gain_to_set} dB")
#                 except:
#                     print("  ⚠ Could not set gain")
#
#         except PySpin.SpinnakerException as ex:
#             print(f"ERROR configuring camera: {ex}")
#
#     def capture_thread(self):
#         """Thread for continuously capturing frames"""
#         print("Starting capture thread...")
#
#         while self.running:
#             try:
#                 # Retrieve next received image
#                 image_result = self.cam.GetNextImage(1000)  # 1000ms timeout
#
#                 if image_result.IsIncomplete():
#                     print(f"Image incomplete with status {image_result.GetImageStatus()}")
#                 else:
#                     # Convert to numpy array
#                     width = image_result.GetWidth()
#                     height = image_result.GetHeight()
#
#                     # Get pixel format
#                     pixel_format = image_result.GetPixelFormat()
#
#                     # Convert based on pixel format
#                     if pixel_format == PySpin.PixelFormat_Mono8:
#                         # Grayscale image - convert directly
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
#                     elif pixel_format == PySpin.PixelFormat_BayerRG8:
#                         # Bayer format - convert to BGR
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2BGR)
#                     elif pixel_format == PySpin.PixelFormat_BayerGR8:
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GR2BGR)
#                     elif pixel_format == PySpin.PixelFormat_BayerGB8:
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GB2BGR)
#                     elif pixel_format == PySpin.PixelFormat_BayerBG8:
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)
#                     elif pixel_format == PySpin.PixelFormat_RGB8:
#                         # RGB - convert to BGR
#                         image_data = image_result.GetNDArray()
#                         frame = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
#                     elif pixel_format == PySpin.PixelFormat_BGR8:
#                         # Already BGR
#                         frame = image_result.GetNDArray()
#                     else:
#                         # Try to use processor for conversion
#                         try:
#                             processor = PySpin.ImageProcessor()
#                             processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
#                             image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
#                             frame = image_converted.GetNDArray()
#                         except:
#                             # Fallback: just get raw data and convert to grayscale then BGR
#                             print(f"Warning: Unknown pixel format {pixel_format}, using grayscale conversion")
#                             image_data = image_result.GetNDArray().reshape((height, width))
#                             frame = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                     # Update frame
#                     with self.frame_lock:
#                         self.frame = frame.copy()
#                         self.frame_count += 1
#
#                 # Release image
#                 image_result.Release()
#
#             except PySpin.SpinnakerException as ex:
#                 print(f"Capture error: {ex}")
#                 time.sleep(0.1)
#             except Exception as ex:
#                 print(f"Unexpected capture error: {ex}")
#                 time.sleep(0.1)
#
#     def detection_thread(self):
#         """Thread for running blade detection"""
#         print("Starting detection thread...")
#
#         detection_interval = 1.0 / self.config['detection_fps']
#         last_detection_time = 0
#
#         while self.running:
#             current_time = time.time()
#
#             # Check if it's time for detection
#             if current_time - last_detection_time >= detection_interval:
#                 with self.frame_lock:
#                     if self.frame is not None:
#                         frame_to_process = self.frame.copy()
#                     else:
#                         time.sleep(0.01)
#                         continue
#
#                 # Check if we should UPDATE grinder position this cycle
#                 with self.grinder_lock:
#                     should_update_grinder = self.detect_grinder_flag
#                     if should_update_grinder:
#                         self.detect_grinder_flag = False  # Reset flag
#
#                 # Run detection
#                 try:
#                     # Save frame temporarily
#                     temp_path = '/tmp/temp_blade_frame.png'
#                     cv2.imwrite(temp_path, frame_to_process)
#
#                     # Create analyzer and run detection
#                     analyzer = SerratedBladeAnalyzer(temp_path)
#                     analyzer.preprocess_image()
#                     analyzer.detect_edges()
#                     analyzer.find_blade_contours()
#
#                     # ALWAYS run blade detection (for edge points and grinder detection)
#                     analyzer.detect_blade_and_grinder()
#
#                     # Only UPDATE stored grinder position if flag was set
#                     if should_update_grinder:
#                         detected_grinder_tip = analyzer.grinder_tip
#
#                         if detected_grinder_tip is not None:
#                             # Convert to native Python types
#                             detected_grinder_tip = (int(detected_grinder_tip[0]), int(detected_grinder_tip[1]))
#
#                             # Update stored position
#                             with self.grinder_lock:
#                                 self.grinder_tip_stored = detected_grinder_tip
#                             # Save to file
#                             self.save_grinder_position(detected_grinder_tip)
#                             print(f"✓ Grinder tip UPDATED and saved: {detected_grinder_tip}")
#                         else:
#                             print("⚠ Grinder tip detection failed - keeping previous stored position")
#
#                     # ALWAYS use stored grinder position for offset calculations
#                     with self.grinder_lock:
#                         grinder_tip_to_use = self.grinder_tip_stored
#
#                     # Calculate blade angle
#                     if hasattr(analyzer, "blade_edge_points") and analyzer.blade_edge_points is not None and len(
#                             analyzer.blade_edge_points) > 2:
#                         pts = np.array(analyzer.blade_edge_points)
#
#                         # Fit a straight line (least squares)
#                         fit_result = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
#
#                         # Extract values properly to avoid deprecation warnings
#                         vx = float(fit_result[0][0])
#                         vy = float(fit_result[1][0])
#                         x0 = float(fit_result[2][0])
#                         y0 = float(fit_result[3][0])
#
#                         # Compute angle from Y-axis (in degrees)
#                         angle_from_x = np.arctan2(vy, vx)
#                         angle_from_y = np.pi / 2 - angle_from_x  # offset from Y axis
#                         angle_deg = float(np.degrees(angle_from_y))
#
#                         blade_angle_info = {
#                             'vx': vx,
#                             'vy': vy,
#                             'x0': x0,
#                             'y0': y0,
#                             'angle_from_y_deg': angle_deg
#                         }
#                     else:
#                         blade_angle_info = None
#
#                     # Extract tooth profiles (CONTINUOUS)
#                     analyzer.teeth_profiles = analyzer.extract_tooth_profiles(
#                         window_size=self.config['window_size'],
#                         min_depth_px=self.config['min_depth_px']
#                     )
#
#                     # Calculate coordinates using STORED grinder position
#                     # Temporarily set the grinder_tip in analyzer for coordinate generation
#                     analyzer.grinder_tip = grinder_tip_to_use
#                     grinding_coords = analyzer.generate_grinding_coordinates(
#                         pixels_per_mm=self.config['pixels_per_mm']
#                     )
#
#                     # Store results
#                     results = {
#                         'timestamp': datetime.now(),
#                         'num_grooves': len(analyzer.teeth_profiles),
#                         'grinding_coordinates': grinding_coords,
#                         'grinder_tip': grinder_tip_to_use,  # Use stored position
#                         'blade_edge_points': analyzer.blade_edge_points,
#                         'teeth_profiles': analyzer.teeth_profiles,
#                         'blade_angle': blade_angle_info,
#                         'grinder_updated_this_cycle': should_update_grinder  # For display info
#                     }
#
#                     with self.results_lock:
#                         self.latest_results = results
#
#                 except Exception as e:
#                     print(f"Detection error: {e}")
#                     import traceback
#                     traceback.print_exc()
#
#                 last_detection_time = current_time
#             else:
#                 time.sleep(0.01)
#
#     def draw_overlay(self, frame):
#         """Draw detection overlay on frame"""
#         overlay = frame.copy()
#
#         with self.results_lock:
#             if self.latest_results is None:
#                 return overlay
#
#             results = self.latest_results
#
#         try:
#             # Draw grinder tip
#             if results['grinder_tip'] is not None:
#                 tip = results['grinder_tip']
#                 cv2.circle(overlay, tip, 12, (0, 255, 255), 3)
#                 cv2.circle(overlay, tip, 15, (0, 0, 0), 2)
#
#                 # Show if it's stored vs newly updated
#                 label = "GRINDER TIP (STORED)"
#                 label_color = (0, 255, 255)
#                 if results.get('grinder_updated_this_cycle', False):
#                     label = "GRINDER TIP (JUST UPDATED!)"
#                     label_color = (0, 255, 0)
#
#                 cv2.putText(overlay, label,
#                             (tip[0] + 20, tip[1] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
#             else:
#                 # No grinder position available
#                 cv2.putText(overlay, "NO GRINDER POSITION - Press 'g' to detect",
#                             (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#             # Draw fitted line & display angle
#             if results.get('blade_angle') is not None:
#                 ang_info = results['blade_angle']
#                 vx, vy, x0, y0 = ang_info['vx'], ang_info['vy'], ang_info['x0'], ang_info['y0']
#                 angle_deg = ang_info['angle_from_y_deg']
#
#                 # Draw line along the blade
#                 pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
#                 pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
#                 cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
#
#                 # Show angle
#                 cv2.putText(overlay, f"Blade Angle from Y-axis: {angle_deg:.2f}°",
#                             (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#
#             # Draw grooves with their offset values
#             for i, coord in enumerate(results['grinding_coordinates']):
#                 groove_pos = (coord['groove_position_x_px'], coord['groove_position_y_px'])
#
#                 # Draw groove circle
#                 cv2.circle(overlay, groove_pos, 10, (255, 0, 255), -1)
#                 cv2.circle(overlay, groove_pos, 12, (255, 255, 255), 2)
#
#                 # Draw label with groove ID
#                 label = f"#{coord['tooth_id']}"
#                 cv2.putText(overlay, label,
#                             (groove_pos[0] - 30, groove_pos[1] - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#
#                 # Draw offset values next to groove
#                 offset_text = f"({coord['move_x_mm']:+.1f}, {coord['move_y_mm']:+.1f})mm"
#
#                 # Background box for offset text
#                 text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                 box_x = groove_pos[0] - 35
#                 box_y = groove_pos[1] - 40
#                 cv2.rectangle(overlay,
#                               (box_x - 5, box_y - text_size[1] - 5),
#                               (box_x + text_size[0] + 5, box_y + 5),
#                               (0, 0, 0), -1)
#                 cv2.rectangle(overlay,
#                               (box_x - 5, box_y - text_size[1] - 5),
#                               (box_x + text_size[0] + 5, box_y + 5),
#                               (0, 255, 255), 2)
#
#                 # Draw offset text
#                 cv2.putText(overlay, offset_text,
#                             (box_x, box_y - 3),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#
#                 # Draw arrow to grinder
#                 if results['grinder_tip'] is not None:
#                     cv2.arrowedLine(overlay, groove_pos, results['grinder_tip'],
#                                     (0, 255, 255), 2, tipLength=0.02)
#
#             # Draw info panel (top-left)
#             num_grooves = results['num_grooves']
#             panel_height = 30 + (num_grooves * 30) + 20
#
#             cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
#             cv2.rectangle(overlay, (10, 10), (450, panel_height), (255, 255, 255), 2)
#
#             info_y = 30
#
#             # FPS
#             cv2.putText(overlay, f"FPS: {self.fps:.1f}",
#                         (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             info_y += 30
#
#             # Number of grooves
#             cv2.putText(overlay, f"Grooves Detected: {num_grooves}",
#                         (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             info_y += 30
#
#             # List all groove offsets
#             if results['grinding_coordinates']:
#                 cv2.putText(overlay, "Blade Offsets:",
#                             (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#                 info_y += 25
#
#                 for coord in results['grinding_coordinates']:
#                     offset_line = f"  #{coord['tooth_id']}: ({coord['move_x_mm']:+6.1f}, {coord['move_y_mm']:+6.1f})mm  [{coord['distance_to_grinder_mm']:5.1f}mm]"
#                     cv2.putText(overlay, offset_line,
#                                 (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
#                     info_y += 22
#
#         except Exception as e:
#             print(f"Overlay error: {e}")
#             import traceback
#             traceback.print_exc()
#
#         return overlay
#
#     def run(self):
#         """Main run loop"""
#         print("\n" + "=" * 70)
#         print("REAL-TIME BLADE DETECTION SYSTEM")
#         print("=" * 70)
#         print("Mode: CONTINUOUS blade detection with stored grinder position")
#         print("=" * 70)
#
#         # Initialize camera
#         if not self.initialize_camera():
#             print("Failed to initialize camera!")
#             return
#
#         self.running = True
#
#         # Start capture thread
#         capture_thread = Thread(target=self.capture_thread, daemon=True)
#         capture_thread.start()
#
#         # Start detection thread
#         detection_thread = Thread(target=self.detection_thread, daemon=True)
#         detection_thread.start()
#
#         print("\n✓ System running!")
#         print("  Press 'q' to quit")
#         print("  Press 's' to save current frame")
#         print("  Press 'g' to UPDATE grinder position (save new position)")
#         print("  Press 'r' to reset detection")
#         print("\n  Blade detection runs continuously!")
#         print("  Offsets calculated using stored grinder position\n")
#
#         # Main display loop
#         try:
#             while self.running:
#                 with self.frame_lock:
#                     if self.frame is not None:
#                         display_frame = self.frame.copy()
#                     else:
#                         time.sleep(0.01)
#                         continue
#
#                 # Draw overlay
#                 display_frame = self.draw_overlay(display_frame)
#
#                 # Calculate FPS
#                 elapsed = time.time() - self.start_time
#                 if elapsed > 0:
#                     self.fps = self.frame_count / elapsed
#
#                 # Resize for display if too large
#                 height, width = display_frame.shape[:2]
#                 if width > 1280:
#                     scale = 1280 / width
#                     new_width = 1280
#                     new_height = int(height * scale)
#                     display_frame = cv2.resize(display_frame, (new_width, new_height))
#
#                 # Show frame
#                 cv2.namedWindow('Real-Time Blade Detection', cv2.WINDOW_NORMAL)
#                 cv2.imshow('Real-Time Blade Detection', display_frame)
#
#                 # Handle keys
#                 key = cv2.waitKey(1) & 0xFF
#
#                 if key == ord('q'):
#                     print("\nShutting down...")
#                     self.running = False
#                     break
#                 elif key == ord('s'):
#                     filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#                     cv2.imwrite(filename, display_frame)
#                     print(f"✓ Saved: {filename}")
#                 elif key == ord('g'):
#                     # Trigger grinder position UPDATE
#                     with self.grinder_lock:
#                         self.detect_grinder_flag = True
#                     print("✓ Grinder position update triggered - will update on next cycle")
#                 elif key == ord('r'):
#                     with self.results_lock:
#                         self.latest_results = None
#                     print("✓ Detection reset")
#
#         except KeyboardInterrupt:
#             print("\n\nInterrupted by user")
#             self.running = False
#
#         finally:
#             self.cleanup()
#
#     def cleanup(self):
#         """Clean up resources"""
#         print("Cleaning up...")
#
#         self.running = False
#         time.sleep(0.5)
#
#         try:
#             if self.cam is not None:
#                 self.cam.EndAcquisition()
#                 self.cam.DeInit()
#                 del self.cam
#
#             if self.cam_list is not None:
#                 self.cam_list.Clear()
#
#             if self.system is not None:
#                 self.system.ReleaseInstance()
#
#         except Exception as e:
#             print(f"Cleanup error: {e}")
#
#         cv2.destroyAllWindows()
#         print("✓ Cleanup complete")
#
#
# def main():
#     """Main entry point"""
#
#     # Configuration
#     config = {
#         'pixels_per_mm': 86.96,  # Calibration factor
#         'window_size': 50,  # Detection window size
#         'min_depth_px': 200,  # Minimum groove depth
#         'detection_fps': 2.0,  # Detection rate (Hz)
#         'frame_rate': 30.0,  # Camera frame rate
#         'exposure_time': 5000,  # Exposure time (μs)
#         'gain': 0.0,  # Gain (dB)
#         'grinder_position_file': 'grinder_position.json',  # File to store grinder position
#     }
#
#     # Create and run detector
#     detector = RealtimeBladeDetector(config)
#     detector.run()
#
#
# if __name__ == "__main__":
#     main()

# !/usr/bin/env python3
#!/usr/bin/env python3
# """
# Real-Time Serrated Blade Detection System with TCP/IP Publishing
# Uses FLIR USB Camera with PySpin driver
# Python 3.10
#
# Features:
# - Blade detection runs continuously
# - Grinder position stored and updated on demand
# - TCP/IP server publishes data to network clients
# """
#
# !/usr/bin/env python3
"""
Real-Time Serrated Blade Detection System with TCP/IP Publishing
Uses FLIR USB Camera with PySpin driver
Python 3.10

Features:
- Blade detection runs continuously
- Grinder position stored and updated on demand
- TCP/IP server publishes data to network clients
"""

import cv2
import numpy as np
import PySpin
import time
from datetime import datetime
from threading import Thread, Lock
from serrated_blade_detector import SerratedBladeAnalyzer
import sys
import json
import os

# Import TCP server module
from tcp_blade_server import BladeDataTCPServer


class RealtimeBladeDetector:
    """Real-time blade detection with network publishing"""

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

        # Grinder tip control
        self.grinder_tip_stored = None
        self.grinder_position_file = config.get('grinder_position_file', 'grinder_position.json')
        self.detect_grinder_flag = False
        self.grinder_lock = Lock()

        # TCP/IP Server
        self.tcp_server = None
        if config.get('tcp_enabled', True):
            tcp_host = config.get('tcp_host', '0.0.0.0')
            tcp_port = config.get('tcp_port', 5000)
            tcp_max_clients = config.get('tcp_max_clients', 5)

            self.tcp_server = BladeDataTCPServer(
                host=tcp_host,
                port=tcp_port,
                max_clients=tcp_max_clients
            )

        # Load stored grinder position
        self.load_grinder_position()

    def load_grinder_position(self):
        """Load grinder position from JSON file"""
        if os.path.exists(self.grinder_position_file):
            try:
                with open(self.grinder_position_file, 'r') as f:
                    data = json.load(f)
                    self.grinder_tip_stored = tuple(data['grinder_tip'])
                    print(f"✓ Loaded grinder position: {self.grinder_tip_stored}")
            except Exception as e:
                print(f"⚠ Could not load grinder position: {e}")
                self.grinder_tip_stored = None
        else:
            print("ℹ No saved grinder position found")
            self.grinder_tip_stored = None

    def save_grinder_position(self, grinder_tip):
        """Save grinder position to JSON file"""
        try:
            data = {
                'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
                'timestamp': datetime.now().isoformat()
            }
            with open(self.grinder_position_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✓ Saved grinder position: ({data['grinder_tip'][0]}, {data['grinder_tip'][1]})")
        except Exception as e:
            print(f"⚠ Could not save grinder position: {e}")

    def initialize_camera(self):
        """Initialize FLIR camera using PySpin"""
        print("Initializing FLIR camera...")

        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            num_cameras = self.cam_list.GetSize()

            print(f"Number of cameras detected: {num_cameras}")

            if num_cameras == 0:
                print("ERROR: No cameras detected!")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                return False

            self.cam = self.cam_list[0]
            self.cam.Init()
            self._configure_camera()
            self.cam.BeginAcquisition()

            print("✓ Camera initialized successfully!")
            return True

        except PySpin.SpinnakerException as ex:
            print(f"ERROR: {ex}")
            return False

    def _configure_camera(self):
        """Configure camera settings"""
        print("Configuring camera settings...")

        try:
            # Continuous acquisition mode
            node_acquisition_mode = PySpin.CEnumerationPtr(
                self.cam.GetNodeMap().GetNode('AcquisitionMode')
            )
            if PySpin.IsAvailable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode):
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if PySpin.IsAvailable(node_acquisition_mode_continuous) and PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                    print("  ✓ Acquisition mode: continuous")

            # Frame rate
            if 'frame_rate' in self.config:
                try:
                    node_framerate_enable = PySpin.CBooleanPtr(
                        self.cam.GetNodeMap().GetNode('AcquisitionFrameRateEnable')
                    )
                    if PySpin.IsAvailable(node_framerate_enable) and PySpin.IsWritable(node_framerate_enable):
                        node_framerate_enable.SetValue(True)

                    node_framerate = PySpin.CFloatPtr(
                        self.cam.GetNodeMap().GetNode('AcquisitionFrameRate')
                    )
                    if PySpin.IsAvailable(node_framerate) and PySpin.IsWritable(node_framerate):
                        framerate_to_set = min(node_framerate.GetMax(), self.config['frame_rate'])
                        node_framerate.SetValue(framerate_to_set)
                        print(f"  ✓ Frame rate: {framerate_to_set} fps")
                except:
                    print("  ⚠ Could not set frame rate")

            # Exposure time
            if 'exposure_time' in self.config:
                try:
                    node_exposure_auto = PySpin.CEnumerationPtr(
                        self.cam.GetNodeMap().GetNode('ExposureAuto')
                    )
                    if PySpin.IsAvailable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
                        node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
                        if PySpin.IsAvailable(node_exposure_auto_off):
                            node_exposure_auto.SetIntValue(node_exposure_auto_off.GetValue())

                    node_exposure_time = PySpin.CFloatPtr(
                        self.cam.GetNodeMap().GetNode('ExposureTime')
                    )
                    if PySpin.IsAvailable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
                        exposure_time_to_set = min(node_exposure_time.GetMax(), self.config['exposure_time'])
                        node_exposure_time.SetValue(exposure_time_to_set)
                        print(f"  ✓ Exposure: {exposure_time_to_set} μs")
                except:
                    print("  ⚠ Could not set exposure")

            # Gain
            if 'gain' in self.config:
                try:
                    node_gain_auto = PySpin.CEnumerationPtr(
                        self.cam.GetNodeMap().GetNode('GainAuto')
                    )
                    if PySpin.IsAvailable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
                        node_gain_auto_off = node_gain_auto.GetEntryByName('Off')
                        if PySpin.IsAvailable(node_gain_auto_off):
                            node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())

                    node_gain = PySpin.CFloatPtr(self.cam.GetNodeMap().GetNode('Gain'))
                    if PySpin.IsAvailable(node_gain) and PySpin.IsWritable(node_gain):
                        gain_to_set = min(node_gain.GetMax(), self.config['gain'])
                        node_gain.SetValue(gain_to_set)
                        print(f"  ✓ Gain: {gain_to_set} dB")
                except:
                    print("  ⚠ Could not set gain")

        except PySpin.SpinnakerException as ex:
            print(f"ERROR configuring camera: {ex}")

    def capture_thread(self):
        """Thread for continuously capturing frames"""
        print("Starting capture thread...")

        while self.running:
            try:
                image_result = self.cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    print(f"Image incomplete: {image_result.GetImageStatus()}")
                else:
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    pixel_format = image_result.GetPixelFormat()

                    # Convert based on pixel format
                    if pixel_format == PySpin.PixelFormat_Mono8:
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                    elif pixel_format == PySpin.PixelFormat_BayerRG8:
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
                        image_data = image_result.GetNDArray()
                        frame = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                    elif pixel_format == PySpin.PixelFormat_BGR8:
                        frame = image_result.GetNDArray()
                    else:
                        try:
                            processor = PySpin.ImageProcessor()
                            processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
                            image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                            frame = image_converted.GetNDArray()
                        except:
                            print(f"Unknown pixel format {pixel_format}")
                            image_data = image_result.GetNDArray().reshape((height, width))
                            frame = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    with self.frame_lock:
                        self.frame = frame.copy()
                        self.frame_count += 1

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print(f"Capture error: {ex}")
                time.sleep(0.1)
            except Exception as ex:
                print(f"Unexpected capture error: {ex}")
                time.sleep(0.1)

    def detection_thread(self):
        """Thread for running blade detection and publishing data"""
        print("Starting detection thread...")

        detection_interval = 1.0 / self.config['detection_fps']
        last_detection_time = 0

        while self.running:
            current_time = time.time()

            if current_time - last_detection_time >= detection_interval:
                with self.frame_lock:
                    if self.frame is not None:
                        frame_to_process = self.frame.copy()
                    else:
                        time.sleep(0.01)
                        continue

                with self.grinder_lock:
                    should_update_grinder = self.detect_grinder_flag
                    if should_update_grinder:
                        self.detect_grinder_flag = False

                try:
                    temp_path = '/tmp/temp_blade_frame.png'
                    cv2.imwrite(temp_path, frame_to_process)

                    analyzer = SerratedBladeAnalyzer(temp_path)
                    analyzer.preprocess_image()
                    analyzer.detect_edges()
                    analyzer.find_blade_contours()
                    analyzer.detect_blade_and_grinder()

                    # Update grinder position if requested
                    if should_update_grinder:
                        detected_grinder_tip = analyzer.grinder_tip

                        if detected_grinder_tip is not None:
                            detected_grinder_tip = (int(detected_grinder_tip[0]), int(detected_grinder_tip[1]))
                            with self.grinder_lock:
                                self.grinder_tip_stored = detected_grinder_tip
                            self.save_grinder_position(detected_grinder_tip)
                            print(f"✓ Grinder updated: {detected_grinder_tip}")
                        else:
                            print("⚠ Grinder detection failed")

                    # Use stored grinder position
                    with self.grinder_lock:
                        grinder_tip_to_use = self.grinder_tip_stored

                    # Calculate blade angle
                    blade_angle_info = None
                    if hasattr(analyzer, "blade_edge_points") and analyzer.blade_edge_points is not None and len(
                            analyzer.blade_edge_points) > 2:
                        pts = np.array(analyzer.blade_edge_points)
                        fit_result = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

                        vx = float(fit_result[0][0])
                        vy = float(fit_result[1][0])
                        x0 = float(fit_result[2][0])
                        y0 = float(fit_result[3][0])

                        angle_from_x = np.arctan2(vy, vx)
                        angle_from_y = np.pi / 2 - angle_from_x
                        angle_deg = float(np.degrees(angle_from_y))

                        blade_angle_info = {
                            'vx': vx,
                            'vy': vy,
                            'x0': x0,
                            'y0': y0,
                            'angle_from_y_deg': angle_deg
                        }

                    # Extract tooth profiles
                    analyzer.teeth_profiles = analyzer.extract_tooth_profiles(
                        window_size=self.config['window_size'],
                        min_depth_px=self.config['min_depth_px']
                    )

                    # Generate coordinates
                    analyzer.grinder_tip = grinder_tip_to_use
                    grinding_coords = analyzer.generate_grinding_coordinates(
                        pixels_per_mm=self.config['pixels_per_mm']
                    )

                    # Store results
                    results = {
                        'timestamp': datetime.now(),
                        'num_grooves': len(analyzer.teeth_profiles),
                        'grinding_coordinates': grinding_coords,
                        'grinder_tip': grinder_tip_to_use,
                        'blade_edge_points': analyzer.blade_edge_points,
                        'teeth_profiles': analyzer.teeth_profiles,
                        'blade_angle': blade_angle_info,
                        'grinder_updated_this_cycle': should_update_grinder
                    }

                    with self.results_lock:
                        self.latest_results = results

                    # Publish to TCP clients
                    if self.tcp_server and self.tcp_server.running and (self.tcp_server.get_client_count() > 0):
                        self.tcp_server.publish_data(results)
                        self.tcp_server.read_client(0)

                except Exception as e:
                    print(f"Detection error: {e}")
                    import traceback
                    traceback.print_exc()

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

                label = "GRINDER (STORED)"
                label_color = (0, 255, 255)
                if results.get('grinder_updated_this_cycle', False):
                    label = "GRINDER (UPDATED!)"
                    label_color = (0, 255, 0)

                cv2.putText(overlay, label, (tip[0] + 20, tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            else:
                cv2.putText(overlay, "NO GRINDER - Press 'g'",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Blade angle
            if results.get('blade_angle') is not None:
                ang = results['blade_angle']
                vx, vy, x0, y0 = ang['vx'], ang['vy'], ang['x0'], ang['y0']
                angle_deg = ang['angle_from_y_deg']

                pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
                pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

                cv2.putText(overlay, f"Angle: {angle_deg:.2f}°",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Grooves
            for i, coord in enumerate(results['grinding_coordinates']):
                pos = (coord['groove_position_x_px'], coord['groove_position_y_px'])

                cv2.circle(overlay, pos, 10, (255, 0, 255), -1)
                cv2.circle(overlay, pos, 12, (255, 255, 255), 2)

                label = f"#{coord['tooth_id']}"
                cv2.putText(overlay, label, (pos[0] - 30, pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                offset_text = f"({coord['move_x_mm']:+.1f}, {coord['move_y_mm']:+.1f})mm"
                text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                box_x = pos[0] - 35
                box_y = pos[1] - 40

                cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5), (0, 0, 0), -1)
                cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5), (0, 255, 255), 2)
                cv2.putText(overlay, offset_text, (box_x, box_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if results['grinder_tip'] is not None:
                    cv2.arrowedLine(overlay, pos, results['grinder_tip'],
                                    (0, 255, 255), 2, tipLength=0.02)

            # Info panel
            num_grooves = results['num_grooves']
            tcp_clients = self.tcp_server.get_client_count() if self.tcp_server else 0
            panel_height = 90 + (num_grooves * 30)

            cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (500, panel_height), (255, 255, 255), 2)

            y = 30
            cv2.putText(overlay, f"FPS: {self.fps:.1f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30

            # TCP status
            tcp_color = (0, 255, 0) if tcp_clients > 0 else (128, 128, 128)
            cv2.putText(overlay, f"TCP Clients: {tcp_clients}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcp_color, 2)
            y += 30

            cv2.putText(overlay, f"Grooves: {num_grooves}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

            if results['grinding_coordinates']:
                cv2.putText(overlay, "Blade Offsets:", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y += 25

                for coord in results['grinding_coordinates']:
                    line = f"  #{coord['tooth_id']}: ({coord['move_x_mm']:+6.1f}, {coord['move_y_mm']:+6.1f})mm  [{coord['distance_to_grinder_mm']:5.1f}mm]"
                    cv2.putText(overlay, line, (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y += 22

        except Exception as e:
            print(f"Overlay error: {e}")

        return overlay

    def run(self):
        """Main run loop"""
        print("\n" + "=" * 70)
        print("REAL-TIME BLADE DETECTION WITH TCP/IP SERVER")
        print("=" * 70)

        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera!")
            return

        # Start TCP server
        if self.tcp_server:
            if not self.tcp_server.start():
                print("Warning: TCP server failed to start")

        self.running = True

        # Start threads
        capture_thread = Thread(target=self.capture_thread, daemon=True)
        capture_thread.start()

        detection_thread = Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()

        print("\n✓ System running!")
        print("  'q' = quit")
        print("  's' = save frame")
        print("  'g' = update grinder position")
        print("  'r' = reset detection")
        if self.tcp_server:
            print(f"\n  TCP Server: {self.config.get('tcp_host', '0.0.0.0')}:{self.config.get('tcp_port', 5000)}")

        try:
            while self.running:
                with self.frame_lock:
                    if self.frame is not None:
                        display_frame = self.frame.copy()
                    else:
                        time.sleep(0.01)
                        continue

                display_frame = self.draw_overlay(display_frame)

                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                # Resize if needed
                height, width = display_frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(display_frame,
                                               (1280, int(height * scale)))

                cv2.namedWindow('Blade Detection', cv2.WINDOW_NORMAL)
                cv2.imshow('Blade Detection', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nShutting down...")
                    self.running = False
                    break
                elif key == ord('s'):
                    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Saved: {filename}")
                elif key == ord('g'):
                    with self.grinder_lock:
                        self.detect_grinder_flag = True
                    print("✓ Grinder update triggered")
                elif key == ord('r'):
                    with self.results_lock:
                        self.latest_results = None
                    print("✓ Detection reset")

        except KeyboardInterrupt:
            print("\n\nInterrupted")
            self.running = False

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        time.sleep(0.5)

        # Stop TCP server
        if self.tcp_server:
            self.tcp_server.stop()

        # Clean up camera
        try:
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

    config = {
        'pixels_per_mm': 86.96,
        'window_size': 50,
        'min_depth_px': 200,
        'detection_fps': 2.0,
        'frame_rate': 30.0,
        'exposure_time': 5000,
        'gain': 0.0,
        'grinder_position_file': 'grinder_position.json',

        # TCP/IP Server settings
        'tcp_enabled': True,
        'tcp_host': '172.24.9.15',  # Listen on all interfaces
        'tcp_port': 5000,
        'tcp_max_clients': 5,
    }

    detector = RealtimeBladeDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
