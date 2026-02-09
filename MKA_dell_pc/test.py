# """
# Optimized Real-Time Serrated Blade Detection System
# Performance improvements:
# - Removed unnecessary file I/O
# - Optimized frame processing
# - Reduced lock contention
# - Removed redundant conversions
# - Streamlined overlay drawing
# """
#
# import cv2
# import numpy as np
# import PySpin
# import time
# from datetime import datetime
# from threading import Thread, Lock
# from serrated_blade_detector import SerratedBladeAnalyzer
# import json
# import os
# from tcp_blade_server import BladeDataTCPServer
#
#
# class RealtimeBladeDetector:
#     """Optimized real-time blade detection with network publishing"""
#
#     def __init__(self, config):
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
#         self.detect_grinder_flag = False
#         self.grinder_lock = Lock()
#
#         # TCP/IP Server
#         self.tcp_server = None
#         if config.get('tcp_enabled', True):
#             self.tcp_server = BladeDataTCPServer(
#                 host=config.get('tcp_host', '0.0.0.0'),
#                 port=config.get('tcp_port', 5000),
#                 max_clients=config.get('tcp_max_clients', 5)
#             )
#
#         self.load_grinder_position()
#
#     def load_grinder_position(self):
#         """Load grinder position from JSON file"""
#         if os.path.exists(self.grinder_position_file):
#             try:
#                 with open(self.grinder_position_file, 'r') as f:
#                     data = json.load(f)
#                     self.grinder_tip_stored = tuple(data['grinder_tip'])
#                     print(f"✓ Loaded grinder position: {self.grinder_tip_stored}")
#             except Exception as e:
#                 print(f"⚠ Could not load grinder position: {e}")
#                 self.grinder_tip_stored = None
#         else:
#             self.grinder_tip_stored = None
#
#     def save_grinder_position(self, grinder_tip):
#         """Save grinder position to JSON file"""
#         try:
#             data = {
#                 'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
#                 'timestamp': datetime.now().isoformat()
#             }
#             with open(self.grinder_position_file, 'w') as f:
#                 json.dump(data, f, indent=4)
#             print(f"✓ Saved grinder position: {grinder_tip}")
#         except Exception as e:
#             print(f"⚠ Could not save grinder position: {e}")
#
#     def initialize_camera(self):
#         """Initialize FLIR camera using PySpin"""
#         print("Initializing FLIR camera...")
#
#         try:
#             self.system = PySpin.System.GetInstance()
#             self.cam_list = self.system.GetCameras()
#
#             if self.cam_list.GetSize() == 0:
#                 print("ERROR: No cameras detected!")
#                 self.cam_list.Clear()
#                 self.system.ReleaseInstance()
#                 return False
#
#             self.cam = self.cam_list[0]
#             self.cam.Init()
#             self._configure_camera()
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
#         """Configure camera settings"""
#         try:
#             nodemap = self.cam.GetNodeMap()
#
#             # Continuous acquisition mode
#             node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
#             if PySpin.IsWritable(node_acq_mode):
#                 node_continuous = node_acq_mode.GetEntryByName('Continuous')
#                 node_acq_mode.SetIntValue(node_continuous.GetValue())
#
#             # Frame rate
#             if 'frame_rate' in self.config:
#                 node_fr_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
#                 if PySpin.IsWritable(node_fr_enable):
#                     node_fr_enable.SetValue(True)
#
#                 node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
#                 if PySpin.IsWritable(node_fr):
#                     node_fr.SetValue(min(node_fr.GetMax(), self.config['frame_rate']))
#
#             # Exposure
#             if 'exposure_time' in self.config:
#                 node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
#                 if PySpin.IsWritable(node_exp_auto):
#                     node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
#
#                 node_exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
#                 if PySpin.IsWritable(node_exp):
#                     node_exp.SetValue(min(node_exp.GetMax(), self.config['exposure_time']))
#
#             # Gain
#             if 'gain' in self.config:
#                 node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
#                 if PySpin.IsWritable(node_gain_auto):
#                     node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
#
#                 node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
#                 if PySpin.IsWritable(node_gain):
#                     node_gain.SetValue(min(node_gain.GetMax(), self.config['gain']))
#
#         except PySpin.SpinnakerException as ex:
#             print(f"ERROR configuring camera: {ex}")
#
#     def capture_thread(self):
#         """Thread for continuously capturing frames"""
#         print("Starting capture thread...")
#
#         # Pre-create processor for efficiency
#         processor = PySpin.ImageProcessor()
#         processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
#
#         while self.running:
#             try:
#                 image_result = self.cam.GetNextImage(1000)
#
#                 if not image_result.IsIncomplete():
#                     pixel_format = image_result.GetPixelFormat()
#
#                     # Fast conversion based on pixel format
#                     if pixel_format == PySpin.PixelFormat_BGR8:
#                         frame = image_result.GetNDArray()
#                     elif pixel_format == PySpin.PixelFormat_Mono8:
#                         frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
#                     else:
#                         # Use pre-created processor
#                         image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
#                         frame = image_converted.GetNDArray()
#
#                     # Single rotation operation
#                     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#                     with self.frame_lock:
#                         self.frame = frame
#                         self.frame_count += 1
#
#                 image_result.Release()
#
#             except PySpin.SpinnakerException:
#                 time.sleep(0.01)
#             except Exception:
#                 time.sleep(0.01)
#
#     def detection_thread(self):
#         """Thread for running blade detection and publishing data"""
#         print("Starting detection thread...")
#
#         detection_interval = 1.0 / self.config['detection_fps']
#         last_detection_time = 0
#
#         while self.running:
#             current_time = time.time()
#
#             if current_time - last_detection_time >= detection_interval:
#                 # Quick frame grab
#                 with self.frame_lock:
#                     frame_to_process = self.frame
#
#                 if frame_to_process is None:
#                     time.sleep(0.1)
#                     continue
#
#                 # Check grinder update flag
#                 with self.grinder_lock:
#                     should_update_grinder = self.detect_grinder_flag
#                     if should_update_grinder:
#                         self.detect_grinder_flag = False
#
#                 try:
#                     # REMOVED: Unnecessary temp file write
#                     # analyzer = SerratedBladeAnalyzer(temp_path)
#
#                     # Pass frame directly to analyzer (modify SerratedBladeAnalyzer to accept numpy array)
#                     analyzer = SerratedBladeAnalyzer(frame_to_process)
#                     analyzer.preprocess_image()
#                     analyzer.detect_edges()
#                     analyzer.find_blade_contours()
#                     analyzer.detect_blade_and_grinder()
#
#                     # Update grinder position if requested
#                     if should_update_grinder and analyzer.grinder_tip is not None:
#                         detected_grinder_tip = (int(analyzer.grinder_tip[0]), int(analyzer.grinder_tip[1]))
#                         with self.grinder_lock:
#                             self.grinder_tip_stored = detected_grinder_tip
#                         self.save_grinder_position(detected_grinder_tip)
#                         print(f"✓ Grinder updated: {detected_grinder_tip}")
#
#                     # Get stored grinder position
#                     with self.grinder_lock:
#                         grinder_tip_to_use = self.grinder_tip_stored
#
#                     # Calculate blade angle (optimized)
#                     blade_angle_info = None
#                     if hasattr(analyzer, "blade_edge_points") and analyzer.blade_edge_points is not None:
#                         pts = np.array(analyzer.blade_edge_points)
#                         if len(pts) > 2:
#                             fit_result = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
#                             vx, vy = float(fit_result[0][0]), float(fit_result[1][0])
#                             x0, y0 = float(fit_result[2][0]), float(fit_result[3][0])
#                             angle_deg = float(np.degrees(np.pi / 2 - np.arctan2(vy, vx)))
#
#                             blade_angle_info = {
#                                 'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0,
#                                 'angle_from_y_deg': angle_deg
#                             }
#
#                     # Extract tooth profiles
#                     analyzer.teeth_profiles = analyzer.extract_tooth_profiles(
#                         window_size=self.config['window_size'],
#                         min_depth_px=self.config['min_depth_px']
#                     )
#
#                     # Generate coordinates
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
#                         'grinder_tip': grinder_tip_to_use,
#                         'blade_edge_points': analyzer.blade_edge_points,
#                         'teeth_profiles': analyzer.teeth_profiles,
#                         'blade_angle': blade_angle_info,
#                         'grinder_updated_this_cycle': should_update_grinder
#                     }
#
#                     with self.results_lock:
#                         self.latest_results = results
#
#                     # Publish to TCP clients (only if clients connected)
#                     if self.tcp_server and self.tcp_server.running and self.tcp_server.get_client_count() > 0:
#                         self.tcp_server.publish_data(results)
#                         # REMOVED: Unnecessary read_client call
#
#                 except Exception as e:
#                     print(f"Detection error: {e}")
#
#                 last_detection_time = current_time
#             else:
#                 time.sleep(0.01)  # Reduced from 0.5s
#
#     def draw_overlay(self, frame):
#         """Draw detection overlay on frame (optimized)"""
#         overlay = frame.copy()
#
#         with self.results_lock:
#             if self.latest_results is None:
#                 return overlay
#             results = self.latest_results.copy()  # Copy to minimize lock time
#
#         try:
#             # Draw grinder tip
#             grinder_tip = results['grinder_tip']
#             if grinder_tip is not None:
#                 cv2.circle(overlay, grinder_tip, 12, (0, 255, 255), 3)
#                 cv2.circle(overlay, grinder_tip, 15, (0, 0, 0), 2)
#
#                 label = "GRINDER (UPDATED!)" if results.get('grinder_updated_this_cycle') else "GRINDER (STORED)"
#                 label_color = (0, 255, 0) if results.get('grinder_updated_this_cycle') else (0, 255, 255)
#                 cv2.putText(overlay, label, (grinder_tip[0] + 20, grinder_tip[1] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
#
#             # Blade angle
#             blade_angle = results.get('blade_angle')
#             if blade_angle:
#                 vx, vy, x0, y0 = blade_angle['vx'], blade_angle['vy'], blade_angle['x0'], blade_angle['y0']
#                 pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
#                 pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
#                 cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
#                 cv2.putText(overlay, f"Angle: {blade_angle['angle_from_y_deg']:.2f}°",
#                             (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#
#             # Grooves (optimized drawing)
#             for coord in results['grinding_coordinates']:
#                 pos = (coord['groove_position_x_px'], coord['groove_position_y_px'])
#
#                 cv2.circle(overlay, pos, 10, (255, 0, 255), -1)
#                 cv2.circle(overlay, pos, 12, (255, 255, 255), 2)
#
#                 cv2.putText(overlay, f"#{coord['tooth_id']}", (pos[0] - 30, pos[1] - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#
#                 offset_text = f"({coord['move_x_mm']:+.1f}, {coord['move_y_mm']:+.1f})mm"
#                 text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                 box_x, box_y = pos[0] - 35, pos[1] - 40
#
#                 cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
#                               (box_x + text_size[0] + 5, box_y + 5), (0, 0, 0), -1)
#                 cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
#                               (box_x + text_size[0] + 5, box_y + 5), (0, 255, 255), 2)
#                 cv2.putText(overlay, offset_text, (box_x, box_y - 3),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#
#                 if grinder_tip:
#                     cv2.arrowedLine(overlay, pos, grinder_tip, (0, 255, 255), 2, tipLength=0.02)
#
#             # Info panel
#             tcp_clients = self.tcp_server.get_client_count() if self.tcp_server else 0
#             panel_height = 90 + (results['num_grooves'] * 30)
#
#             cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
#             cv2.rectangle(overlay, (10, 10), (500, panel_height), (255, 255, 255), 2)
#
#             y = 30
#             cv2.putText(overlay, f"FPS: {self.fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             y += 30
#
#             tcp_color = (0, 255, 0) if tcp_clients > 0 else (128, 128, 128)
#             cv2.putText(overlay, f"TCP Clients: {tcp_clients}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcp_color, 2)
#             y += 30
#
#             cv2.putText(overlay, f"Grooves: {results['num_grooves']}", (20, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             y += 30
#
#             if results['grinding_coordinates']:
#                 cv2.putText(overlay, "Blade Offsets:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#                 y += 25
#
#                 for coord in results['grinding_coordinates']:
#                     line = f"  #{coord['tooth_id']}: ({coord['move_x_mm']:+6.1f}, {coord['move_y_mm']:+6.1f})mm  [{coord['distance_to_grinder_mm']:5.1f}mm]"
#                     cv2.putText(overlay, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
#                     y += 22
#
#         except Exception as e:
#             print(f"Overlay error: {e}")
#
#         return overlay
#
#     def run(self):
#         """Main run loop"""
#         print("\n" + "=" * 70)
#         print("OPTIMIZED REAL-TIME BLADE DETECTION")
#         print("=" * 70)
#
#         if not self.initialize_camera():
#             print("Failed to initialize camera!")
#             return
#
#         if self.tcp_server:
#             if not self.tcp_server.start():
#                 print("Warning: TCP server failed to start")
#
#         self.running = True
#
#         # Start threads
#         Thread(target=self.capture_thread, daemon=True).start()
#         Thread(target=self.detection_thread, daemon=True).start()
#
#         print("\n✓ System running!")
#         print("  'q' = quit | 's' = save | 'g' = update grinder | 'r' = reset")
#         if self.tcp_server:
#             print(f"  TCP: {self.config.get('tcp_host')}:{self.config.get('tcp_port')}")
#
#         cv2.namedWindow('Blade Detection', cv2.WINDOW_NORMAL)
#
#         try:
#             while self.running:
#                 with self.frame_lock:
#                     display_frame = self.frame
#
#                 if display_frame is None:
#                     time.sleep(0.01)
#                     continue
#
#                 display_frame = self.draw_overlay(display_frame)
#
#                 # Update FPS
#                 elapsed = time.time() - self.start_time
#                 if elapsed > 0:
#                     self.fps = self.frame_count / elapsed
#
#                 # Resize if needed
#                 h, w = display_frame.shape[:2]
#                 if w > 1280:
#                     display_frame = cv2.resize(display_frame, (1280, int(h * 1280 / w)))
#
#                 cv2.imshow('Blade Detection', display_frame)
#
#                 key = cv2.waitKey(1) & 0xFF
#
#                 if key == ord('q'):
#                     print("\nShutting down...")
#                     self.running = False
#                 elif key == ord('s'):
#                     filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#                     cv2.imwrite(filename, display_frame)
#                     print(f"✓ Saved: {filename}")
#                 elif key == ord('g'):
#                     with self.grinder_lock:
#                         self.detect_grinder_flag = True
#                     print("✓ Grinder update triggered")
#                 elif key == ord('r'):
#                     with self.results_lock:
#                         self.latest_results = None
#                     print("✓ Detection reset")
#
#         except KeyboardInterrupt:
#             print("\n\nInterrupted")
#             self.running = False
#         finally:
#             self.cleanup()
#
#     def cleanup(self):
#         """Clean up resources"""
#         print("Cleaning up...")
#         self.running = False
#         time.sleep(0.2)
#
#         if self.tcp_server:
#             self.tcp_server.stop()
#
#         try:
#             if self.cam:
#                 self.cam.EndAcquisition()
#                 self.cam.DeInit()
#                 del self.cam
#             if self.cam_list:
#                 self.cam_list.Clear()
#             if self.system:
#                 self.system.ReleaseInstance()
#         except Exception as e:
#             print(f"Cleanup error: {e}")
#
#         cv2.destroyAllWindows()
#         print("✓ Cleanup complete")
#
#
# def main():
#     config = {
#         'pixels_per_mm': 86.96,
#         'window_size': 50,
#         'min_depth_px': 200,
#         'detection_fps': 2.0,
#         'frame_rate': 10.0,
#         'exposure_time': 7000,
#         'gain': 0.0,
#         'grinder_position_file': 'grinder_position.json',
#         'tcp_enabled': True,
#         'tcp_host': '172.24.9.15',
#         'tcp_port': 5000,
#         'tcp_max_clients': 5,
#     }
#
#     detector = RealtimeBladeDetector(config)
#     detector.run()
#
#
# if __name__ == "__main__":
#     main()

"""
Optimized Real-Time Tooth Point Detection System
Detects sharp tooth points (peaks) instead of grooves
Performance optimized for real-time operation
"""

import cv2
import numpy as np
import PySpin
import time
from datetime import datetime
from threading import Thread, Lock
from serrated_blade_detector import SerratedBladeAnalyzer
import json
import os
from tcp_blade_server import BladeDataTCPServer


class RealtimeBladeDetector:
    """Optimized real-time tooth point detection with network publishing"""

    def __init__(self, config):
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
            self.tcp_server = BladeDataTCPServer(
                host=config.get('tcp_host', '0.0.0.0'),
                port=config.get('tcp_port', 5000),
                max_clients=config.get('tcp_max_clients', 5)
            )

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
            print(f"✓ Saved grinder position: {grinder_tip}")
        except Exception as e:
            print(f"⚠ Could not save grinder position: {e}")

    def initialize_camera(self):
        """Initialize FLIR camera using PySpin"""
        print("Initializing FLIR camera...")

        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()

            if self.cam_list.GetSize() == 0:
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
        try:
            nodemap = self.cam.GetNodeMap()

            # Continuous acquisition mode
            node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if PySpin.IsWritable(node_acq_mode):
                node_continuous = node_acq_mode.GetEntryByName('Continuous')
                node_acq_mode.SetIntValue(node_continuous.GetValue())

            # Frame rate
            if 'frame_rate' in self.config:
                node_fr_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
                if PySpin.IsWritable(node_fr_enable):
                    node_fr_enable.SetValue(True)

                node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsWritable(node_fr):
                    node_fr.SetValue(min(node_fr.GetMax(), self.config['frame_rate']))

            # Exposure
            if 'exposure_time' in self.config:
                node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
                if PySpin.IsWritable(node_exp_auto):
                    node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())

                node_exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
                if PySpin.IsWritable(node_exp):
                    node_exp.SetValue(min(node_exp.GetMax(), self.config['exposure_time']))

            # Gain
            if 'gain' in self.config:
                node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
                if PySpin.IsWritable(node_gain_auto):
                    node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())

                node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
                if PySpin.IsWritable(node_gain):
                    node_gain.SetValue(min(node_gain.GetMax(), self.config['gain']))

        except PySpin.SpinnakerException as ex:
            print(f"ERROR configuring camera: {ex}")

    def capture_thread(self):
        """Thread for continuously capturing frames"""
        print("Starting capture thread...")

        # Pre-create processor for efficiency
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        while self.running:
            try:
                image_result = self.cam.GetNextImage(1000)

                if not image_result.IsIncomplete():
                    pixel_format = image_result.GetPixelFormat()

                    # Fast conversion based on pixel format
                    if pixel_format == PySpin.PixelFormat_BGR8:
                        frame = image_result.GetNDArray()
                    elif pixel_format == PySpin.PixelFormat_Mono8:
                        frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
                    else:
                        # Use pre-created processor
                        image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                        frame = image_converted.GetNDArray()

                    # Single rotation operation
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    with self.frame_lock:
                        self.frame = frame
                        self.frame_count += 1

                image_result.Release()

            except PySpin.SpinnakerException:
                time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def detection_thread(self):
        """Thread for running tooth point detection and publishing data"""
        print("Starting detection thread...")

        detection_interval = 1.0 / self.config['detection_fps']
        last_detection_time = 0

        while self.running:
            current_time = time.time()

            if current_time - last_detection_time >= detection_interval:
                # Quick frame grab
                with self.frame_lock:
                    frame_to_process = self.frame

                if frame_to_process is None:
                    time.sleep(0.1)
                    continue

                # Check grinder update flag
                with self.grinder_lock:
                    should_update_grinder = self.detect_grinder_flag
                    if should_update_grinder:
                        self.detect_grinder_flag = False

                try:
                    # Pass frame directly to analyzer
                    analyzer = SerratedBladeAnalyzer(frame_to_process)
                    analyzer.preprocess_image()
                    analyzer.detect_edges()
                    analyzer.find_blade_contours()
                    analyzer.detect_blade_and_grinder()

                    # Update grinder position if requested
                    if should_update_grinder and analyzer.grinder_tip is not None:
                        detected_grinder_tip = (int(analyzer.grinder_tip[0]), int(analyzer.grinder_tip[1]))
                        with self.grinder_lock:
                            self.grinder_tip_stored = detected_grinder_tip
                        self.save_grinder_position(detected_grinder_tip)
                        print(f"✓ Grinder updated: {detected_grinder_tip}")

                    # Get stored grinder position
                    with self.grinder_lock:
                        grinder_tip_to_use = self.grinder_tip_stored

                    # Calculate blade angle
                    blade_angle_info = None
                    if hasattr(analyzer, "blade_edge_points") and analyzer.blade_edge_points is not None:
                        pts = np.array(analyzer.blade_edge_points)
                        if len(pts) > 2:
                            fit_result = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                            vx, vy = float(fit_result[0][0]), float(fit_result[1][0])
                            x0, y0 = float(fit_result[2][0]), float(fit_result[3][0])
                            angle_deg = float(np.degrees(np.pi / 2 - np.arctan2(vy, vx)))

                            blade_angle_info = {
                                'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0,
                                'angle_from_y_deg': angle_deg
                            }

                    # Extract tooth profiles (TOOTH POINTS now!)
                    analyzer.teeth_profiles = analyzer.extract_tooth_profiles(
                        window_size=self.config['window_size'],
                        min_height_px=self.config.get('min_height_px', self.config.get('min_depth_px', 100))
                    )

                    # Generate coordinates
                    analyzer.grinder_tip = grinder_tip_to_use
                    grinding_coords = analyzer.generate_grinding_coordinates(
                        pixels_per_mm=self.config['pixels_per_mm']
                    )

                    # Store results
                    results = {
                        'timestamp': datetime.now(),
                        'num_grooves': len(analyzer.teeth_profiles),  # Keep for compatibility
                        'num_teeth': len(analyzer.teeth_profiles),
                        'grinding_coordinates': grinding_coords,
                        'grinder_tip': grinder_tip_to_use,
                        'blade_edge_points': analyzer.blade_edge_points,
                        'teeth_profiles': analyzer.teeth_profiles,
                        'blade_angle': blade_angle_info,
                        'grinder_updated_this_cycle': should_update_grinder
                    }

                    with self.results_lock:
                        self.latest_results = results

                    if self.tcp_server and self.tcp_server.running and self.tcp_server.get_client_count() > 0:
                        self.tcp_server.publish_data(results)

                except Exception as e:
                    print(f"Detection error: {e}")

                last_detection_time = current_time
            else:
                time.sleep(0.01)

    def draw_overlay(self, frame):
        """Draw detection overlay on frame - TOOTH POINTS"""
        overlay = frame.copy()

        with self.results_lock:
            if self.latest_results is None:
                return overlay
            results = self.latest_results.copy()

        try:
            # Draw grinder tip
            grinder_tip = results['grinder_tip']
            if grinder_tip is not None:
                cv2.circle(overlay, grinder_tip, 12, (0, 255, 255), 3)
                cv2.circle(overlay, grinder_tip, 15, (0, 0, 0), 2)

                label = "GRINDER (UPDATED!)" if results.get('grinder_updated_this_cycle') else "GRINDER (STORED)"
                label_color = (0, 255, 0) if results.get('grinder_updated_this_cycle') else (0, 255, 255)
                cv2.putText(overlay, label, (grinder_tip[0] + 20, grinder_tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

            # Blade angle
            blade_angle = results.get('blade_angle')
            if blade_angle:
                vx, vy, x0, y0 = blade_angle['vx'], blade_angle['vy'], blade_angle['x0'], blade_angle['y0']
                pt1 = (int(x0 - vx * 1000), int(y0 - vy * 1000))
                pt2 = (int(x0 + vx * 1000), int(y0 + vy * 1000))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(overlay, f"Angle: {blade_angle['angle_from_y_deg']:.2f}°",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Tooth points (cyan circles)
            for coord in results['grinding_coordinates']:
                # Use tooth_tip if available, otherwise groove_position for compatibility
                pos_x = coord.get('tooth_tip_x_px', coord.get('groove_position_x_px'))
                pos_y = coord.get('tooth_tip_y_px', coord.get('groove_position_y_px'))
                pos = (pos_x, pos_y)

                # Cyan for tooth tips
                cv2.circle(overlay, pos, 10, (255, 255, 0), -1)  # Yellow/cyan
                cv2.circle(overlay, pos, 12, (255, 255, 255), 2)

                cv2.putText(overlay, f"T#{coord['tooth_id']}", (pos[0] - 30, pos[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                offset_text = f"({coord['move_x_mm']:+.1f}, {coord['move_y_mm']:+.1f})mm"
                text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                box_x, box_y = pos[0] - 35, pos[1] - 40

                cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5), (0, 0, 0), -1)
                cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                              (box_x + text_size[0] + 5, box_y + 5), (0, 255, 255), 2)
                cv2.putText(overlay, offset_text, (box_x, box_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if grinder_tip:
                    cv2.arrowedLine(overlay, pos, grinder_tip, (0, 255, 255), 2, tipLength=0.02)

            # Info panel
            tcp_clients = self.tcp_server.get_client_count() if self.tcp_server else 0
            num_items = results.get('num_teeth', results.get('num_grooves', 0))
            panel_height = 90 + (num_items * 30)

            cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (500, panel_height), (255, 255, 255), 2)

            y = 30
            cv2.putText(overlay, f"FPS: {self.fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30

            tcp_color = (0, 255, 0) if tcp_clients > 0 else (128, 128, 128)
            cv2.putText(overlay, f"TCP Clients: {tcp_clients}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tcp_color, 2)
            y += 30

            cv2.putText(overlay, f"Teeth: {num_items}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

            if results['grinding_coordinates']:
                cv2.putText(overlay, "Tooth Points:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y += 25

                for coord in results['grinding_coordinates']:
                    line = f"  T#{coord['tooth_id']}: ({coord['move_x_mm']:+6.1f}, {coord['move_y_mm']:+6.1f})mm  [{coord['distance_to_grinder_mm']:5.1f}mm]"
                    cv2.putText(overlay, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y += 22

        except Exception as e:
            print(f"Overlay error: {e}")

        return overlay

    def run(self):
        """Main run loop"""
        print("\n" + "=" * 70)
        print("REAL-TIME TOOTH POINT DETECTION")
        print("=" * 70)

        if not self.initialize_camera():
            print("Failed to initialize camera!")
            return

        if self.tcp_server:
            if not self.tcp_server.start():
                print("Warning: TCP server failed to start")

        self.running = True

        # Start threads
        Thread(target=self.capture_thread, daemon=True).start()
        Thread(target=self.detection_thread, daemon=True).start()

        print("\n✓ System running!")
        print("  'q' = quit | 's' = save | 'g' = update grinder | 'r' = reset")
        if self.tcp_server:
            print(f"  TCP: {self.config.get('tcp_host')}:{self.config.get('tcp_port')}")

        cv2.namedWindow('Tooth Point Detection', cv2.WINDOW_NORMAL)

        try:
            while self.running:
                with self.frame_lock:
                    display_frame = self.frame

                if display_frame is None:
                    time.sleep(0.01)
                    continue

                display_frame = self.draw_overlay(display_frame)

                # Update FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                # Resize if needed
                h, w = display_frame.shape[:2]
                if w > 1280:
                    display_frame = cv2.resize(display_frame, (1280, int(h * 1280 / w)))

                cv2.imshow('Tooth Point Detection', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nShutting down...")
                    self.running = False
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
        time.sleep(0.2)

        if self.tcp_server:
            self.tcp_server.stop()

        try:
            if self.cam:
                self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam
            if self.cam_list:
                self.cam_list.Clear()
            if self.system:
                self.system.ReleaseInstance()
        except Exception as e:
            print(f"Cleanup error: {e}")

        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    config = {
        'pixels_per_mm': 86.96,
        'window_size': 20,
        'min_height_px': 50,  # Changed from min_depth_px
        'min_depth_px': 50,   # Keep for backward compatibility
        'detection_fps': 2.0,
        'frame_rate': 10.0,
        'exposure_time': 5000,
        'gain': 0.0,
        'grinder_position_file': 'grinder_position.json',
        'tcp_enabled': True,
        'tcp_host': '172.24.9.15',
        'tcp_port': 5000,
        'tcp_max_clients': 5,
    }

    detector = RealtimeBladeDetector(config)
    detector.run()


if __name__ == "__main__":
    main()