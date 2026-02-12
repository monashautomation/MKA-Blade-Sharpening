"""
Restructured Real-Time Blade Detection System
Protocol:
1. Get configuration from user via CLI
2. Start system and send configuration to clients
3. Wait for client requests before sending detection data
"""

from threading import Thread, Lock
import time, json, os
import PySpin
import cv2
from datetime import datetime
from serrated_blade_detector import SerratedBladeAnalyzer
from modbus_blade_client import BladeDataModbusClient  # <-- new client class


class RealtimeBladeDetectorV2:
    """Real-time detector with Modbus client protocol (writes to robot)"""

    def __init__(self, config, initial_blade_config):
        self.config = config
        self.blade_config = initial_blade_config  # Current BAY_ID, GRINDER_ID, etc.
        self.running = False
        self.frame = None
        self.frame_lock = Lock()
        self.latest_results = None
        self.results_lock = Lock()

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Grinder tip control
        self.grinder_tip_stored = None
        self.grinder_position_file = config.get('grinder_position_file', 'grinder_position.json')
        self.detect_grinder_flag = False
        self.grinder_lock = Lock()

        # Modbus Client to robot
        self.modbus_client = None
        if config.get('modbus_enabled', True):
            robot_host = config.get('modbus_host', '172.24.89.89')
            robot_port = config.get('modbus_port', 502)
            self.modbus_client = BladeDataModbusClient(host=robot_host, port=robot_port)

            # Send initial configuration
            self.update_blade_configuration(initial_blade_config)

        self.load_grinder_position()

    def load_grinder_position(self):
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

    def update_blade_configuration(self, new_config):
        """Update blade configuration and write to robot via Modbus client"""
        with Lock():
            self.blade_config = new_config

        if self.modbus_client:
            self.modbus_client.write_configuration(
                bay_id=new_config['bay_id'],
                grinder_id=new_config['grinder_id'],
                angle=new_config['angle'],
                depth=new_config['depth'],
                length=new_config['length'],
                config_version=1
            )

        print(f"✓ Updated blade configuration sent to robot: {new_config}")


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

        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        while self.running:
            try:
                image_result = self.cam.GetNextImage(1000)

                if not image_result.IsIncomplete():
                    pixel_format = image_result.GetPixelFormat()

                    if pixel_format == PySpin.PixelFormat_BGR8:
                        frame = image_result.GetNDArray()
                    elif pixel_format == PySpin.PixelFormat_Mono8:
                        frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
                    else:
                        image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                        frame = image_converted.GetNDArray()

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
        """Run tooth detection and send results to robot"""
        detection_interval = 1.0 / self.config['detection_fps']
        last_detection_time = 0

        while self.running:
            current_time = time.time()
            if current_time - last_detection_time < detection_interval:
                time.sleep(0.01)
                continue
            last_detection_time = current_time

            with self.frame_lock:
                frame_to_process = self.frame

            if frame_to_process is None:
                continue

            try:
                # Grinder update logic
                update_grinder = False
                with self.grinder_lock:
                    if self.detect_grinder_flag:
                        update_grinder = True
                        self.detect_grinder_flag = False

                analyzer = SerratedBladeAnalyzer(frame_to_process)
                results = analyzer.analyze_blade(pixels_per_mm=self.config['pixels_per_mm'])

                if update_grinder and results.get('grinder_tip'):
                    self.grinder_tip_stored = results['grinder_tip']
                    self.save_grinder_position(self.grinder_tip_stored)
                    results['grinder_updated_this_cycle'] = True
                elif self.grinder_tip_stored:
                    results['grinder_tip'] = self.grinder_tip_stored
                    results['grinder_updated_this_cycle'] = False

                with self.results_lock:
                    self.latest_results = results

                # Publish detection to robot
                if self.modbus_client:
                    closest_valley = self.compute_closest_valley(results)
                    if closest_valley:
                        self.modbus_client.write_detection(
                            x_mm=closest_valley['move_y_mm'],
                            y_mm=closest_valley['move_x_mm'],
                            status=1
                        )
                        self.modbus_client.write_command(11)  # Robot read_detection
                    else:
                        self.modbus_client.write_detection(0, 0, 2)  # No detection

            except Exception as e:
                print(f"Detection error: {e}")

    # def compute_closest_valley(self, blade_results):
    #     """
    #     Find the tooth tip closest to the grinder tip.
    #     Matches the request-response logic from the TCP version.
    #     """
    #     grinding_coords = blade_results.get('grinding_coordinates', [])
    #
    #     if not grinding_coords:
    #         return None
    #
    #     closest_tooth = None
    #     min_distance = float('inf')
    #
    #     for coord in grinding_coords:
    #         # mx/my are already calculated by the Analyzer class
    #         mx = coord['move_x_mm']
    #         my = coord['move_y_mm']
    #
    #         # Calculate distance to grinder
    #         dist = (mx ** 2 + my ** 2) ** 0.5
    #
    #         # Filter: Tooth must be 'before' the grinder (my > 0)
    #         # We use a small buffer (0.1mm) to keep detection active near the contact point
    #         if my > 0.5:
    #             if dist < min_distance:
    #                 min_distance = dist
    #                 closest_tooth = {
    #                     'move_x_mm': mx,  # The SWAP: Y-offset is robot's X (feed)
    #                     'move_y_mm': my,  # The SWAP: X-offset is robot's Y (depth)
    #                 }
    #
    #     return closest_tooth
    def compute_closest_valley(self, blade_results):
        grinding_coords = blade_results.get('grinding_coordinates', [])
        if not grinding_coords:
            return None

        # Check if we even have a grinder position stored
        if self.grinder_tip_stored is None:
            print("⚠ Cannot compute: No grinder tip position stored. Press 'g' to detect.")
            return None

        closest_tooth = None
        min_distance = float('inf')

        for coord in grinding_coords:
            # These values represent the relative distance TO the grinder
            # calculated as: (Grinder_px - Tooth_px) / scale
            mx = coord['move_x_mm']
            my = coord['move_y_mm']

            # The distance from this tooth to the grinder
            dist = (mx ** 2 + my ** 2) ** 0.5

            # Filter: Only pick teeth that are 'approaching' the grinder
            # If my > 0, the tooth is still 'behind' the grinder in the feed direction
            if my > 0.5:
                if dist < min_distance:
                    min_distance = dist
                    closest_tooth = {
                        'move_x_mm': mx,
                        'move_y_mm': my,
                    }

        return closest_tooth
    def draw_overlay(self, frame):
        """Draw detection overlay on frame"""
        overlay = frame.copy()

        try:
            with self.results_lock:
                results = self.latest_results

            if results is None:
                return overlay

            grinder_tip = results.get('grinder_tip')

            # Draw grinder tip
            # Draw grinder tip safely
            grinder_tip = results.get('grinder_tip')
            if grinder_tip and isinstance(grinder_tip, (tuple, list)) and len(grinder_tip) == 2:
                grinder_tip_int = (int(grinder_tip[0]), int(grinder_tip[1]))
                cv2.circle(overlay, grinder_tip_int, 12, (0, 255, 255), 3)
                cv2.circle(overlay, grinder_tip_int, 15, (0, 0, 0), 2)

                label = "GRINDER (UPDATED!)" if results.get('grinder_updated_this_cycle') else "GRINDER (STORED)"
                label_color = (0, 255, 0) if results.get('grinder_updated_this_cycle') else (0, 255, 255)
                cv2.putText(overlay, label, (grinder_tip_int[0] + 20, grinder_tip_int[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

            # Blade angle
            # --- Corrected Blade angle section ---
            blade_angle = results.get('blade_angle')
            if blade_angle is not None:
                # Since analyzer returns a float, just display the text
                # (If you need to draw the line, you'd need to update the analyzer to return the fitLine points)
                cv2.putText(overlay, f"Angle: {blade_angle:.2f} deg",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Tooth points
            for coord in results.get('grinding_coordinates', []):
                pos_x = coord.get('tooth_tip_x_px', coord.get('groove_position_x_px', 0))
                pos_y = coord.get('tooth_tip_y_px', coord.get('groove_position_y_px', 0))
                pos = (int(pos_x), int(pos_y))
                cv2.circle(overlay, pos, 10, (255, 255, 0), -1)
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
            # modbus_status = "ACTIVE" if self.modbus_server and self.modbus_server.running else "INACTIVE"
            num_items = results.get('num_teeth', results.get('num_grooves', 0))

            # Add blade config to panel
            panel_height = 150 + (num_items * 30)

            cv2.rectangle(overlay, (10, 10), (550, panel_height), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (550, panel_height), (255, 255, 255), 2)

            y = 30
            cv2.putText(overlay, f"FPS: {self.fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30

            # modbus_color = (0, 255, 0) if modbus_status == "ACTIVE" else (128, 128, 128)
            # cv2.putText(overlay, f"Modbus: {modbus_status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, modbus_color, 2)
            y += 30

            # Display blade configuration
            cv2.putText(overlay, f"Bay: {self.blade_config['bay_id']} | Grinder: {self.blade_config['grinder_id']}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            y += 30
            cv2.putText(overlay, f"Angle: {self.blade_config['angle']:.1f}° | Depth: {self.blade_config['depth']:.2f} | Len: {self.blade_config['length']}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
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
        print("REAL-TIME TOOTH POINT DETECTION - MODBUS PROTOCOL")
        print("=" * 70)
        print(f"\nBlade Configuration:")
        print(f"  Bay ID:      {self.blade_config['bay_id']}")
        print(f"  Grinder ID:  {self.blade_config['grinder_id']}")
        print(f"  Angle:       {self.blade_config['angle']:.1f}°")
        print(f"  Depth:       {self.blade_config['depth']:.2f}")
        print(f"  Length:      {self.blade_config['length']}")

        if not self.initialize_camera():
            print("Failed to initialize camera!")
            return
        self.running = True

        # Start threads
        Thread(target=self.capture_thread, daemon=True).start()
        Thread(target=self.detection_thread, daemon=True).start()

        print("\n✓ System running!")
        print("  'q' = quit | 's' = save | 'g' = update grinder | 'r' = reset | 'c' = new config")
        print("  Robot can read Modbus registers for detection data")

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
                elif key == ord('c'):
                    # Request new configuration
                    print("\n" + "="*70)
                    print("REQUESTING NEW CONFIGURATION")
                    print("="*70)
                    new_config = get_user_configuration_inline()
                    if new_config:
                        self.update_blade_configuration(new_config)

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


def get_user_configuration():
    """Get blade configuration from user via CLI"""
    print("\n" + "=" * 70)
    print("BLADE CONFIGURATION")
    print("=" * 70)

    while True:
        try:
            bay_id = int(input("Enter BAY ID (1-10): "))
            if 1 <= bay_id <= 10:
                break
            print("❌ BAY ID must be between 1 and 10")
        except ValueError:
            print("❌ Please enter a valid number")

    while True:
        try:
            grinder_id = int(input("Enter GRINDER ID (1-3): "))
            if 1 <= grinder_id <= 3:
                break
            print("❌ GRINDER ID must be between 1 and 3")
        except ValueError:
            print("❌ Please enter a valid number")

    while True:
        try:
            angle = float(input("Enter ANGLE (degrees, e.g., 45.5): "))
            break
        except ValueError:
            print("❌ Please enter a valid number")

    while True:
        try:
            depth = float(input("Enter DEPTH (e.g., 1.25): "))
            break
        except ValueError:
            print("❌ Please enter a valid number")

    while True:
        try:
            length = int(input("Enter LENGTH (0-999): "))
            if 0 <= length <= 999:
                break
            print("❌ LENGTH must be between 0 and 999")
        except ValueError:
            print("❌ Please enter a valid number")

    config = {
        'bay_id': bay_id,
        'grinder_id': grinder_id,
        'angle': angle,
        'depth': depth,
        'length': length
    }

    print("\n" + "=" * 70)
    print("Configuration Summary:")
    print(f"  Bay ID:      {config['bay_id']}")
    print(f"  Grinder ID:  {config['grinder_id']}")
    print(f"  Angle:       {config['angle']:.1f}°")
    print(f"  Depth:       {config['depth']:.2f}")
    print(f"  Length:      {config['length']}")
    print("=" * 70)

    confirm = input("\nProceed with this configuration? (y/n): ").lower()
    if confirm != 'y':
        print("Configuration cancelled. Exiting.")
        exit(0)

    return config


def get_user_configuration_inline():
    """
    Get blade configuration from user via CLI (inline - no confirmation needed)
    Used when updating configuration during runtime
    """
    print("\nEnter new blade configuration:")

    while True:
        try:
            bay_id = int(input("  BAY ID (1-10): "))
            if 1 <= bay_id <= 10:
                break
            print("  ❌ BAY ID must be between 1 and 10")
        except ValueError:
            print("  ❌ Please enter a valid number")

    while True:
        try:
            grinder_id = int(input("  GRINDER ID (1-3): "))
            if 1 <= grinder_id <= 3:
                break
            print("  ❌ GRINDER ID must be between 1 and 3")
        except ValueError:
            print("  ❌ Please enter a valid number")

    while True:
        try:
            angle = float(input("  ANGLE (degrees, e.g., 45.5): "))
            break
        except ValueError:
            print("  ❌ Please enter a valid number")

    while True:
        try:
            depth = float(input("  DEPTH (e.g., 1.25): "))
            break
        except ValueError:
            print("  ❌ Please enter a valid number")

    while True:
        try:
            length = int(input("  LENGTH (0-999): "))
            if 0 <= length <= 999:
                break
            print("  ❌ LENGTH must be between 0 and 999")
        except ValueError:
            print("  ❌ Please enter a valid number")

    config = {
        'bay_id': bay_id,
        'grinder_id': grinder_id,
        'angle': angle,
        'depth': depth,
        'length': length
    }

    return config


def main():
    # Get blade configuration from user
    blade_config = get_user_configuration()

    # Camera and detection configuration
    system_config = {
        'pixels_per_mm': 86.96,
        'window_size': 20,
        'min_height_px': 50,
        'min_depth_px': 50,
        'detection_fps': 2.0,
        'frame_rate': 10.0,
        'exposure_time': 5000,
        'gain': 0.0,
        'grinder_position_file': 'grinder_position.json',
        'modbus_enabled': True,
        'modbus_host': '172.24.89.89',  # Robot's Modbus server IP
        'modbus_port': 502,  # Standard Modbus TCP port
    }

    # Initialize detector with configuration
    detector = RealtimeBladeDetectorV2(system_config, blade_config)
    detector.run()


if __name__ == "__main__":
    main()
