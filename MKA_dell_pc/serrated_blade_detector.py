import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ToothProfile:
    """Data class to store information about each tooth/serration"""
    tooth_id: int  # Tooth number
    apex_point: Tuple[int, int]  # GROOVE CENTER (valley - the grinding target!)
    top_valley: Tuple[int, int]  # Top tooth peak
    bottom_valley: Tuple[int, int]  # Bottom tooth peak
    angle: float  # Angle of the groove in degrees
    grinding_point: Tuple[int, int]  # Point to align with grinder (the groove center)
    depth: float  # Depth of the groove
    move_to_grinder: Tuple[float, float]  # X,Y movement to align groove with grinder tip


class SerratedBladeAnalyzer:
    """Analyzer for detecting serrated blade edges and calculating grinding coordinates"""

    def __init__(self, image_path: str):
        """
        Initialize the analyzer with an image

        Args:
            image_path: Path to the blade image
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self.teeth_profiles = []
        self.blade_edge_points = None
        self.grinder_tip = None
        self.grinder_center = None

    def preprocess_image(self, blur_kernel=31, threshold_method='adaptive'):
        """
        Preprocess the image for edge detection

        Args:
            blur_kernel: Size of Gaussian blur kernel
            threshold_method: 'adaptive' or 'otsu'
        """
        # Apply Gaussian blur to reduce noise
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)

        # Apply thresholding
        if threshold_method == 'adaptive':
            self.binary = cv2.adaptiveThreshold(
                self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, self.binary = cv2.threshold(
                self.blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        return self.binary

    def detect_edges(self, canny_low=100, canny_high=150):
        """
        Detect edges using Canny edge detection

        Args:
            canny_low: Lower threshold for Canny
            canny_high: Upper threshold for Canny
        """
        self.edges = cv2.Canny(self.blurred, canny_low, canny_high)
        return self.edges

    def find_blade_contours(self):
        """Find contours of the serrated blade"""
        contours, hierarchy = cv2.findContours(
            self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area to get significant ones
        min_area = 50
        self.blade_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        return self.blade_contours

    def detect_blade_and_grinder(self, sampling_step=3):
        """
        Detect the blade edge (left side) and grinder tip (right side)
        The grinder tip is the sharp point facing the blade

        Args:
            sampling_step: Step size for scanning (in pixels)
        """
        blade_edge = []
        grinder_points = []

        # Scan each row to find edge transitions
        for y in range(0, self.height, sampling_step):
            row = self.binary[y, :]

            # Find white pixels (object pixels)
            white_pixels = np.where(row > 170)[0]

            if len(white_pixels) > 0:
                # Leftmost white pixel = blade edge (serrated side)
                blade_x = white_pixels[0]
                blade_edge.append((blade_x, y))

                # For grinder: collect the leftmost points on the right side
                # (the points facing the blade)
                if len(white_pixels) > 20:  # Make sure it's the grinder, not noise
                    grinder_x = white_pixels[-1]
                    # Also get the leftmost point of the grinder (facing blade)
                    rightmost_region = white_pixels[white_pixels > self.width // 3 * 2]
                    if len(rightmost_region) > 0:
                        grinder_left_x = rightmost_region[0]
                        grinder_points.append((grinder_left_x, y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        # Find grinder TIP - the leftmost point of the grinder (pointing toward blade)
        if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
            # The tip is the leftmost point (minimum X) of the grinder
            min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
            self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
            self.grinder_edge = self.grinder_tip  # The tip IS the edge we care about

            # Calculate average of points near the tip for better accuracy
            min_x = self.grinder_edge_points[min_x_idx, 0]
            tip_points = self.grinder_edge_points[
                np.abs(self.grinder_edge_points[:, 0] - min_x) < 15
                ]

            self.grinder_edge_center = (
                int(np.mean(tip_points[:, 0])),
                int(np.mean(tip_points[:, 1]))
            )

            print(f"   Grinder tip (sharp edge facing blade): {self.grinder_tip}")

        return self.blade_edge_points, self.grinder_tip

    def extract_tooth_profiles(self, window_size=30, min_depth_px=100):
        """
        Extract individual tooth profiles from blade edge points
        Identifies grooves (valleys) as the grinding points

        Args:
            window_size: Window size for local maxima/minima detection
            min_depth_px: Minimum groove depth in pixels to filter out noise
        """
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return []

        x_coords = self.blade_edge_points[:, 0]
        y_coords = self.blade_edge_points[:, 1]

        # Smooth the x-coordinates to reduce noise
        x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

        # Find local maxima (peaks - tooth tips pointing right) and minima (valleys/grooves)
        peaks = []  # Tooth tips (points furthest right)
        valleys = []  # Grooves (points furthest left) - THESE ARE WHAT WE GRIND

        for i in range(window_size, len(x_smooth) - window_size):
            window = x_smooth[i - window_size:i + window_size]

            # Tooth tips are local maxima (pointing right = higher x values)
            if x_smooth[i] == np.max(window) and x_smooth[i] > np.mean(x_smooth) + 20:
                peaks.append(i)
            # Valleys/GROOVES are local minima (lower x values) - THE GRINDING TARGETS
            elif x_smooth[i] == np.min(window) and x_smooth[i] < np.mean(x_smooth) - 20:
                valleys.append(i)

        # Remove duplicate peaks/valleys that are too close
        peaks = self._filter_close_points(peaks, min_distance=window_size)
        valleys = self._filter_close_points(valleys, min_distance=window_size)

        print(f"   Found {len(peaks)} tooth peaks and {len(valleys)} grooves/valleys")

        # Create tooth profiles - each valley (groove) becomes a grinding target
        tooth_profiles = []
        tooth_id = 1

        # Each valley (groove) becomes a grinding target
        for i, valley_idx in enumerate(valleys):
            # Find peaks above and below this valley
            peaks_above = [p for p in peaks if p < valley_idx]
            peaks_below = [p for p in peaks if p > valley_idx]

            # Handle edge cases by sampling actual edge points
            if len(peaks_above) == 0 and len(peaks_below) > 0:
                # TOP EDGE GROOVE - sample points above the valley and average them
                edge_sample_size = min(window_size // 2, 50)  # Max 20 points
                sample_start = 0  # Start from absolute top
                sample_end = min(edge_sample_size, valley_idx)
                # sample_start = max(0, valley_idx - window_size * 2)
                # sample_end = valley_idx
                sample_indices = range(sample_start, sample_end)

                # Average the actual edge points in this region
                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                if sampled_x and sampled_y:
                    top_peak = (int(np.mean(sampled_x)), int(np.mean(sampled_y)))
                else:
                    top_peak = (int(x_smooth[sample_start]), int(y_coords[sample_start]))

                bottom_peak_idx = peaks_below[0]
                bottom_peak = (int(x_smooth[bottom_peak_idx]), int(y_coords[bottom_peak_idx]))

                print(f"   Top edge groove: sampled {len(sampled_x)} points for top boundary")

            elif len(peaks_below) == 0 and len(peaks_above) > 0:
                # BOTTOM EDGE GROOVE - sample points below the valley and average them
                sample_start = valley_idx
                sample_end = min(len(x_smooth) - 1, valley_idx + window_size * 2)
                sample_indices = range(sample_start, sample_end)

                # Average the actual edge points in this region
                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                top_peak_idx = peaks_above[-1]
                top_peak = (int(x_smooth[top_peak_idx]), int(y_coords[top_peak_idx]))

                if sampled_x and sampled_y:
                    bottom_peak = (int(np.mean(sampled_x)), int(np.mean(sampled_y)))
                else:
                    bottom_peak = (int(x_smooth[sample_end]), int(y_coords[sample_end]))

                print(f"   Bottom edge groove: sampled {len(sampled_x)} points for bottom boundary")

            elif len(peaks_above) > 0 and len(peaks_below) > 0:
                # MIDDLE GROOVE - normal case, use detected peaks
                top_peak_idx = peaks_above[-1]
                bottom_peak_idx = peaks_below[0]

                top_peak = (int(x_smooth[top_peak_idx]), int(y_coords[top_peak_idx]))
                bottom_peak = (int(x_smooth[bottom_peak_idx]), int(y_coords[bottom_peak_idx]))

            else:
                # No peaks found - skip this groove
                continue

            # The GROOVE is our grinding point (the valley)
            # Instead of using just the single valley point, average nearby points for accuracy
            groove_sample_size = max(3, window_size // 10)  # Sample 3-5 points around valley
            groove_start = max(0, valley_idx - groove_sample_size)
            groove_end = min(len(x_smooth), valley_idx + groove_sample_size + 1)

            groove_x_samples = x_smooth[groove_start:groove_end]
            groove_y_samples = y_coords[groove_start:groove_end]

            if len(groove_x_samples) > 0:
                groove_point = (int(np.mean(groove_x_samples)), int(np.mean(groove_y_samples)))
                print(f"   Groove at y≈{groove_point[1]}: averaged {len(groove_x_samples)} points")
            else:
                groove_point = (int(x_smooth[valley_idx]), int(y_coords[valley_idx]))

            # Calculate groove depth (distance from peaks to valley)
            depth = abs(groove_point[0] - ((top_peak[0] + bottom_peak[0]) / 2))

            # Filter out shallow grooves (noise)
            # if depth < 100:
            #     print(f"   Skipping shallow groove at y={groove_point[1]} (depth={depth:.0f}px < {min_depth_px}px)")
            #     continue

            # Calculate angle of the groove (angle between the two peaks through the valley)
            angle = self._calculate_tooth_angle(top_peak, groove_point, bottom_peak)

            # The grinding point is the GROOVE (valley), not the apex!
            grinding_point = groove_point

            # Calculate movement needed to align this GROOVE with grinder tip
            if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
                move_x = self.grinder_tip[0] - groove_point[0]
                move_y = self.grinder_tip[1] - groove_point[1]
                move_to_grinder = (move_x, move_y)
            else:
                move_to_grinder = (0, 0)

            tooth_profiles.append(ToothProfile(
                tooth_id=tooth_id,
                apex_point=groove_point,  # This is actually the groove/valley!
                top_valley=top_peak,  # Top tooth peak (or averaged boundary)
                bottom_valley=bottom_peak,  # Bottom tooth peak (or averaged boundary)
                angle=angle,
                grinding_point=grinding_point,  # The groove center
                depth=depth,
                move_to_grinder=move_to_grinder
            ))

            tooth_id += 1

        return tooth_profiles

    def _filter_close_points(self, points, min_distance):
        """Filter out points that are too close to each other"""
        if len(points) == 0:
            return points

        filtered = [points[0]]
        for p in points[1:]:
            if abs(p - filtered[-1]) >= min_distance:
                filtered.append(p)

        return filtered

    def _calculate_tooth_angle(self, top_valley, apex, bottom_valley):
        """
        Calculate the angle of a tooth

        Args:
            top_valley: Top valley point
            apex: Apex point (tooth tip)
            bottom_valley: Bottom valley point

        Returns:
            Angle in degrees
        """
        # Calculate vectors from apex to valleys
        v1 = np.array([top_valley[0] - apex[0], top_valley[1] - apex[1]])
        v2 = np.array([bottom_valley[0] - apex[0], bottom_valley[1] - apex[1]])

        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)

        if magnitude == 0:
            return 0

        angle = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
        angle_degrees = np.degrees(angle)

        return angle_degrees

    def calculate_blade_orientation(self):
        """Calculate the overall orientation/angle of the blade edge"""
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return 0

        # Fit a line to the blade edge points
        [vx, vy, x0, y0] = cv2.fitLine(
            self.blade_edge_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )

        # Calculate angle from vertical (since blade is oriented vertically)
        angle = np.degrees(np.arctan2(vy, vx))

        return float(angle[0])

    def generate_grinding_coordinates(self, pixels_per_mm=1.0):
        """
        Generate grinding coordinates for robot control
        Coordinates align each GROOVE with the GRINDER TIP

        Args:
            pixels_per_mm: Calibration factor (pixels per millimeter)

        Returns:
            List of grinding coordinate dictionaries for each groove
        """
        grinding_coords = []

        grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None

        for tooth in self.teeth_profiles:
            # Movement needed in pixels to align GROOVE with GRINDER TIP
            move_x_px = tooth.move_to_grinder[0]
            move_y_px = tooth.move_to_grinder[1]

            # Convert to millimeters
            move_x_mm = move_x_px / pixels_per_mm
            move_y_mm = move_y_px / pixels_per_mm

            # Distance to grinder
            distance_px = np.sqrt(move_x_px ** 2 + move_y_px ** 2)
            distance_mm = distance_px / pixels_per_mm

            grinding_coords.append({
                'tooth_id': tooth.tooth_id,
                'groove_position_x_px': tooth.grinding_point[0],  # Groove center X
                'groove_position_y_px': tooth.grinding_point[1],  # Groove center Y
                'grinder_tip_x_px': grinder_ref[0] if grinder_ref else 0,
                'grinder_tip_y_px': grinder_ref[1] if grinder_ref else 0,
                'move_x_pixels': int(move_x_px),
                'move_y_pixels': int(move_y_px),
                'move_x_mm': round(move_x_mm, 2),
                'move_y_mm': round(move_y_mm, 2),
                'distance_to_grinder_px': round(distance_px, 1),
                'distance_to_grinder_mm': round(distance_mm, 2),
                'groove_angle_degrees': round(tooth.angle, 2),
                'groove_depth_px': round(tooth.depth, 1),
                'top_peak': tooth.top_valley,  # Top tooth peak
                'bottom_peak': tooth.bottom_valley  # Bottom tooth peak
            })

        return grinding_coords

    def analyze_blade(self, pixels_per_mm=1.0):
        """
        Run complete analysis pipeline
        Detects grooves (grinding targets) and grinder edge

        Args:
            pixels_per_mm: Calibration factor for converting pixels to mm
        """
        print("Starting blade analysis...")

        # Step 1: Preprocess
        print("1. Preprocessing image...")
        self.preprocess_image()

        # Step 2: Detect edges
        print("2. Detecting edges...")
        self.detect_edges()

        # Step 3: Find contours
        print("3. Finding blade contours...")
        self.find_blade_contours()

        # Step 4: Detect blade edge and grinder tip
        print("4. Detecting blade edge and grinder tip...")
        blade_points, grinder_tip = self.detect_blade_and_grinder()

        if blade_points is not None:
            print(f"   Blade edge detected with {len(blade_points)} points")
        if grinder_tip is not None:
            print(f"   Grinder tip (pointing at blade) detected at {grinder_tip}")

        # Step 5: Extract groove profiles from blade edge
        print("5. Extracting groove profiles (grinding targets)...")
        self.teeth_profiles = self.extract_tooth_profiles()
        print(f"   Found {len(self.teeth_profiles)} grooves to grind")

        # Step 6: Calculate blade orientation
        print("6. Calculating blade orientation...")
        blade_angle = self.calculate_blade_orientation()
        print(f"   Blade orientation: {blade_angle:.2f} degrees from vertical")

        # Step 7: Generate grinding coordinates
        print("7. Generating grinding coordinates...")
        grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)

        print(f"\nAnalysis complete! Found {len(self.teeth_profiles)} grooves to grind")

        return {
            'blade_angle': blade_angle,
            'num_grooves': len(self.teeth_profiles),
            'grinder_tip': self.grinder_tip if hasattr(self, 'grinder_tip') else None,
            'grinder_center': self.grinder_center if hasattr(self, 'grinder_center') else None,
            'grinding_coordinates': grinding_coords,
            'tooth_profiles': self.teeth_profiles
        }

    def visualize_results(self, save_path=None):
        """
        Visualize the detection results
        Shows grooves (grinding targets) and grinder edge

        Args:
            save_path: Path to save the visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image\n(Blade Left, Grinder Right)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Binary image
        axes[0, 1].imshow(self.binary, cmap='gray')
        axes[0, 1].set_title('Binary Image', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Edge detection
        axes[0, 2].imshow(self.edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Detected blade edge and grinder edge
        viz_image1 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        if self.blade_edge_points is not None:
            for point in self.blade_edge_points[::3]:  # Sample every 3rd point
                cv2.circle(viz_image1, (point[0], point[1]), 2, (255, 0, 0), -1)

        if self.grinder_edge_points is not None:
            for point in self.grinder_edge_points[::3]:
                cv2.circle(viz_image1, (point[0], point[1]), 2, (0, 255, 0), -1)

        # Mark grinder TIP (sharp point facing blade)
        if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
            cv2.circle(viz_image1, self.grinder_tip, 10, (255, 255, 0), -1)
            cv2.circle(viz_image1, self.grinder_tip, 12, (0, 0, 0), 2)
            cv2.putText(viz_image1, "GRINDER TIP",
                        (self.grinder_tip[0] + 15, self.grinder_tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(viz_image1, "(pointing at blade)",
                        (self.grinder_tip[0] + 15, self.grinder_tip[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        axes[1, 0].imshow(viz_image1)
        axes[1, 0].set_title('Detected Edges\n(Red: Blade, Green: Grinder, Yellow: Grinder Tip)', fontsize=11)
        axes[1, 0].axis('off')

        # Tooth profiles with grooves highlighted
        viz_image2 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        for i, tooth in enumerate(self.teeth_profiles):
            # Draw GROOVE (the grinding target) - LARGE and BRIGHT
            cv2.circle(viz_image2, tooth.grinding_point, 8, (255, 0, 255), -1)  # Magenta for groove
            cv2.circle(viz_image2, tooth.grinding_point, 10, (255, 255, 255), 2)  # White outline

            # Draw tooth peaks (smaller)
            cv2.circle(viz_image2, tooth.top_valley, 5, (0, 255, 0), -1)
            cv2.circle(viz_image2, tooth.bottom_valley, 5, (0, 255, 0), -1)

            # Draw tooth outline
            cv2.line(viz_image2, tooth.top_valley, tooth.grinding_point, (255, 255, 0), 2)
            cv2.line(viz_image2, tooth.grinding_point, tooth.bottom_valley, (255, 255, 0), 2)

            # Add labels
            label_pos = (tooth.grinding_point[0] - 70, tooth.grinding_point[1])
            cv2.putText(viz_image2, f"GROOVE #{tooth.tooth_id}",
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(viz_image2, f"{tooth.angle:.1f}°",
                        (label_pos[0], label_pos[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Mark grinder tip
        if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
            cv2.circle(viz_image2, self.grinder_tip, 12, (255, 128, 0), 3)
            cv2.putText(viz_image2, "GRINDER TIP",
                        (self.grinder_tip[0] + 15, self.grinder_tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

        axes[1, 1].imshow(viz_image2)
        axes[1, 1].set_title('Tooth Grooves & Angles\n(Magenta: Grooves to Grind, Green: Peaks, Orange: Grinder Tip)',
                             fontsize=11)
        axes[1, 1].axis('off')

        # Movement vectors from grooves to grinder tip
        viz_image3 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)

        grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None

        for i, tooth in enumerate(self.teeth_profiles):
            # Draw groove point (grinding target)
            cv2.circle(viz_image3, tooth.grinding_point, 8, (255, 0, 255), -1)
            cv2.circle(viz_image3, tooth.grinding_point, 10, (255, 255, 255), 2)

            # Draw movement arrow from GROOVE to GRINDER EDGE
            if grinder_ref is not None:
                cv2.arrowedLine(viz_image3, tooth.grinding_point, grinder_ref,
                                (0, 255, 255), 3, tipLength=0.03)

                # Add movement details
                mid_x = (tooth.grinding_point[0] + grinder_ref[0]) // 2
                mid_y = (tooth.grinding_point[1] + grinder_ref[1]) // 2
                distance_px = np.sqrt(tooth.move_to_grinder[0] ** 2 + tooth.move_to_grinder[1] ** 2)

                # Background box for text
                cv2.rectangle(viz_image3, (mid_x - 55, mid_y - 35), (mid_x + 55, mid_y + 35),
                              (0, 0, 0), -1)
                cv2.rectangle(viz_image3, (mid_x - 55, mid_y - 35), (mid_x + 55, mid_y + 35),
                              (255, 255, 255), 2)

                cv2.putText(viz_image3, f"Groove #{tooth.tooth_id}",
                            (mid_x - 50, mid_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
                cv2.putText(viz_image3, f"{distance_px:.0f}px",
                            (mid_x - 30, mid_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(viz_image3, f"dx:{tooth.move_to_grinder[0]:.0f}",
                            (mid_x - 50, mid_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(viz_image3, f"dy:{tooth.move_to_grinder[1]:.0f}",
                            (mid_x - 50, mid_y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Mark grinder tip with crosshair
        if grinder_ref is not None:
            cv2.circle(viz_image3, grinder_ref, 15, (255, 128, 0), 4)
            cv2.drawMarker(viz_image3, grinder_ref, (255, 255, 0),
                           cv2.MARKER_CROSS, 30, 4)
            cv2.putText(viz_image3, "GRINDER TIP",
                        (grinder_ref[0] + 20, grinder_ref[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

        axes[1, 2].imshow(viz_image3)
        axes[1, 2].set_title('Grinding Path: Groove → Grinder Tip\n(Cyan Arrows: Move Blade to Align Grooves)',
                             fontsize=11)
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def export_coordinates_to_csv(self, filename='grinding_coordinates.csv', pixels_per_mm=1.0):
        """Export grinding coordinates to CSV file"""
        import csv

        grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['tooth_id',
                          'groove_x_px', 'groove_y_px',
                          'grinder_tip_x_px', 'grinder_tip_y_px',
                          'move_x_px', 'move_y_px',
                          'move_x_mm', 'move_y_mm',
                          'distance_to_grinder_px', 'distance_to_grinder_mm',
                          'groove_angle_deg', 'groove_depth_px',
                          'top_peak_x', 'top_peak_y',
                          'bottom_peak_x', 'bottom_peak_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for coord in grinding_coords:
                writer.writerow({
                    'tooth_id': coord['tooth_id'],
                    'groove_x_px': coord['groove_position_x_px'],
                    'groove_y_px': coord['groove_position_y_px'],
                    'grinder_tip_x_px': coord['grinder_tip_x_px'],
                    'grinder_tip_y_px': coord['grinder_tip_y_px'],
                    'move_x_px': coord['move_x_pixels'],
                    'move_y_px': coord['move_y_pixels'],
                    'move_x_mm': coord['move_x_mm'],
                    'move_y_mm': coord['move_y_mm'],
                    'distance_to_grinder_px': coord['distance_to_grinder_px'],
                    'distance_to_grinder_mm': coord['distance_to_grinder_mm'],
                    'groove_angle_deg': coord['groove_angle_degrees'],
                    'groove_depth_px': coord['groove_depth_px'],
                    'top_peak_x': coord['top_peak'][0],
                    'top_peak_y': coord['top_peak'][1],
                    'bottom_peak_x': coord['bottom_peak'][0],
                    'bottom_peak_y': coord['bottom_peak'][1]
                })

        print(f"Coordinates exported to {filename}")


def main():
    """Main execution function"""
    # Load and analyze the blade image
    image_path = 'untitled3.png'

    # Calibration: pixels per millimeter (you need to calibrate this based on your camera setup)
    PIXELS_PER_MM = 86.96  # Default value - adjust based on your calibration!

    print("=" * 80)
    print(" " * 20 + "SERRATED BLADE SHARPENING ANALYZER")
    print("=" * 80)
    print(f"\nCalibration: {PIXELS_PER_MM} pixels per millimeter")
    print("(Adjust PIXELS_PER_MM in the code based on your camera calibration)\n")

    # Create analyzer instance
    analyzer = SerratedBladeAnalyzer(image_path)

    # Run analysis
    results = analyzer.analyze_blade(pixels_per_mm=PIXELS_PER_MM)

    # Print results
    print("\n" + "=" * 80)
    print(" " * 30 + "ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nBlade Configuration:")
    print(f"  • Blade orientation: {results['blade_angle']:.2f}° from vertical")
    print(f"  • Number of grooves detected: {results['num_grooves']}")

    if results['grinder_tip']:
        print(f"\nGrinder Tip Position:")
        print(f"  • Grinder tip location: ({results['grinder_tip'][0]}, {results['grinder_tip'][1]}) pixels")

    print("\n" + "-" * 80)
    print("ROBOT MOVEMENT COORDINATES FOR EACH GROOVE:")
    print("-" * 80)
    print(
        f"{'Groove':>6} | {'Current Pos (px)':>18} | {'Move X':>10} | {'Move Y':>10} | {'Distance':>12} | {'Angle':>8}")
    print(f"{'ID':>6} | {'(X, Y)':>18} | {'(px/mm)':>10} | {'(px/mm)':>10} | {'(px/mm)':>12} | {'(deg)':>8}")
    print("-" * 80)

    for coord in results['grinding_coordinates']:
        print(f"{coord['tooth_id']:>6} | "
              f"({coord['groove_position_x_px']:>4}, {coord['groove_position_y_px']:>4}){' ':>5} | "
              f"{coord['move_x_pixels']:>4}/{coord['move_x_mm']:>4.1f} | "
              f"{coord['move_y_pixels']:>4}/{coord['move_y_mm']:>4.1f} | "
              f"{coord['distance_to_grinder_px']:>5.0f}/{coord['distance_to_grinder_mm']:>4.1f} | "
              f"{coord['groove_angle_degrees']:>7.1f}°")

    print("\n" + "=" * 80)
    print("INSTRUCTIONS FOR ROBOT CONTROL:")
    print("=" * 80)
    print("""
For each groove, move the blade by the specified (move_x, move_y) coordinates
to align the groove with the grinder edge. Then activate grinding.

Example pseudocode:
    for each groove in grinding_coordinates:
        robot.move_blade_relative(
            x = groove.move_x_mm,
            y = groove.move_y_mm
        )
        robot.activate_grinder()
        robot.wait(grinding_time_seconds)
        robot.deactivate_grinder()
        robot.move_to_home_position()
""")

    # Export to CSV
    csv_path = 'grinding_coordinates.csv'
    analyzer.export_coordinates_to_csv(csv_path, pixels_per_mm=PIXELS_PER_MM)

    # Visualize results
    print("\nGenerating visualization...")
    viz_path = 'blade_analysis_results.png'
    analyzer.visualize_results(save_path=viz_path)

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  • Coordinates CSV: {csv_path}")
    print(f"  • Visualization: {viz_path}")
    print("\n")


if __name__ == "__main__":
    main()