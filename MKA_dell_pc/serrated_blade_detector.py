# """
# Optimized Serrated Blade Analyzer - TOOTH POINT DETECTION
# Detects sharp tooth points (peaks) instead of grooves (valleys)
# """
#
# import cv2
# import numpy as np
# from scipy import ndimage
# from dataclasses import dataclass
# from typing import List, Tuple, Union
#
#
# @dataclass
# class ToothProfile:
#     """Data class to store tooth point information"""
#     tooth_id: int
#     apex_point: Tuple[int, int]  # Tooth tip (peak - the grinding target)
#     top_valley: Tuple[int, int]
#     bottom_valley: Tuple[int, int]
#     angle: float
#     grinding_point: Tuple[int, int]
#     height: float  # Height of the tooth from valleys
#     move_to_grinder: Tuple[float, float]
#
#
# class SerratedBladeAnalyzer:
#     """Optimized analyzer for serrated blade tooth point detection"""
#
#     def __init__(self, image_input: Union[str, np.ndarray]):
#         """
#         Initialize analyzer with image path or numpy array
#
#         Args:
#             image_input: Path to image file OR numpy array (BGR format)
#         """
#         if isinstance(image_input, str):
#             self.image = cv2.imread(image_input)
#             if self.image is None:
#                 raise ValueError(f"Could not load image from {image_input}")
#         elif isinstance(image_input, np.ndarray):
#             self.image = image_input
#         else:
#             raise ValueError("image_input must be string path or numpy array")
#
#         self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#         self.height, self.width = self.gray.shape
#         self.teeth_profiles = []
#         self.blade_edge_points = None
#         self.grinder_tip = None
#         self.grinder_center = None
#
#     def preprocess_image(self, blur_kernel=31):
#         """Preprocess image for edge detection"""
#         self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
#         self.binary = cv2.adaptiveThreshold(
#             self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
#         return self.binary
#
#     def detect_edges(self, canny_low=100, canny_high=150):
#         """Detect edges using Canny"""
#         self.edges = cv2.Canny(self.blurred, canny_low, canny_high)
#         return self.edges
#
#     def find_blade_contours(self):
#         """Find blade contours"""
#         contours, _ = cv2.findContours(
#             self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         self.blade_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
#         return self.blade_contours
#
#     def detect_blade_and_grinder(self, sampling_step=3):
#         """Detect blade edge and grinder tip"""
#         blade_edge = []
#         grinder_points = []
#
#         for y in range(0, self.height, sampling_step):
#             row = self.binary[y, :]
#             white_pixels = np.where(row > 90)[0]
#
#             if len(white_pixels) > 0:
#                 blade_edge.append((white_pixels[0], y))
#
#                 if len(white_pixels) > 10:
#                     rightmost_region = white_pixels[white_pixels > self.width // 3 * 2]
#                     if len(rightmost_region) > 0:
#                         grinder_points.append((rightmost_region[0], y))
#
#         self.blade_edge_points = np.array(blade_edge) if blade_edge else None
#         self.grinder_edge_points = np.array(grinder_points) if grinder_points else None
#
#         # Find grinder tip (leftmost point)
#         if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
#             min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
#             self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
#
#             # Average nearby points for accuracy
#             min_x = self.grinder_edge_points[min_x_idx, 0]
#             tip_points = self.grinder_edge_points[
#                 np.abs(self.grinder_edge_points[:, 0] - min_x) < 15
#                 ]
#
#             self.grinder_edge_center = (
#                 int(np.mean(tip_points[:, 0])),
#                 int(np.mean(tip_points[:, 1]))
#             )
#
#         return self.blade_edge_points, self.grinder_tip
#
#     def extract_tooth_profiles(self, window_size=10, min_height_px=100):
#         """
#         Extract tooth POINTS (peaks) from blade edge
#         NOW DETECTS SHARP TOOTH TIPS INSTEAD OF GROOVES
#
#         Args:
#             window_size: Window size for local maxima/minima detection
#             min_height_px: Minimum tooth height in pixels to filter out noise
#         """
#         if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
#             return []
#
#         x_coords = self.blade_edge_points[:, 0]
#         y_coords = self.blade_edge_points[:, 1]
#
#         # Smooth x-coordinates
#         x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)
#
#         # Find peaks (TOOTH TIPS - what we want!) and valleys (grooves between teeth)
#         peaks = []  # TOOTH TIPS - the grinding targets!
#         valleys = []  # Grooves between teeth
#
#         mean_x = np.mean(x_smooth)
#
#         for i in range(window_size, len(x_smooth) - window_size):
#             window = x_smooth[i - window_size:i + window_size]
#
#             # PEAKS are tooth tips (pointing right = higher x values) - GRINDING TARGETS
#             if x_smooth[i] == np.max(window) and x_smooth[i] > mean_x + 20:
#                 peaks.append(i)
#             # Valleys are grooves between teeth (lower x values)
#             elif x_smooth[i] == np.min(window) and x_smooth[i] < mean_x - 20:
#                 valleys.append(i)
#
#         # Filter close points
#         peaks = self._filter_close_points(peaks, window_size)
#         valleys = self._filter_close_points(valleys, window_size)
#
#         # Create tooth profiles - each PEAK becomes a grinding target
#         tooth_profiles = []
#         tooth_id = 1
#
#         for peak_idx in peaks:
#             # Find valleys above and below this peak
#             valleys_above = [v for v in valleys if v < peak_idx]
#             valleys_below = [v for v in valleys if v > peak_idx]
#
#             # Handle edge cases
#             if len(valleys_above) == 0 and len(valleys_below) > 0:
#                 # Top edge tooth - sample points above the peak
#                 sample_size = min(window_size // 2, 50)
#                 sample_end = min(sample_size, peak_idx)
#                 sample_indices = range(0, sample_end)
#
#                 sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
#                 sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]
#
#                 top_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (int(x_smooth[0]),
#                                                                                                    int(y_coords[0]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
#
#             elif len(valleys_below) == 0 and len(valleys_above) > 0:
#                 # Bottom edge tooth - sample points below the peak
#                 sample_start = peak_idx
#                 sample_end = min(len(x_smooth) - 1, peak_idx + window_size * 2)
#                 sample_indices = range(sample_start, sample_end)
#
#                 sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
#                 sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]
#
#                 top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (
#                     int(x_smooth[sample_end]), int(y_coords[sample_end]))
#
#             elif len(valleys_above) > 0 and len(valleys_below) > 0:
#                 # Middle tooth - normal case
#                 top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
#             else:
#                 continue
#
#             # Average nearby peak points for accuracy
#             peak_sample_size = max(3, window_size // 10)
#             peak_start = max(0, peak_idx - peak_sample_size)
#             peak_end = min(len(x_smooth), peak_idx + peak_sample_size + 1)
#
#             peak_x_samples = x_smooth[peak_start:peak_end]
#             peak_y_samples = y_coords[peak_start:peak_end]
#
#             tooth_point = (int(np.mean(peak_x_samples)), int(np.mean(peak_y_samples))) if len(peak_x_samples) > 0 else (
#                 int(x_smooth[peak_idx]), int(y_coords[peak_idx]))
#
#             # Calculate tooth height (distance from valleys to peak)
#             height = abs(tooth_point[0] - ((top_valley[0] + bottom_valley[0]) / 2))
#
#             # Calculate angle
#             angle = self._calculate_tooth_angle(top_valley, tooth_point, bottom_valley)
#
#             # Calculate movement to grinder
#             if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
#                 move_to_grinder = (
#                     self.grinder_tip[0] - tooth_point[0],
#                     self.grinder_tip[1] - tooth_point[1]
#                 )
#             else:
#                 move_to_grinder = (0, 0)
#
#             tooth_profiles.append(ToothProfile(
#                 tooth_id=tooth_id,
#                 apex_point=tooth_point,
#                 top_valley=top_valley,
#                 bottom_valley=bottom_valley,
#                 angle=angle,
#                 grinding_point=tooth_point,
#                 height=height,
#                 move_to_grinder=move_to_grinder
#             ))
#
#             tooth_id += 1
#
#         return tooth_profiles
#
#     def _filter_close_points(self, points, min_distance):
#         """Filter out points too close together"""
#         if len(points) == 0:
#             return points
#
#         filtered = [points[0]]
#         for p in points[1:]:
#             if abs(p - filtered[-1]) >= min_distance:
#                 filtered.append(p)
#         return filtered
#
#     def _calculate_tooth_angle(self, top_valley, apex, bottom_valley):
#         """Calculate tooth angle in degrees"""
#         v1 = np.array([top_valley[0] - apex[0], top_valley[1] - apex[1]], dtype=float)
#         v2 = np.array([bottom_valley[0] - apex[0], bottom_valley[1] - apex[1]], dtype=float)
#
#         dot_product = np.dot(v1, v2)
#         magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
#
#         if magnitude == 0:
#             return 0
#
#         angle = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
#         return float(np.degrees(angle))
#
#     def calculate_blade_orientation(self):
#         """Calculate blade edge orientation"""
#         if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
#             return 0
#
#         fit_result = cv2.fitLine(
#             self.blade_edge_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
#         )
#
#         vx, vy = fit_result[0][0], fit_result[1][0]
#         angle = np.degrees(np.arctan2(vy, vx))
#         return float(angle)
#
#     def generate_grinding_coordinates(self, pixels_per_mm=1.0):
#         """Generate grinding coordinates for robot control - TOOTH POINTS"""
#         grinding_coords = []
#         grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None
#
#         for tooth in self.teeth_profiles:
#             move_x_px = tooth.move_to_grinder[0]
#             move_y_px = tooth.move_to_grinder[1]
#
#             move_x_mm = move_x_px / pixels_per_mm
#             move_y_mm = move_y_px / pixels_per_mm
#
#             distance_px = np.sqrt(move_x_px ** 2 + move_y_px ** 2)
#             distance_mm = distance_px / pixels_per_mm
#
#             grinding_coords.append({
#                 'tooth_id': tooth.tooth_id,
#                 'tooth_tip_x_px': tooth.grinding_point[0],  # Tooth tip X
#                 'tooth_tip_y_px': tooth.grinding_point[1],  # Tooth tip Y
#                 'groove_position_x_px': tooth.grinding_point[0],  # Keep for compatibility
#                 'groove_position_y_px': tooth.grinding_point[1],  # Keep for compatibility
#                 'grinder_tip_x_px': grinder_ref[0] if grinder_ref else 0,
#                 'grinder_tip_y_px': grinder_ref[1] if grinder_ref else 0,
#                 'move_x_pixels': int(move_x_px),
#                 'move_y_pixels': int(move_y_px),
#                 'move_x_mm': round(move_x_mm, 2),
#                 'move_y_mm': round(move_y_mm, 2),
#                 'distance_to_grinder_px': round(distance_px, 1),
#                 'distance_to_grinder_mm': round(distance_mm, 2),
#                 'tooth_angle_degrees': round(tooth.angle, 2),
#                 'tooth_height_px': round(tooth.height, 1),
#                 'groove_angle_degrees': round(tooth.angle, 2),  # Keep for compatibility
#                 'groove_depth_px': round(tooth.height, 1),  # Keep for compatibility
#                 'top_peak': tooth.top_valley,
#                 'bottom_peak': tooth.bottom_valley
#             })
#
#         return grinding_coords
#
#     def analyze_blade(self, pixels_per_mm=1.0):
#         """Run complete analysis pipeline"""
#         self.preprocess_image()
#         self.detect_edges()
#         self.find_blade_contours()
#         self.detect_blade_and_grinder()
#         self.teeth_profiles = self.extract_tooth_profiles()
#         blade_angle = self.calculate_blade_orientation()
#         grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)
#
#         return {
#             'blade_angle': blade_angle,
#             'num_grooves': len(self.teeth_profiles),  # Keep name for compatibility
#             'num_teeth': len(self.teeth_profiles),
#             'grinder_tip': self.grinder_tip if hasattr(self, 'grinder_tip') else None,
#             'grinder_center': self.grinder_center if hasattr(self, 'grinder_center') else None,
#             'grinding_coordinates': grinding_coords,
#             'tooth_profiles': self.teeth_profiles
#         }
#
#     def visualize_results(self, save_path=None):
#         """Visualize detection results - TOOTH POINTS"""
#         import matplotlib.pyplot as plt
#
#         fig, axes = plt.subplots(2, 3, figsize=(20, 13))
#
#         # Original image
#         axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
#         axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
#         axes[0, 0].axis('off')
#
#         # Binary image
#         axes[0, 1].imshow(self.binary, cmap='gray')
#         axes[0, 1].set_title('Binary Image', fontsize=12, fontweight='bold')
#         axes[0, 1].axis('off')
#
#         # Edge detection
#         axes[0, 2].imshow(self.edges, cmap='gray')
#         axes[0, 2].set_title('Edge Detection', fontsize=12, fontweight='bold')
#         axes[0, 2].axis('off')
#
#         # Detected edges
#         viz_image1 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
#         if self.blade_edge_points is not None:
#             for point in self.blade_edge_points[::3]:
#                 cv2.circle(viz_image1, tuple(point), 2, (255, 0, 0), -1)
#
#         if self.grinder_edge_points is not None:
#             for point in self.grinder_edge_points[::3]:
#                 cv2.circle(viz_image1, tuple(point), 2, (0, 255, 0), -1)
#
#         if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
#             cv2.circle(viz_image1, self.grinder_tip, 10, (255, 255, 0), -1)
#             cv2.circle(viz_image1, self.grinder_tip, 12, (0, 0, 0), 2)
#             cv2.putText(viz_image1, "GRINDER TIP", (self.grinder_tip[0] + 15, self.grinder_tip[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#
#         axes[1, 0].imshow(viz_image1)
#         axes[1, 0].set_title('Detected Edges', fontsize=11)
#         axes[1, 0].axis('off')
#
#         # Tooth profiles - PEAKS highlighted
#         viz_image2 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
#         for tooth in self.teeth_profiles:
#             # Draw TOOTH TIP (peak) - LARGE and BRIGHT
#             cv2.circle(viz_image2, tooth.grinding_point, 8, (0, 255, 255), -1)  # Cyan for tooth tip
#             cv2.circle(viz_image2, tooth.grinding_point, 10, (255, 255, 255), 2)  # White outline
#
#             # Draw valleys (grooves) - smaller
#             cv2.circle(viz_image2, tooth.top_valley, 5, (255, 0, 0), -1)  # Red for valleys
#             cv2.circle(viz_image2, tooth.bottom_valley, 5, (255, 0, 0), -1)
#
#             # Draw tooth outline
#             cv2.line(viz_image2, tooth.top_valley, tooth.grinding_point, (255, 255, 0), 2)
#             cv2.line(viz_image2, tooth.grinding_point, tooth.bottom_valley, (255, 255, 0), 2)
#
#             label_pos = (tooth.grinding_point[0] + 15, tooth.grinding_point[1])
#             cv2.putText(viz_image2, f"TOOTH #{tooth.tooth_id}", label_pos,
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#         if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
#             cv2.circle(viz_image2, self.grinder_tip, 12, (255, 128, 0), 3)
#
#         axes[1, 1].imshow(viz_image2)
#         axes[1, 1].set_title('Tooth Points (Cyan: Tips, Red: Valleys)', fontsize=11)
#         axes[1, 1].axis('off')
#
#         # Movement vectors
#         viz_image3 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
#         grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None
#
#         for tooth in self.teeth_profiles:
#             cv2.circle(viz_image3, tooth.grinding_point, 8, (0, 255, 255), -1)
#             cv2.circle(viz_image3, tooth.grinding_point, 10, (255, 255, 255), 2)
#
#             if grinder_ref is not None:
#                 cv2.arrowedLine(viz_image3, tooth.grinding_point, grinder_ref,
#                                 (0, 255, 255), 3, tipLength=0.03)
#
#         if grinder_ref is not None:
#             cv2.circle(viz_image3, grinder_ref, 15, (255, 128, 0), 4)
#             cv2.drawMarker(viz_image3, grinder_ref, (255, 255, 0),
#                            cv2.MARKER_CROSS, 30, 4)
#
#         axes[1, 2].imshow(viz_image3)
#         axes[1, 2].set_title('Grinding Path: Tooth Tips → Grinder', fontsize=11)
#         axes[1, 2].axis('off')
#
#         plt.tight_layout()
#
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#
#         plt.show()
#
#     def export_coordinates_to_csv(self, filename='grinding_coordinates.csv', pixels_per_mm=1.0):
#         """Export grinding coordinates to CSV"""
#         import csv
#
#         grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)
#
#         with open(filename, 'w', newline='') as csvfile:
#             fieldnames = ['tooth_id', 'tooth_tip_x_px', 'tooth_tip_y_px',
#                           'grinder_tip_x_px', 'grinder_tip_y_px',
#                           'move_x_px', 'move_y_px', 'move_x_mm', 'move_y_mm',
#                           'distance_to_grinder_px', 'distance_to_grinder_mm',
#                           'tooth_angle_deg', 'tooth_height_px',
#                           'top_valley_x', 'top_valley_y', 'bottom_valley_x', 'bottom_valley_y']
#
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#
#             for coord in grinding_coords:
#                 writer.writerow({
#                     'tooth_id': coord['tooth_id'],
#                     'tooth_tip_x_px': coord['tooth_tip_x_px'],
#                     'tooth_tip_y_px': coord['tooth_tip_y_px'],
#                     'grinder_tip_x_px': coord['grinder_tip_x_px'],
#                     'grinder_tip_y_px': coord['grinder_tip_y_px'],
#                     'move_x_px': coord['move_x_pixels'],
#                     'move_y_px': coord['move_y_pixels'],
#                     'move_x_mm': coord['move_x_mm'],
#                     'move_y_mm': coord['move_y_mm'],
#                     'distance_to_grinder_px': coord['distance_to_grinder_px'],
#                     'distance_to_grinder_mm': coord['distance_to_grinder_mm'],
#                     'tooth_angle_deg': coord['tooth_angle_degrees'],
#                     'tooth_height_px': coord['tooth_height_px'],
#                     'top_valley_x': coord['top_peak'][0],
#                     'top_valley_y': coord['top_peak'][1],
#                     'bottom_valley_x': coord['bottom_peak'][0],
#                     'bottom_valley_y': coord['bottom_peak'][1]
#                 })
#
#
# def main():
#     """Main execution"""
#     image_path = 'results/untitled3.png'
#     PIXELS_PER_MM = 86.96
#
#     print("=" * 80)
#     print("SERRATED BLADE ANALYZER - TOOTH POINT DETECTION")
#     print("=" * 80)
#
#     analyzer = SerratedBladeAnalyzer(image_path)
#     results = analyzer.analyze_blade(pixels_per_mm=PIXELS_PER_MM)
#
#     print(f"\nBlade orientation: {results['blade_angle']:.2f}°")
#     print(f"Teeth detected: {results['num_teeth']}")
#
#     if results['grinder_tip']:
#         print(f"Grinder tip: {results['grinder_tip']}")
#
#     print(f"\n{'Tooth':>6} | {'Tip Position':>14} | {'Move (mm)':>14} | {'Distance':>10}")
#     print("-" * 55)
#
#     for coord in results['grinding_coordinates']:
#         print(f"{coord['tooth_id']:>6} | "
#               f"({coord['tooth_tip_x_px']:>4},{coord['tooth_tip_y_px']:>4}) | "
#               f"({coord['move_x_mm']:>5.1f},{coord['move_y_mm']:>5.1f}) | "
#               f"{coord['distance_to_grinder_mm']:>6.1f}mm")
#
#     analyzer.export_coordinates_to_csv('grinding_coordinates.csv', pixels_per_mm=PIXELS_PER_MM)
#     analyzer.visualize_results(save_path='blade_analysis_results.png')
#
#     print("\n✓ Analysis complete!")
#
#
# if __name__ == "__main__":
#     main()

"""
Optimized Serrated Blade Analyzer - TOOTH POINT DETECTION
Detects sharp tooth points (peaks) instead of grooves (valleys)
"""

import cv2
import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass
class ToothProfile:
    """Data class to store tooth point information"""
    tooth_id: int
    apex_point: Tuple[int, int]  # Tooth tip (peak - the grinding target)
    top_valley: Tuple[int, int]
    bottom_valley: Tuple[int, int]
    angle: float
    grinding_point: Tuple[int, int]
    height: float  # Height of the tooth from valleys
    move_to_grinder: Tuple[float, float]


class SerratedBladeAnalyzer:
    """Optimized analyzer for serrated blade tooth point detection"""

    def __init__(self, image_input: Union[str, np.ndarray]):
        """
        Initialize analyzer with image path or numpy array

        Args:
            image_input: Path to image file OR numpy array (BGR format)
        """
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input)
            if self.image is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            self.image = image_input
        else:
            raise ValueError("image_input must be string path or numpy array")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self.teeth_profiles = []
        self.blade_edge_points = None
        self.grinder_tip = None
        self.grinder_center = None

    def preprocess_image(self, blur_kernel=31):
        """Preprocess image for edge detection"""
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
        self.binary = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return self.binary

    def detect_edges(self, canny_low=100, canny_high=150):
        """Detect edges using Canny"""
        self.edges = cv2.Canny(self.blurred, canny_low, canny_high)
        return self.edges

    def find_blade_contours(self):
        """Find blade contours"""
        contours, _ = cv2.findContours(
            self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.blade_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        return self.blade_contours

    def detect_blade_and_grinder(self, sampling_step=1):
        """
        Detect blade edge and grinder tip

        Args:
            sampling_step: Step size for scanning in pixels
                          Use 1 for most precise edge detection
                          Use 3-5 for faster but less precise detection
        """
        blade_edge = []
        grinder_points = []

        for y in range(0, self.height, sampling_step):
            row = self.binary[y, :]
            white_pixels = np.where(row > 170)[0]

            if len(white_pixels) > 0:
                blade_edge.append((white_pixels[0], y))

                if len(white_pixels) > 20:
                    rightmost_region = white_pixels[white_pixels > self.width // 3 * 2]
                    if len(rightmost_region) > 0:
                        grinder_points.append((rightmost_region[0], y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        # Find grinder tip (leftmost point)
        if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
            min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
            self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])

            # Average nearby points for accuracy
            min_x = self.grinder_edge_points[min_x_idx, 0]
            tip_points = self.grinder_edge_points[
                np.abs(self.grinder_edge_points[:, 0] - min_x) < 15
                ]

            self.grinder_edge_center = (
                int(np.mean(tip_points[:, 0])),
                int(np.mean(tip_points[:, 1]))
            )

        return self.blade_edge_points, self.grinder_tip

    def extract_tooth_profiles(self, window_size=30, min_height_px=100):
        """
        Extract tooth POINTS (peaks) from blade edge
        NOW DETECTS SHARP TOOTH TIPS INSTEAD OF GROOVES

        Args:
            window_size: Window size for local maxima/minima detection
            min_height_px: Minimum tooth height in pixels to filter out noise
        """
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return []

        x_coords = self.blade_edge_points[:, 0]
        y_coords = self.blade_edge_points[:, 1]

        # Smooth x-coordinates
        x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

        # Find peaks (TOOTH TIPS - what we want!) and valleys (grooves between teeth)
        peaks = []  # TOOTH TIPS - the grinding targets!
        valleys = []  # Grooves between teeth

        mean_x = np.mean(x_smooth)

        for i in range(window_size, len(x_smooth) - window_size):
            window = x_smooth[i - window_size:i + window_size]

            # PEAKS are tooth tips (pointing right = higher x values) - GRINDING TARGETS
            if x_smooth[i] == np.max(window) and x_smooth[i] > mean_x + 20:
                peaks.append(i)
            # Valleys are grooves between teeth (lower x values)
            elif x_smooth[i] == np.min(window) and x_smooth[i] < mean_x - 20:
                valleys.append(i)

        # Filter close points
        peaks = self._filter_close_points(peaks, window_size)
        valleys = self._filter_close_points(valleys, window_size)

        # Create tooth profiles - each PEAK becomes a grinding target
        tooth_profiles = []
        tooth_id = 1

        for peak_idx in peaks:
            # Find valleys above and below this peak
            valleys_above = [v for v in valleys if v < peak_idx]
            valleys_below = [v for v in valleys if v > peak_idx]

            # Handle edge cases
            if len(valleys_above) == 0 and len(valleys_below) > 0:
                # Top edge tooth - sample points above the peak
                sample_size = min(window_size // 2, 50)
                sample_end = min(sample_size, peak_idx)
                sample_indices = range(0, sample_end)

                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                top_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (int(x_smooth[0]),
                                                                                                   int(y_coords[0]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))

            elif len(valleys_below) == 0 and len(valleys_above) > 0:
                # Bottom edge tooth - sample points below the peak
                sample_start = peak_idx
                sample_end = min(len(x_smooth) - 1, peak_idx + window_size * 2)
                sample_indices = range(sample_start, sample_end)

                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (
                    int(x_smooth[sample_end]), int(y_coords[sample_end]))

            elif len(valleys_above) > 0 and len(valleys_below) > 0:
                # Middle tooth - normal case
                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
            else:
                continue

            # Use exact peak point for precise edge detection
            # OPTION 1: No averaging - most precise (use this for exact edge)
            tooth_point = (int(x_smooth[peak_idx]), int(y_coords[peak_idx]))

            # OPTION 2: Minimal averaging (uncomment if you want slight smoothing)
            # peak_sample_size = 2  # Only average 2-3 points
            # peak_start = max(0, peak_idx - peak_sample_size)
            # peak_end = min(len(x_smooth), peak_idx + peak_sample_size + 1)
            # peak_x_samples = x_smooth[peak_start:peak_end]
            # peak_y_samples = y_coords[peak_start:peak_end]
            # tooth_point = (int(np.mean(peak_x_samples)), int(np.mean(peak_y_samples))) if len(peak_x_samples) > 0 else (int(x_smooth[peak_idx]), int(y_coords[peak_idx]))

            # Calculate tooth height (distance from valleys to peak)
            height = abs(tooth_point[0] - ((top_valley[0] + bottom_valley[0]) / 2))

            # Calculate angle
            angle = self._calculate_tooth_angle(top_valley, tooth_point, bottom_valley)

            # Calculate movement to grinder
            if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
                move_to_grinder = (
                    self.grinder_tip[0] - tooth_point[0],
                    self.grinder_tip[1] - tooth_point[1]
                )
            else:
                move_to_grinder = (0, 0)

            tooth_profiles.append(ToothProfile(
                tooth_id=tooth_id,
                apex_point=tooth_point,
                top_valley=top_valley,
                bottom_valley=bottom_valley,
                angle=angle,
                grinding_point=tooth_point,
                height=height,
                move_to_grinder=move_to_grinder
            ))

            tooth_id += 1

        return tooth_profiles

    def _filter_close_points(self, points, min_distance):
        """Filter out points too close together"""
        if len(points) == 0:
            return points

        filtered = [points[0]]
        for p in points[1:]:
            if abs(p - filtered[-1]) >= min_distance:
                filtered.append(p)
        return filtered

    def _calculate_tooth_angle(self, top_valley, apex, bottom_valley):
        """Calculate tooth angle in degrees"""
        v1 = np.array([top_valley[0] - apex[0], top_valley[1] - apex[1]], dtype=float)
        v2 = np.array([bottom_valley[0] - apex[0], bottom_valley[1] - apex[1]], dtype=float)

        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)

        if magnitude == 0:
            return 0

        angle = np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))
        return float(np.degrees(angle))

    def calculate_blade_orientation(self):
        """Calculate blade edge orientation"""
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return 0

        fit_result = cv2.fitLine(
            self.blade_edge_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )

        vx, vy = fit_result[0][0], fit_result[1][0]
        angle = np.degrees(np.arctan2(vy, vx))
        return float(angle)

    def generate_grinding_coordinates(self, pixels_per_mm=1.0):
        """Generate grinding coordinates for robot control - TOOTH POINTS"""
        grinding_coords = []
        grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None

        for tooth in self.teeth_profiles:
            move_x_px = tooth.move_to_grinder[0]
            move_y_px = tooth.move_to_grinder[1]

            move_x_mm = move_x_px / pixels_per_mm
            move_y_mm = move_y_px / pixels_per_mm

            distance_px = np.sqrt(move_x_px ** 2 + move_y_px ** 2)
            distance_mm = distance_px / pixels_per_mm

            grinding_coords.append({
                'tooth_id': tooth.tooth_id,
                'tooth_tip_x_px': tooth.grinding_point[0],  # Tooth tip X
                'tooth_tip_y_px': tooth.grinding_point[1],  # Tooth tip Y
                'groove_position_x_px': tooth.grinding_point[0],  # Keep for compatibility
                'groove_position_y_px': tooth.grinding_point[1],  # Keep for compatibility
                'grinder_tip_x_px': grinder_ref[0] if grinder_ref else 0,
                'grinder_tip_y_px': grinder_ref[1] if grinder_ref else 0,
                'move_x_pixels': int(move_x_px),
                'move_y_pixels': int(move_y_px),
                'move_x_mm': round(move_x_mm, 2),
                'move_y_mm': round(move_y_mm, 2),
                'distance_to_grinder_px': round(distance_px, 1),
                'distance_to_grinder_mm': round(distance_mm, 2),
                'tooth_angle_degrees': round(tooth.angle, 2),
                'tooth_height_px': round(tooth.height, 1),
                'groove_angle_degrees': round(tooth.angle, 2),  # Keep for compatibility
                'groove_depth_px': round(tooth.height, 1),  # Keep for compatibility
                'top_peak': tooth.top_valley,
                'bottom_peak': tooth.bottom_valley
            })

        return grinding_coords

    def analyze_blade(self, pixels_per_mm=1.0):
        """Run complete analysis pipeline"""
        self.preprocess_image()
        self.detect_edges()
        self.find_blade_contours()
        self.detect_blade_and_grinder()
        self.teeth_profiles = self.extract_tooth_profiles()
        blade_angle = self.calculate_blade_orientation()
        grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)

        return {
            'blade_angle': blade_angle,
            'num_grooves': len(self.teeth_profiles),  # Keep name for compatibility
            'num_teeth': len(self.teeth_profiles),
            'grinder_tip': self.grinder_tip if hasattr(self, 'grinder_tip') else None,
            'grinder_center': self.grinder_center if hasattr(self, 'grinder_center') else None,
            'grinding_coordinates': grinding_coords,
            'tooth_profiles': self.teeth_profiles
        }

    def visualize_results(self, save_path=None):
        """Visualize detection results - TOOTH POINTS"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Binary image
        axes[0, 1].imshow(self.binary, cmap='gray')
        axes[0, 1].set_title('Binary Image', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Edge detection
        axes[0, 2].imshow(self.edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Detected edges
        viz_image1 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        if self.blade_edge_points is not None:
            for point in self.blade_edge_points[::3]:
                cv2.circle(viz_image1, tuple(point), 2, (255, 0, 0), -1)

        if self.grinder_edge_points is not None:
            for point in self.grinder_edge_points[::3]:
                cv2.circle(viz_image1, tuple(point), 2, (0, 255, 0), -1)

        if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
            cv2.circle(viz_image1, self.grinder_tip, 10, (255, 255, 0), -1)
            cv2.circle(viz_image1, self.grinder_tip, 12, (0, 0, 0), 2)
            cv2.putText(viz_image1, "GRINDER TIP", (self.grinder_tip[0] + 15, self.grinder_tip[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        axes[1, 0].imshow(viz_image1)
        axes[1, 0].set_title('Detected Edges', fontsize=11)
        axes[1, 0].axis('off')

        # Tooth profiles - PEAKS highlighted
        viz_image2 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        for tooth in self.teeth_profiles:
            # Draw TOOTH TIP (peak) - LARGE and BRIGHT
            cv2.circle(viz_image2, tooth.grinding_point, 8, (0, 255, 255), -1)  # Cyan for tooth tip
            cv2.circle(viz_image2, tooth.grinding_point, 10, (255, 255, 255), 2)  # White outline

            # Draw valleys (grooves) - smaller
            cv2.circle(viz_image2, tooth.top_valley, 5, (255, 0, 0), -1)  # Red for valleys
            cv2.circle(viz_image2, tooth.bottom_valley, 5, (255, 0, 0), -1)

            # Draw tooth outline
            cv2.line(viz_image2, tooth.top_valley, tooth.grinding_point, (255, 255, 0), 2)
            cv2.line(viz_image2, tooth.grinding_point, tooth.bottom_valley, (255, 255, 0), 2)

            label_pos = (tooth.grinding_point[0] + 15, tooth.grinding_point[1])
            cv2.putText(viz_image2, f"TOOTH #{tooth.tooth_id}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if hasattr(self, 'grinder_tip') and self.grinder_tip is not None:
            cv2.circle(viz_image2, self.grinder_tip, 12, (255, 128, 0), 3)

        axes[1, 1].imshow(viz_image2)
        axes[1, 1].set_title('Tooth Points (Cyan: Tips, Red: Valleys)', fontsize=11)
        axes[1, 1].axis('off')

        # Movement vectors
        viz_image3 = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        grinder_ref = self.grinder_tip if hasattr(self, 'grinder_tip') else None

        for tooth in self.teeth_profiles:
            cv2.circle(viz_image3, tooth.grinding_point, 8, (0, 255, 255), -1)
            cv2.circle(viz_image3, tooth.grinding_point, 10, (255, 255, 255), 2)

            if grinder_ref is not None:
                cv2.arrowedLine(viz_image3, tooth.grinding_point, grinder_ref,
                                (0, 255, 255), 3, tipLength=0.03)

        if grinder_ref is not None:
            cv2.circle(viz_image3, grinder_ref, 15, (255, 128, 0), 4)
            cv2.drawMarker(viz_image3, grinder_ref, (255, 255, 0),
                           cv2.MARKER_CROSS, 30, 4)

        axes[1, 2].imshow(viz_image3)
        axes[1, 2].set_title('Grinding Path: Tooth Tips → Grinder', fontsize=11)
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def export_coordinates_to_csv(self, filename='grinding_coordinates.csv', pixels_per_mm=1.0):
        """Export grinding coordinates to CSV"""
        import csv

        grinding_coords = self.generate_grinding_coordinates(pixels_per_mm=pixels_per_mm)

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['tooth_id', 'tooth_tip_x_px', 'tooth_tip_y_px',
                          'grinder_tip_x_px', 'grinder_tip_y_px',
                          'move_x_px', 'move_y_px', 'move_x_mm', 'move_y_mm',
                          'distance_to_grinder_px', 'distance_to_grinder_mm',
                          'tooth_angle_deg', 'tooth_height_px',
                          'top_valley_x', 'top_valley_y', 'bottom_valley_x', 'bottom_valley_y']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for coord in grinding_coords:
                writer.writerow({
                    'tooth_id': coord['tooth_id'],
                    'tooth_tip_x_px': coord['tooth_tip_x_px'],
                    'tooth_tip_y_px': coord['tooth_tip_y_px'],
                    'grinder_tip_x_px': coord['grinder_tip_x_px'],
                    'grinder_tip_y_px': coord['grinder_tip_y_px'],
                    'move_x_px': coord['move_x_pixels'],
                    'move_y_px': coord['move_y_pixels'],
                    'move_x_mm': coord['move_x_mm'],
                    'move_y_mm': coord['move_y_mm'],
                    'distance_to_grinder_px': coord['distance_to_grinder_px'],
                    'distance_to_grinder_mm': coord['distance_to_grinder_mm'],
                    'tooth_angle_deg': coord['tooth_angle_degrees'],
                    'tooth_height_px': coord['tooth_height_px'],
                    'top_valley_x': coord['top_peak'][0],
                    'top_valley_y': coord['top_peak'][1],
                    'bottom_valley_x': coord['bottom_peak'][0],
                    'bottom_valley_y': coord['bottom_peak'][1]
                })


def main():
    """Main execution"""
    image_path = 'results/untitled3.png'
    PIXELS_PER_MM = 86.96

    print("=" * 80)
    print("SERRATED BLADE ANALYZER - TOOTH POINT DETECTION")
    print("=" * 80)

    analyzer = SerratedBladeAnalyzer(image_path)
    results = analyzer.analyze_blade(pixels_per_mm=PIXELS_PER_MM)

    print(f"\nBlade orientation: {results['blade_angle']:.2f}°")
    print(f"Teeth detected: {results['num_teeth']}")

    if results['grinder_tip']:
        print(f"Grinder tip: {results['grinder_tip']}")

    print(f"\n{'Tooth':>6} | {'Tip Position':>14} | {'Move (mm)':>14} | {'Distance':>10}")
    print("-" * 55)

    for coord in results['grinding_coordinates']:
        print(f"{coord['tooth_id']:>6} | "
              f"({coord['tooth_tip_x_px']:>4},{coord['tooth_tip_y_px']:>4}) | "
              f"({coord['move_x_mm']:>5.1f},{coord['move_y_mm']:>5.1f}) | "
              f"{coord['distance_to_grinder_mm']:>6.1f}mm")

    analyzer.export_coordinates_to_csv('grinding_coordinates.csv', pixels_per_mm=PIXELS_PER_MM)
    analyzer.visualize_results(save_path='blade_analysis_results.png')

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()