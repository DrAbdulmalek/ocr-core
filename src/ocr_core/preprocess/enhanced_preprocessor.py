# src/scanner_fixer/enhanced_preprocessor.py
"""
Enhanced Document Preprocessor for Medical OCR.
Full pipeline: Shadow Removal → Noise Removal → Perspective Correction →
Advanced Deskew (Hough Lines) → Auto-Crop → Sharpening.

Based on Kimi Code review recommendations for DrAbdulmalek/scanner-fixer.
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Complete document preprocessing pipeline for medical OCR."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.debug_dir = Path("debug_preprocess")
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)

    # ==================== Noise Removal ====================
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Strong noise removal using Non-local Means + Bilateral Filter."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Non-local Means denoising (excellent for medical scans)
        denoised = cv2.fastNlMeansDenoising(
            gray, h=10, searchWindowSize=21, templateWindowSize=7
        )
        # Bilateral Filter (preserves edges while smoothing)
        bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)

    # ==================== Shadow Removal ====================
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows using CLAHE + background normalization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Background subtraction for uneven lighting
        blur = cv2.GaussianBlur(enhanced, (0, 0), 15)
        normalized = cv2.addWeighted(enhanced, 4, blur, -4, 128)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    # ==================== Advanced Deskew ====================
    def _advanced_deskew(self, image: np.ndarray) -> np.ndarray:
        """Advanced deskew using Hough Lines + Projection Profile analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough Lines for angle detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        angles = []

        if lines is not None:
            for rho_theta in lines[:, 0]:
                theta = rho_theta[1]
                angle = np.degrees(theta) - 90
                if -45 <= angle <= 45:  # reasonable range for documents
                    angles.append(angle)

        if angles:
            median_angle = float(np.median(angles))
        else:
            # Fallback: minAreaRect on text pixels
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) == 0:
                logger.warning("No text pixels found for deskew, skipping")
                return image
            rect = cv2.minAreaRect(coords)
            median_angle = rect[2]
            if median_angle < -45:
                median_angle += 90

        # Apply rotation
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.info(f"Deskew angle: {median_angle:.2f} degrees")
        return rotated

    # ==================== Perspective Correction ====================
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left (smallest sum)
        rect[2] = pts[np.argmax(s)]   # bottom-right (largest sum)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right (smallest diff)
        rect[3] = pts[np.argmax(diff)]  # bottom-left (largest diff)
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get a top-down view."""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute output dimensions
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Detect document edges and apply perspective correction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Auto Canny with dynamic thresholds
        v = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)

        # Morphological operations to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Find document contour
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found, skipping perspective correction")
            return image

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            warped = self._four_point_transform(image, pts)
            logger.info("Perspective correction applied (4 points detected)")
            return warped
        else:
            logger.info(f"Detected {len(approx)} points (not 4), skipping perspective")
            return image

    # ==================== Auto-Crop ====================
    def _auto_crop(self, image: np.ndarray, padding: int = 5) -> np.ndarray:
        """Remove empty borders from the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[
            max(0, y - padding):y + h + padding,
            max(0, x - padding):x + w + padding
        ]

    # ==================== Main Pipeline ====================
    def process(self, input_path, output_path: str = None, debug: bool = None) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            input_path: Path to input image (str or Path)
            output_path: Optional path to save output. If None, returns array only.
            debug: Override debug flag for this call.

        Returns:
            Processed image as numpy array.
        """
        if debug is None:
            debug = self.debug

        # Read image
        if isinstance(input_path, np.ndarray):
            img = input_path.copy()
        else:
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Cannot read image: {input_path}")
                return None

        original = img.copy()

        # Step 1: Shadow removal
        no_shadows = self._remove_shadows(original)
        if debug:
            cv2.imwrite(str(self.debug_dir / "01_no_shadows.jpg"), no_shadows)

        # Step 2: Noise removal
        no_noise = self._remove_noise(no_shadows)
        if debug:
            cv2.imwrite(str(self.debug_dir / "02_no_noise.jpg"), no_noise)

        # Step 3: Perspective correction
        warped = self._correct_perspective(no_noise)
        if debug:
            cv2.imwrite(str(self.debug_dir / "03_perspective.jpg"), warped)

        # Step 4: Advanced deskew
        deskewed = self._advanced_deskew(warped)
        if debug:
            cv2.imwrite(str(self.debug_dir / "04_deskewed.jpg"), deskewed)

        # Step 5: Auto-crop
        cropped = self._auto_crop(deskewed)

        # Step 6: Final sharpening
        final = cv2.detailEnhance(cropped, sigma_s=10, sigma_r=0.15)

        if debug:
            cv2.imwrite(str(self.debug_dir / "05_final.jpg"), final)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), final)
            logger.info(f"Saved processed image to {output_path}")

        return final

    def process_array(self, image: np.ndarray, debug: bool = None) -> np.ndarray:
        """Process a numpy array directly (for Gradio integration)."""
        return self.process(image, output_path=None, debug=debug)


if __name__ == "__main__":
    import sys
    proc = DocumentPreprocessor(debug=True)
    input_file = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"
    result = proc.process(input_file, output_file)
    if result is not None:
        print("Done!")
    else:
        print("Failed to process image.")