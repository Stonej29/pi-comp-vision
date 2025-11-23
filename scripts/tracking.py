"""Zoom tracking state and calculations."""


class TrackingState:
    """Manages smooth zoom tracking of detected persons."""

    def __init__(self, smooth_factor=0.1, zoom_out_delay=30,
                 confidence_threshold=0.5, padding=0.3):
        self.current_crop = [0.0, 0.0, 1.0, 1.0]
        self.target_crop = [0.0, 0.0, 1.0, 1.0]
        self.smooth_factor = smooth_factor
        self.frames_without_person = 0
        self.zoom_out_delay = zoom_out_delay
        self.confidence_threshold = confidence_threshold
        self.padding = padding

    def interpolate(self):
        """Smoothly interpolate current crop toward target."""
        for i in range(4):
            self.current_crop[i] += (self.target_crop[i] - self.current_crop[i]) * self.smooth_factor

    def update_target(self, persons):
        """Update target crop based on detected persons.

        Args:
            persons: List of detection objects with get_bbox() method
        """
        if persons:
            self.frames_without_person = 0

            # Calculate bounding box that contains all persons
            min_x = min(det.get_bbox().xmin() for det in persons)
            min_y = min(det.get_bbox().ymin() for det in persons)
            max_x = max(det.get_bbox().xmin() + det.get_bbox().width() for det in persons)
            max_y = max(det.get_bbox().ymin() + det.get_bbox().height() for det in persons)

            # Calculate center and size
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y

            # Make crop square using the larger dimension
            size = max(width, height) * (1 + 2 * self.padding)
            # Clamp to frame bounds
            x = max(0.0, min(cx - size / 2, 1.0 - size))
            y = max(0.0, min(cy - size / 2, 1.0 - size))
            size = min(size, 1.0)
            self.target_crop = [x, y, size, size]
        else:
            # Delay before zooming out
            self.frames_without_person += 1
            if self.frames_without_person > self.zoom_out_delay:
                self.target_crop = [0.0, 0.0, 1.0, 1.0]

    def get_crop_pixels(self, width, height):
        """Convert normalized crop to pixel coordinates.

        Returns:
            tuple: (x, y, crop_width, crop_height) in pixels
        """
        x = int(self.current_crop[0] * width)
        y = int(self.current_crop[1] * height)
        cw = int(self.current_crop[2] * width)
        ch = int(self.current_crop[3] * height)
        return x, y, cw, ch
