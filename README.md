# Pi Computer Vision - Auto-Tracking Camera

Real-time person detection and auto-tracking camera system for Raspberry Pi 5 with Hailo AI accelerator.

## Features

- **Person Detection**: YOLOv8s model running on Hailo-8L NPU
- **Auto-Tracking Zoom**: Smooth camera zoom that follows detected persons
- **Web Streaming**: Live MJPEG stream accessible from any browser
- **Zoom Hysteresis**: Delayed zoom-out prevents jittery tracking

## Requirements

- Raspberry Pi 5
- Hailo-8L AI Kit
- Pi Camera Module (wide-angle supported)
- Python 3.x

### Dependencies

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gst-1.0 python3-flask python3-opencv python3-numpy
```

## Usage

```bash
# Run with camera
python detection_web.py

# Run with video file
python detection_web.py -i video.mp4

# Custom port
python detection_web.py -p 8000
```

Access the stream at `http://<pi-ip>:8080`

## Configuration

Edit these values in `detection_web.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `smooth_factor` | 0.1 | Zoom smoothness (lower = smoother) |
| `zoom_out_delay` | 30 | Frames to wait before zooming out |
| `padding` | 0.3 | Extra space around detected person |
| Confidence threshold | 0.5 | Minimum detection confidence |

## How It Works

1. Camera captures 640x480 frames
2. Hailo NPU runs YOLOv8s inference
3. Best person detection (highest confidence) is selected
4. Target crop region is calculated with padding
5. Current crop smoothly interpolates toward target
6. Cropped/zoomed frame is streamed via Flask