# Pi Auto-Tracking Camera

Real-time person detection and auto-tracking camera system for Raspberry Pi 5 with Hailo AI accelerator.

![Demo](demo.gif)

## Features

- **Person Detection**: YOLOv8s model running on Hailo-8L NPU at ~30 FPS
- **Auto-Tracking Zoom**: Smooth camera zoom that follows detected persons
- **Multi-Person Tracking**: Automatically frames all detected people
- **Web Streaming**: Live MJPEG stream accessible from any browser
- **Zoom Hysteresis**: Delayed zoom-out prevents jittery tracking
- **Virtual Frame Mode**: Preview tracking region without zooming
- **Configurable**: All tracking parameters adjustable via command line

## Requirements

- Raspberry Pi 5
- Hailo-8L AI Kit
- Pi Camera Module (wide-angle supported)
- Python 3.x

### Dependencies

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gst-1.0 python3-flask python3-opencv python3-numpy
pip install -r requirements.txt
```

## Project Structure

```
├── main.py                   # Main entry point
├── scripts/
│   ├── tracking.py           # Zoom tracking state and calculations
│   ├── pipeline.py           # GStreamer pipeline setup
│   └── streaming.py          # Flask web streaming
├── requirements.txt
└── README.md
```

## Usage

```bash
# Run with camera
python detection_web.py

# Run with video file
python detection_web.py -i video.mp4

# Virtual frame mode (shows tracking region without zooming)
python detection_web.py --frame-mode

# Hide detection boxes
python detection_web.py --no-boxes

# Custom settings
python detection_web.py -p 8000 -s 0.05 -c 0.6
```

Access the stream at `http://<pi-ip>:8080`

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | camera | Video file path |
| `-p, --port` | 8080 | Web server port |
| `-s, --smooth` | 0.1 | Smooth factor (lower = smoother) |
| `-d, --delay` | 30 | Frames to wait before zooming out |
| `-c, --confidence` | 0.5 | Minimum detection confidence |
| `--padding` | 0.3 | Padding around detected person |
| `--no-boxes` | false | Hide detection boxes |
| `--frame-mode` | false | Show virtual frame instead of zooming |
| `--fps` | false | Show FPS counter |

## How It Works

1. Camera captures 640x480 frames
2. Hailo NPU runs YOLOv8s inference
3. All persons above confidence threshold are detected
4. Bounding box containing all persons is calculated
5. Target crop region is calculated with padding
6. Current crop smoothly interpolates toward target
7. Cropped/zoomed frame is streamed via Flask