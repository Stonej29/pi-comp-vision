"""Flask web streaming setup."""

import time
from flask import Flask, Response


def create_app():
    """Create and configure Flask application.

    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)

    @app.route('/')
    def index():
        return '<html><body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;min-height:100vh;"><img src="/stream" style="max-width:100%;"></body></html>'

    return app


def generate_frames(frame_lock, get_frame):
    """Generator for MJPEG stream frames.

    Args:
        frame_lock: Threading lock for frame access
        get_frame: Callable that returns current frame bytes

    Yields:
        bytes: MJPEG frame data
    """
    while True:
        with frame_lock:
            frame = get_frame()
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        time.sleep(0.033)
