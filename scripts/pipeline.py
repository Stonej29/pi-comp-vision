"""GStreamer pipeline setup for Hailo AI inference."""

import os
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


MODEL_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"
POST_PROCESS_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"


def build_pipeline(video_source=None):
    """Build GStreamer pipeline for video capture and Hailo inference.

    Args:
        video_source: Path to video file, or None for camera

    Returns:
        Gst.Pipeline: Configured GStreamer pipeline
    """
    # Check for required Hailo files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: YOLO model not found at {MODEL_PATH}")
        print("Please install hailo-models or specify correct path")
        sys.exit(1)

    if not os.path.exists(POST_PROCESS_PATH):
        print(f"Error: Post-process library not found at {POST_PROCESS_PATH}")
        print("Please install hailo-tappas or specify correct path")
        sys.exit(1)

    if video_source:
        abs_path = os.path.abspath(video_source)
        if not os.path.exists(abs_path):
            print(f"Error: Video file not found: {abs_path}")
            sys.exit(1)
        source = f'filesrc location="{abs_path}" ! qtdemux ! h264parse ! avdec_h264 ! videoconvert !'
        sync = 'true'
    else:
        source = 'libcamerasrc ! video/x-raw,format=RGB,width=640,height=480 !'
        sync = 'false'

    pipeline_str = f"""
        {source}
        queue leaky=no max-size-buffers=3 !
        videoscale n-threads=2 add-borders=true !
        video/x-raw,format=RGB,width=640,height=640 !
        queue leaky=no max-size-buffers=3 !
        hailonet hef-path={MODEL_PATH} batch-size=1 force-writable=true !
        queue leaky=no max-size-buffers=3 !
        hailofilter so-path={POST_PROCESS_PATH} qos=false !
        identity name=cb !
        queue leaky=no max-size-buffers=3 !
        hailooverlay !
        videoscale n-threads=2 !
        video/x-raw,width=1280,height=720 !
        videoconvert n-threads=2 !
        video/x-raw,format=BGR !
        appsink name=sink emit-signals=true sync={sync} drop=true max-buffers=1
    """

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"Error: Failed to create pipeline: {e}")
        sys.exit(1)

    return pipeline


def handle_message(bus, msg, loop, pipeline):
    """Handle GStreamer bus messages.

    Args:
        bus: GStreamer bus
        msg: Message object
        loop: GLib main loop
        pipeline: GStreamer pipeline
    """
    if msg.type == Gst.MessageType.EOS:
        pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
    elif msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error: {err.message}")
        loop.quit()
