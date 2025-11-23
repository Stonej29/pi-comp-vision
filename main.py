"""
Auto-tracking camera system with person detection using Hailo AI accelerator.

Streams video with smooth zoom that follows detected persons.
"""

import argparse
import threading
import time
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import hailo
from flask import Response
import numpy as np
import cv2

from scripts import TrackingState, build_pipeline, create_app, generate_frames

Gst.init(None)


class DetectionStream:
    """Real-time person detection and auto-tracking camera stream."""

    def __init__(self, port=8080, video_source=None, smooth_factor=0.1,
                 zoom_out_delay=30, confidence_threshold=0.5, padding=0.3,
                 show_boxes=True, zoom_mode=True, show_fps=False):
        self.port = port
        self.video_source = video_source
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.pipeline = None
        self.loop = None

        # Flask setup
        self.app = create_app()
        self._setup_stream_route()

        # Tracking state
        self.tracking = TrackingState(
            smooth_factor=smooth_factor,
            zoom_out_delay=zoom_out_delay,
            confidence_threshold=confidence_threshold,
            padding=padding
        )
        self.confidence_threshold = confidence_threshold
        self.show_boxes = show_boxes
        self.zoom_mode = zoom_mode
        self.show_fps = show_fps
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

    def _setup_stream_route(self):
        @self.app.route('/stream')
        def stream():
            return Response(
                generate_frames(self.frame_lock, lambda: self.latest_frame),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

    def _on_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample:
            caps = sample.get_caps()
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # Smooth interpolation toward target
                self.tracking.interpolate()

                # Get frame dimensions from caps
                height = caps.get_structure(0).get_value("height")
                width = caps.get_structure(0).get_value("width")
                frame = np.ndarray((height, width, 3), buffer=info.data, dtype=np.uint8)

                if frame is not None:
                    h, w = frame.shape[:2]
                    x, y, cw, ch = self.tracking.get_crop_pixels(w, h)

                    if self.zoom_mode:
                        # Crop and resize back to original size
                        cropped = frame[y:y+ch, x:x+cw]
                        if cropped.size > 0:
                            zoomed = cv2.resize(cropped, (w, h))
                            if self.show_fps:
                                cv2.putText(zoomed, f"{self.fps} FPS", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            _, jpeg = cv2.imencode('.jpg', zoomed, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            with self.frame_lock:
                                self.latest_frame = jpeg.tobytes()
                    else:
                        # Draw virtual camera frame (copy since we're modifying)
                        frame = frame.copy()
                        cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 255), 3)
                        if self.show_fps:
                            cv2.putText(frame, f"{self.fps} FPS", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        with self.frame_lock:
                            self.latest_frame = jpeg.tobytes()

                buf.unmap(info)
        return Gst.FlowReturn.OK

    def _on_buffer(self, pad, info):
        buf = info.get_buffer()
        if buf:
            roi = hailo.get_roi_from_buffer(buf)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            persons = []

            for det in detections:
                if det.get_label() != "person" or det.get_confidence() < self.confidence_threshold:
                    roi.remove_object(det)
                else:
                    persons.append(det)
                    if not self.show_boxes:
                        roi.remove_object(det)

            # Update target crop based on all detected persons
            self.tracking.update_target(persons)

        return Gst.PadProbeReturn.OK

    def run(self):
        self.pipeline = build_pipeline(self.video_source)

        # Detection callback
        cb = self.pipeline.get_by_name("cb")
        cb.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        # Frame capture
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)

        self.pipeline.set_state(Gst.State.PLAYING)

        # Start Flask
        threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False),
            daemon=True
        ).start()

        print(f"Stream available at http://0.0.0.0:{self.port}")

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", lambda bus, msg: self._on_message(bus, msg))

        # Handle Ctrl+C via GLib
        def shutdown():
            print("\nShutting down...")
            self.pipeline.send_event(Gst.Event.new_eos())

        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, 2, shutdown)  # 2 = SIGINT

        self.loop.run()

    def _on_message(self, bus, msg):
        if msg.type == Gst.MessageType.EOS:
            print("End-of-Stream reached.")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Error: {err.message}")
            self.loop.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-tracking camera with person detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", help="Video file path (default: camera)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Web server port")
    parser.add_argument("-s", "--smooth", type=float, default=0.1, help="Smooth factor (lower = smoother)")
    parser.add_argument("-d", "--delay", type=int, default=30, help="Frames to wait before zooming out")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum detection confidence")
    parser.add_argument("--padding", type=float, default=0.3, help="Padding around detected person")
    parser.add_argument("--no-boxes", action="store_true", help="Hide detection boxes")
    parser.add_argument("--frame-mode", action="store_true", help="Show virtual frame instead of zooming")
    parser.add_argument("--fps", action="store_true", help="Show FPS counter")
    args = parser.parse_args()

    DetectionStream(
        port=args.port,
        video_source=args.input,
        smooth_factor=args.smooth,
        zoom_out_delay=args.delay,
        confidence_threshold=args.confidence,
        padding=args.padding,
        show_boxes=not args.no_boxes,
        zoom_mode=not args.frame_mode,
        show_fps=args.fps
    ).run()
