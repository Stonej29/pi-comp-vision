import argparse
import threading
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
from flask import Flask, Response

Gst.init(None)

class DetectionStream:
    def __init__(self, port=8080, video_source=None):
        self.port = port
        self.video_source = video_source
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.pipeline = None
        self.loop = None
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return '<html><body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;min-height:100vh;"><img src="/stream" style="max-width:100%;"></body></html>'

        @self.app.route('/stream')
        def stream():
            return Response(self._generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate(self):
        while True:
            with self.frame_lock:
                if self.latest_frame:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self.latest_frame + b'\r\n'
            time.sleep(0.033)

    def _on_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample:
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                with self.frame_lock:
                    self.latest_frame = bytes(info.data)
                buf.unmap(info)
        return Gst.FlowReturn.OK

    def _on_buffer(self, pad, info):
        buf = info.get_buffer()
        if buf:
            for det in hailo.get_roi_from_buffer(buf).get_objects_typed(hailo.HAILO_DETECTION):
                print(f"{det.get_label()}: {det.get_confidence():.2f}")
        return Gst.PadProbeReturn.OK

    def run(self):
        if self.video_source:
            source = f'filesrc location="{self.video_source}" ! decodebin !'
        else:
            source = 'libcamerasrc ! video/x-raw,format=RGB,width=640,height=480 !'

        pipeline_str = f"""
            {source}
            queue leaky=no max-size-buffers=3 !
            videoflip video-direction=180 !
            videobox autocrop=true !
            videoscale n-threads=2 !
            video/x-raw,format=RGB,width=640,height=640 !
            queue leaky=no max-size-buffers=3 !
            hailonet hef-path=/usr/share/hailo-models/yolov8s_h8l.hef batch-size=1 force-writable=true !
            queue leaky=no max-size-buffers=3 !
            hailofilter so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so qos=false !
            identity name=cb !
            queue leaky=no max-size-buffers=3 !
            hailooverlay !
            videoconvert n-threads=2 !
            jpegenc quality=80 !
            appsink name=sink emit-signals=true sync=false drop=true max-buffers=1
        """

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Detection callback
        cb = self.pipeline.get_by_name("cb")
        cb.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        # Frame capture
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)

        self.pipeline.set_state(Gst.State.PLAYING)

        # Start Flask
        threading.Thread(target=lambda: self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False), daemon=True).start()

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _on_message(self, bus, msg):
        if msg.type == Gst.MessageType.EOS:
            self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
        elif msg.type == Gst.MessageType.ERROR:
            err, _ = msg.parse_error()
            print(f"Error: {err}")
            self.loop.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hailo detection web stream")
    parser.add_argument("-i", "--input", help="Video file path (default: camera)")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Web server port")
    args = parser.parse_args()

    DetectionStream(port=args.port, video_source=args.input).run()
