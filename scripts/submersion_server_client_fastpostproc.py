import socket
import pickle
import struct
import time
import numpy as np
import sys
import cv2
import lunar_tools as lt
from rtd.utils.prompt_provider import PromptProviderMicrophone, PromptProviderTxtFile
from rtd.utils.audio_detector import AudioDetector
from rtd.utils.oscillators import Oscillator
import threading

if len(sys.argv) > 1 and sys.argv[1].lower() == "server":
    from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
    from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
    from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
    from rtd.utils.input_image import InputImageProcessor, AcidProcessor
    # Even though these are imported for the server branch, note that the server will no longer
    # perform optical flow or postprocessing.
    from rtd.utils.optical_flow import OpticalFlowEstimator
    from rtd.utils.posteffect import Posteffect

from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.compression_helpers import send_compressed, recv_compressed
from rtd.utils.optical_flow import OpticalFlowEstimator
from rtd.utils.posteffect import Posteffect

###############################################################################
# SubmersionServer
###############################################################################
class SubmersionServer:
    def __init__(self, host="0.0.0.0", port=9999, device="cuda:0", do_diffusion=True, do_compile=True, bounce=False):
        self.host = host
        self.port = port
        self.device = device
        self.do_diffusion = do_diffusion
        self.do_compile = do_compile
        self.bounce = bounce  # If True, the server will simply echo back the received image without processing

        # These dimensions match the submersion pipeline settings.
        self.height_diffusion = int((384 + 96) * 1.0)
        self.width_diffusion = int((512 + 128) * 1.0)

        # Create and bind the server socket.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency.
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"SubmersionServer listening on {self.host}:{self.port}")

        if not self.bounce:
            # Initialize image processing modules.
            self.input_image_processor = InputImageProcessor(device=device)
            self.input_image_processor.set_flip(do_flip=True, flip_axis=1)

            self.acid_processor = AcidProcessor(
                height_diffusion=self.height_diffusion,
                width_diffusion=self.width_diffusion,
                device=device,
            )
            # Dynamic processing is removed since postprocessing is now handled client‐side.
            # self.dynamic_processor = DynamicProcessor()

            # Removed optical flow and posteffect initializations from the server.
            # self.opt_flow_estimator = OpticalFlowEstimator(use_ema=False)
            # self.posteffect_processor = Posteffect()

            self.de_img = DiffusionEngine(
                use_image2image=True,
                height_diffusion_desired=self.height_diffusion,
                width_diffusion_desired=self.width_diffusion,
                do_compile=self.do_compile,
                do_diffusion=self.do_diffusion,
                device=device,
            )

            if self.do_diffusion:
                self.em = EmbeddingsMixer(self.de_img.pipe)
                init_prompt = 'Dancing people full of glowing neon nerve fibers and filamenets'
                self.embeds = self.em.encode_prompt(init_prompt)
                self.embeds_source = self.em.clone_embeddings(self.embeds)
                self.embeds_target = self.em.clone_embeddings(self.embeds)
                self.de_img.set_embeddings(self.embeds)
                self.fract_blend_embeds = 1.0  # Start with fully blended embedding
                self.transition_start_time = None

            # Removed storage of last diffusion image as it is not needed here.
            # self.last_diffused = None

            self.fps_tracker = lt.FPSTracker()

            print("Submersion server ready.")
        else:
            print("Bounce mode enabled: Server will echo the received image without processing.")

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes from the socket."""
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self, sock):
        """Receive a length-prefixed message."""
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack("!I", raw_msglen)[0]
        return self.recvall(sock, msglen)

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack("!I", len(msg)) + msg
        sock.sendall(msg)

    def handle_client(self, client_sock, addr):
        print(f"Connected by {addr}")
        while True:
            try:
                self.fps_tracker.start_segment("Receive Data")
                data = self.recv_msg(client_sock)
                if data is None:
                    print("Client disconnected")
                    break

                # Unpickle the received payload.
                payload = pickle.loads(data)

                if not self.bounce:
                    self.fps_tracker.start_segment("Process Prompts")
                    mic_prompt = payload.get("mic_prompt")
                    if mic_prompt and self.do_diffusion:
                        print(f"New microphone prompt received: {mic_prompt}")
                        self.transition_start_time = time.time()
                        self.embeds_source = self.em.clone_embeddings(self.embeds)
                        self.embeds_target = self.em.encode_prompt(mic_prompt)
                        self.fract_blend_embeds = 0.0

                    txt_file_prompt = payload.get("txt_file_prompt")
                    if txt_file_prompt and self.do_diffusion:
                        print(f"New text file prompt received: {txt_file_prompt}")
                        self.transition_start_time = time.time()
                        self.embeds_source = self.em.clone_embeddings(self.embeds)
                        self.embeds_target = self.em.encode_prompt(txt_file_prompt)
                        self.fract_blend_embeds = 0.0

                    # Removed dynamic processor related processing.

                print(f"Received payload: {payload}")

                if self.bounce:
                    self.fps_tracker.start_segment("Bounce Mode")
                    img = recv_compressed(client_sock)
                    if img is None:
                        break
                    send_compressed(client_sock, img, quality=90)
                    continue

                print("Received payload")

                self.fps_tracker.start_segment("Receive Image")
                print("Waiting for compressed image from client...")
                img_cam = recv_compressed(client_sock)
                if img_cam is None:
                    print("Client disconnected during image receive")
                    break
                if not isinstance(img_cam, np.ndarray):
                    print(f"Invalid image received: {type(img_cam)}")
                    continue

                print("Received image")

                self.fps_tracker.start_segment("Input Image Processing")
                self.input_image_processor.set_human_seg(payload.get("do_human_seg", True))
                self.input_image_processor.set_resizing_factor_humanseg(0.4)
                self.input_image_processor.set_blur(payload.get("do_blur", False))
                self.input_image_processor.set_brightness(payload.get("brightness", 1.0))
                self.input_image_processor.set_infrared_colorize(payload.get("do_infrared_colorize", False))
                img_proc, human_seg_mask = self.input_image_processor.process(img_cam.copy())

                if not payload.get("do_human_seg", True):
                    human_seg_mask = np.ones_like(img_proc).astype(np.float32) / 255

                self.fps_tracker.start_segment("Acid Processing")
                self.acid_processor.set_acid_strength(payload.get("acid_strength", 0.11))
                self.acid_processor.set_coef_noise(payload.get("coef_noise", 0.15))
                self.acid_processor.set_acid_tracers(payload.get("do_acid_tracers", True))
                self.acid_processor.set_acid_strength_foreground(payload.get("acid_strength_foreground", 0.11))
                self.acid_processor.set_zoom_factor(payload.get("zoom_factor", 1.0))
                self.acid_processor.set_x_shift(payload.get("x_shift", 0))
                self.acid_processor.set_y_shift(payload.get("y_shift", 0))
                self.acid_processor.set_do_acid_wobblers(payload.get("do_acid_wobblers", False))
                self.acid_processor.set_color_matching(payload.get("color_matching", 0.5))
                img_acid = self.acid_processor.process(img_proc, human_seg_mask)

                self.fps_tracker.start_segment("Diffusion")
                self.de_img.set_input_image(img_acid)
                self.de_img.set_guidance_scale(0.5)
                self.de_img.set_strength(1 / self.de_img.num_inference_steps + 0.00001)
                img_diffusion = np.array(self.de_img.generate())

                # Server no longer applies any postprocessing; just send the diffusion result.
                self.fps_tracker.start_segment("Send Result")
                send_compressed(client_sock, img_diffusion, quality=90)

                self.fps_tracker.print_fps()

            except Exception as e:
                print("Error handling client:", e)
                import traceback
                traceback.print_exc()
                break

        client_sock.close()

    def serve_forever(self):
        """Main loop to accept and serve clients."""
        while True:
            client_sock, addr = self.server_socket.accept()
            self.handle_client(client_sock, addr)

###############################################################################
# SubmersionClient
###############################################################################
class SubmersionClient:
    def __init__(self, server_host="localhost", server_port=9999):
        self.server_host = server_host
        self.server_port = server_port

        # Camera settings.
        self.shape_hw_cam = (576, 1024)
        # Renderer settings.
        self.width_render = 1920
        self.height_render = 1080
        self.do_fullscreen = True

        # Initialize LT camera, meta input, and renderer.
        self.cam = lt.WebCam(shape_hw=self.shape_hw_cam)
        self.meta_input = lt.MetaInput()
        self.renderer = lt.Renderer(
            width=self.width_render,
            height=self.height_render,
            backend="opencv",
            do_fullscreen=self.do_fullscreen,
        )

        # Initialize the microphone prompt and speech detection.
        self.speech_detector = lt.Speech2Text()
        self.prompt_provider_microphone = PromptProviderMicrophone()
        self.prompt_provider_txt_file = PromptProviderTxtFile("materials/prompts/dancing_fibers.txt")
        self.audio_detector = AudioDetector()
        self.oscillator = Oscillator()
        self.fps_tracker = lt.FPSTracker()

        # Initialize local processors for optical flow and posteffect processing.
        self.opt_flow_estimator = OpticalFlowEstimator(use_ema=False)
        self.posteffect_processor = Posteffect()
        self.input_image_processor = InputImageProcessor()  # For computing human segmentation locally.

        # Shared variables for asynchronous networking.
        self.network_lock = threading.Lock()
        self.latest_cam_image = None
        self.latest_remote_diffusion = None

        # Connect to the server.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.server_host, self.server_port))
        print(f"Connected to server at {self.server_host}:{self.server_port}")

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack("!I", len(msg)) + msg
        sock.sendall(msg)

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes."""
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self, sock):
        """Receive a length-prefixed message."""
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack("!I", raw_msglen)[0]
        return self.recvall(sock, msglen)

    def network_loop(self):
        """Asynchronous thread for sending camera images to the server and receiving img_diffusion."""
        while True:
            try:
                with self.network_lock:
                    cam_img = self.latest_cam_image.copy() if self.latest_cam_image is not None else None

                payload = {
                    "do_human_seg": self.meta_input.get(akai_lpd8="B1", akai_midimix="E3", button_mode="toggle", val_default=True),
                    "acid_strength": self.meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.05),
                    "acid_strength_foreground": self.meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.05),
                    "coef_noise": self.meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=0.3, val_default=0.05),
                    "zoom_factor": self.meta_input.get(akai_lpd8="F1", akai_midimix="H2", val_min=0.5, val_max=1.5, val_default=1.0),
                    "x_shift": int(self.meta_input.get(akai_midimix="H0", val_min=-50, val_max=50, val_default=0)),
                    "y_shift": int(self.meta_input.get(akai_midimix="H1", val_min=-50, val_max=50, val_default=0)),
                    "color_matching": self.meta_input.get(akai_lpd8="G0", akai_midimix="G0", val_min=0, val_max=1, val_default=0.5),
                    "mic_prompt": self.prompt_provider_microphone.get_current_prompt() if self.prompt_provider_microphone.handle_unmute_button(self.meta_input.get(akai_lpd8="A1", akai_midimix="A3", button_mode="held_down")) else None,
                    "txt_file_prompt": self.prompt_provider_txt_file.get_current_prompt() if self.meta_input.get(akai_lpd8="C0", akai_midimix="A4", button_mode="pressed_once") else None,
                    "dynamic_transcript": self.speech_detector.transcript if self.speech_detector.handle_unmute_button(self.meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")) else None,
                    "brightness": self.meta_input.get(akai_midimix="A2", val_min=0.0, val_max=2, val_default=1.0),
                    "do_infrared_colorize": self.meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False),
                    "dyn_prompt_restore_backup": self.meta_input.get(akai_midimix="F3", button_mode="released_once"),
                    "dyn_prompt_del_current": self.meta_input.get(akai_midimix="F4", button_mode="released_once"),
                    "prompt_transition_time": self.meta_input.get(akai_lpd8="G1", val_min=1, val_max=20, val_default=8.0),
                }
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_msg(self.sock, data)

                print(f"Sending payload: {payload}")

                if cam_img is not None:
                    if cam_img.shape[:2] != self.cam.shape_hw:
                        desired_width = self.cam.shape_hw[1]
                        desired_height = self.cam.shape_hw[0]
                        cam_img = cv2.resize(cam_img, (desired_width, desired_height))
                    print("Sending compressed image to server...")
                    send_compressed(self.sock, cam_img, quality=90)
                    print("Image sent successfully")
                else:
                    print("Warning: No camera image available to send")
                    # Send a dummy image to keep the protocol in sync
                    dummy_img = np.zeros((self.cam.shape_hw[0], self.cam.shape_hw[1], 3), dtype=np.uint8)
                    send_compressed(self.sock, dummy_img, quality=90)

                print("Sent payload")

                print("Waiting for processed image from server...")
                processed_image = recv_compressed(self.sock)
                print("Received processed image")
                
                if processed_image is not None:
                    with self.network_lock:
                        self.latest_remote_diffusion = processed_image
                else:
                    print("Disconnected from server in network thread")
                    break
            except Exception as e:
                print("Error in network thread:", e)
                import traceback
                traceback.print_exc()
                break
            time.sleep(0.01)  # Small sleep to prevent a tight loop

    def run(self):
        # Start the network communication loop in a separate thread.
        network_thread = threading.Thread(target=self.network_loop, daemon=True)
        network_thread.start()
        while True:
            self.fps_tracker.start_segment("Client Processing")
            t_processing_start = time.time()

            self.fps_tracker.start_segment("Camera Capture")
            img_cam = self.cam.get_img()
            with self.network_lock:
                self.latest_cam_image = img_cam.copy()

            # Perform local processing: compute human segmentation and optical flow.
            self.fps_tracker.start_segment("Input Image Processing")
            img_proc, human_seg_mask = self.input_image_processor.process(img_cam.copy())
            if human_seg_mask is None or not np.any(human_seg_mask):
                human_seg_mask = np.ones_like(img_proc, dtype=np.float32) / 255

            self.fps_tracker.start_segment("Optical Flow")
            opt_flow = self.opt_flow_estimator.get_optflow(img_cam.copy(), low_pass_kernel_size=55, window_length=55)

            with self.network_lock:
                remote_diff = self.latest_remote_diffusion.copy() if self.latest_remote_diffusion is not None else None

            # Retrieve postprocessing parameters.
            postproc_func_coef1 = self.meta_input.get(akai_lpd8="H0", akai_midimix="G1", val_min=0, val_max=1, val_default=0.5)
            postproc_func_coef2 = self.meta_input.get(akai_lpd8="H1", akai_midimix="G2", val_min=0, val_max=1, val_default=0.5)
            postproc_mod_button1 = self.meta_input.get(akai_midimix="G4", button_mode="toggle", val_default=True)
            sound_volume = 0
            if self.meta_input.get(akai_midimix="D4", button_mode="toggle", val_default=False):
                sound_volume = self.audio_detector.get_last_volume()

            # If no remote diffusion image is available, fall back to the camera image.
            if remote_diff is None:
                remote_diff = img_cam.copy()

            if opt_flow is None:
                opt_flow = np.zeros(remote_diff.shape, dtype=remote_diff.dtype)
                opt_flow = opt_flow[:,:,::2]

            self.fps_tracker.start_segment("Post Processing")
            output_to_render, _ = self.posteffect_processor.process(
                remote_diff,
                human_seg_mask.astype(np.float32) / 255,
                opt_flow,
                postproc_func_coef1,
                postproc_func_coef2,
                postproc_mod_button1,
                sound_volume,
            )

            self.fps_tracker.start_segment("Rendering")
            self.renderer.render(output_to_render)

            t_processing = time.time() - t_processing_start
            self.fps_tracker.print_fps()

        self.sock.close()

###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [server|client]")
        sys.exit(1)

    role = sys.argv[1].lower()
    if len(sys.argv) > 2:
        server_ip = sys.argv[2].lower()
    else:
        server_ip = "localhost"

    if role == "server":
        # To test pure network latency, enable bounce mode by setting bounce=True.
        server = SubmersionServer(bounce=False, do_compile=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server shutting down.")
    elif role == "client":
        client = SubmersionClient(server_host=server_ip)
        client.run()
    else:
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
