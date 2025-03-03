# RTD - Real-time Diffusion

RTD is a real-time audio-visual system that transforms webcam input using distilled stable diffusion models. The system captures people in front of the camera, applies various real-time transformations using AI diffusion models, and renders the results with customizable visual effects.

![Submersion Demo](path/to/demo/image.gif)

## Features

- **Real-time Diffusion**: Transform camera input using SDXL Turbo for smooth, real-time visual effects
- **Voice-controlled Prompts**: Update the diffusion prompts using microphone input on the fly
- **MIDI Control**: Use AKAI Midimix or other MIDI controllers to adjust all effects parameters
- **Client-Server Architecture**: Run processing on a separate machine from the capture/display device
- **Visual Effects**:
  - Acid-like visual trails and tracers
  - Human segmentation for selective effects application
  - Optical flow for liquid-like movement effects
  - Dynamic post-processing for additional visual transformations
- **Audio Reactivity**: Sound-responsive visual effects

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 8GB VRAM
- [Optional] AKAI Midimix or other MIDI controller
- Webcam

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lunarring/rtd.git
   cd rtd
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```
   
   Alternatively, you can install directly from the pyproject.toml:
   ```bash
   pip install .
   ```

   The installation will automatically download the required dependencies including:
   - PyTorch with CUDA support
   - Diffusers
   - Hugging Face Hub
   - Lunar Tools
   - StableFast
   - Other required libraries

3. [Optional] If you plan to use the client-server architecture, install on both machines.

## Usage

### Client-Server Mode

The system can operate in a distributed mode where one machine captures and displays (client) while another performs the computation (server).

1. Start the server:
   ```bash
   python scripts/submersion_server_client.py server
   ```

2. Start the client (on the display machine):
   ```bash
   python scripts/submersion_server_client.py client [server_ip]
   ```
   Replace `[server_ip]` with the IP address of your server.

## How Submersion Works

### Camera Input Pipeline
- **Input Capture**: The system captures frames from a webcam in real-time.
- **Human Segmentation**: [Optional] Using machine learning models to separate people from backgrounds.
- **Input Processing**: The raw image is adjusted (brightness, rotation, etc.) based on control parameters.

### Diffusion Process
The core of Submersion uses a distilled version of Stable Diffusion (SDXL Turbo) to transform images:
- **Image Conditioning**: Camera input is processed and used as a conditioning input.
- **Prompt Embeddings**: Text prompts are converted to embeddings using CLIP.
- **Real-time Inference**: The diffusion model processes the input with reduced inference steps for real-time performance.
- **Smooth Transitions**: When prompts change, the system smoothly interpolates between the previous and new embedding.

### Effects System
The system offers a rich palette of visual effects:
- **Acid Processor**: Creates trailing, morphing visuals based on previous frames
- **Optical Flow**: Analyzes movement to create fluid, liquid-like effects
- **Dynamic Processor**: Allows complex image manipulations based on voice instructions
- **Post-processing**: Additional visual effects including color manipulation and distortion

### Control System
Submersion can be controlled via several methods:
- **MIDI Controllers**: Parameters can be adjusted in real-time using AKAI Midimix or other MIDI controllers
- **Voice Control**: New prompts can be injected by speaking into a microphone
- **Parameter Oscillators**: Effects can automatically vary over time using built-in oscillators
- **Audio Reactivity**: Visual parameters can respond to audio input levels

## Hardware Requirements

For optimal performance:
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070 or better recommended)
- **CPU**: Modern multi-core CPU
- **RAM**: 16GB+
- **Webcam**: Any decent webcam (1080p recommended)
- **[Optional] MIDI Controller**: AKAI Midimix or compatible controller
- **[Optional] Microphone**: For voice prompt control

## Acknowledgments

This project incorporates Lunar Tools and TAC packages.

## License

BSD 3-Clause License