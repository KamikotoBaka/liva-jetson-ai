# Liva

Liva is a local, privacy-focused voice assistant specifically optimized for the **NVIDIA Jetson Orin Nano (8 GB)**. 
Built on a modular **Pipe-and-Filter architecture**, the system allows individual components (models, adapters, smart home integrations) 
to be easily swapped or extended without refactoring the core application.

The prototype features an intelligent **AI Router** that dynamically splits incoming requests into two distinct execution paths:
- **Non-Thinking Model:** For lightning-fast intent recognition and smart home command routing.
- **Thinking Model:** For deep reasoning, complex contextual understanding, and detailed conversational responses.

## 🚀 Main Features

- **100% Local Execution:** Speech-to-Text (STT), Intent Routing, and Text-to-Speech (TTS) run entirely on-device to ensure maximum data privacy.
- **Pipe-and-Filter Architecture:** High modularity with fully replaceable pipeline stages.
- **Dual-Path AI Routing:** Optimized latency and intelligence using specialized local LLMs.
- **Smart Home Integrations:** Native control via REST APIs and MQTT brokers.
- **Custom Command Engine:** Simple JSON-based configuration for custom voice triggers.
- **Containerized Deployment:** Fully Dockerized stack optimized for NVIDIA Jetson hardware.

## 🏗️ Architecture & Pipeline

Liva processes voice commands sequentially through a structured pipeline:

1. **Audio Input:** Captured from the hardware microphone or received via the frontend API.
2. **STT (Speech-to-Text):** Converts the raw audio into text.
3. **AI Router:** Analyzes the text and determines if it is a direct home automation command or a general question.
4. **Dispatcher:** Routes the request to the corresponding smart home adapter or to the thinking model.
5. **TTS (Text-to-Speech):** Synthesizes the text response back into high-quality audio.

## 📂 Project Structure

```text
├── project/
│   ├── main.py             # FastAPI backend & API endpoints
│   ├── dispatcher.py       # Command routing & adapter execution
│   ├── ai_router.py        # Intent recognition & LLM routing logic
│   ├── adapters/           # Smart home hardware integrations (REST, MQTT, etc.)
│   ├── stt/                # Local Speech-to-Text engines (e.g., Whisper)
│   ├── tts/                # Local Text-to-Speech engines (e.g., Piper)
│   ├── wakeword/           # Wake word detection modules
│   └── tests/              # Benchmark manifests and audio evaluation samples
└── docker-compose.yml      # Multi-container Docker deployment configuration

##🛠️ Target Hardware: Jetson Orin Nano
This specific branch is finely tuned for the NVIDIA Jetson Orin Nano (8 GB VRAM).
It utilizes hardware-accelerated local model execution and shared memory optimizations to achieve low-latency processing without relying on cloud computation.

##🏁 Getting Started
Prerequisites
Ensure your Jetson is running NVIDIA Jetpack with the NVIDIA Container Toolkit installed. 
For a detailed guide on setting up the environment, see project/docs/JETSON_SETUP.md.

Installation
Clone the repository:

git clone 
cd liva

Spin up the local local environment using Docker Compose:

docker compose up --build -d

Access the FastAPI documentation and frontend via your browser to configure your specific models and smart home devices.

##🎛️ Custom Commands & Automations
You can easily define custom voice triggers by editing project/data/custom_commands.json.

Example configurations include:

Controlling lights, switches, and relays.

Toggling custom REST endpoints.

Publishing state changes to MQTT topics.

Triggering local device shell scripts.

##🔌 Extending the System (Adding Adapters)
To add a new smart home device or service integration:

Create a new Python module inside project/adapters/.

Implement your connection logic (e.g., webhooks, custom protocols).

Register the new adapter inside project/dispatcher.py.

Created with 💻 and ☕ by Daniil Agarkov.
