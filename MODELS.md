# Model Setup Guide for LIVA Jetson Deployment

This directory contains scripts to download and manage AI models for the whisper.cpp (STT) and llama.cpp (LLM) services.

## Models Required

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `ggml-base.bin` | Whisper Base | ~147 MB | Speech-to-Text (STT) |
| `deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf` | Deepseek R1 Distill | ~1.2 GB | LLM for complex reasoning (thinking) |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Qwen 2.5 Instruct | ~1.0 GB | LLM for simple queries (non-thinking) |

**Total: ~2.3 GB**

## Quick Start (Recommended)

Run the master setup script to download all models at once:

```bash
chmod +x setup_all_models.sh
./setup_all_models.sh
```

This will:
1. Download whisper base model → `./models/ggml-base.bin`
2. Download thinking LLM → `./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf`
3. Download non-thinking LLM → `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

## Individual Setup (Manual)

### Download Whisper Model Only

```bash
chmod +x setup_models.sh
./setup_models.sh
```

Downloads: `./models/ggml-base.bin`

### Download Llama Models Only

```bash
chmod +x setup_models_llama.sh
./setup_models_llama.sh
```

Downloads:
- `./models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf`
- `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

## After Models Are Downloaded

1. **Start the deployment:**
   ```bash
   docker compose up -d
   ```

2. **Wait for services to initialize** (~30-60 seconds for GPU memory allocation and model loading):
   ```bash
   docker compose logs -f thinking-llm
   ```

   You should see:
   ```
   ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7619 MiB)
   Device 0: Orin, compute capability 8.7
   main: loading model
   ...
   main: Server listening on http://0.0.0.0:8080
   ```

3. **Check all services are healthy:**
   ```bash
   docker compose ps
   docker compose logs my-app | tail -20
   ```

4. **Test the deployment:**
   ```bash
   # Health check
   curl http://localhost:5000/health

   # Test STT (upload audio file)
   curl -X POST -F "audio=@test_audio.wav" http://localhost:5000/api/process-audio

   # Test chat
   curl -X POST -H "Content-Type: application/json" \
        -d '{"message": "Hello"}' \
        http://localhost:5000/api/chat/turn
   ```

## Prerequisites

- **Disk Space**: ~3-4 GB (models + Docker layers)
- **Internet**: 10-20 minutes for downloads at typical speeds
- **On Jetson**: All services will use GPU via NVIDIA Container Runtime

### Optional: Pre-installed Tools

For faster downloads, install `huggingface-hub`:

**On Linux/macOS:**
```bash
pip install huggingface-hub
```

**On Windows (in your project venv):**
```bash
pip install huggingface-hub
```

If not installed, scripts fall back to `wget` (slower but works).

## Troubleshooting
### ZRam deaktivieren (WICHTIG):
- sudo systemctl disable nvzramconfig
- sudo reboot
### Gnome Desktop ausschalten (Headless Mode):
- sudo systemctl set-default multi-user.target
- sudo reboot
# (Um ihn später wieder einzuschalten: sudo systemctl set-default graphical.target)
### Models not found in Docker
- Verify files exist: `ls -lh ./models/`
- Ensure `docker-compose.yml` has `volumes: - ./models:/models` for all services
- Re-run setup script: `./setup_all_models.sh`

### Download fails with "Connection refused"
- Check internet connectivity: `curl https://huggingface.co`
- Try manual download from [HuggingFace](https://huggingface.co) and place in `./models/`

### Jetson out of disk space
- Check: `df -h`
- Clear Docker cache: `docker system prune -a`
- Move `./models` to external storage and create symlink: `ln -s /path/to/models ./models`

### Model loads but inference is slow
- Normal on Jetson with quantized models (4-bit)
- Check GPU utilization: `docker exec thinking-llm nvidia-smi`
- Verify CUDA backend in logs: `docker compose logs thinking-llm | grep "CUDA"`

## Troubleshooting Inside Docker

If a container won't load the model, check:

```bash
# Verify volume mount
docker compose exec thinking-llm ls -lh /models/

# Check container logs
docker compose logs thinking-llm

# Manually test server startup
docker compose exec thinking-llm /usr/local/bin/llama-server \
  -m /models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080
```

## Model Sources

Models are hosted on HuggingFace:

- **Whisper**: [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium)
- **Deepseek R1**: [cjpai/deepseek-r1-distill-qwen-1.5b-gguf](https://huggingface.co/cjpai/deepseek-r1-distill-qwen-1.5b-gguf)
- **Qwen 2.5**: [cjpai/Qwen2.5-1.5B-Instruct-gguf](https://huggingface.co/cjpai/Qwen2.5-1.5B-Instruct-gguf)

## Advanced: Custom Models

To use different models:

1. **Edit compose file:**
   ```yaml
   # docker-compose.yml
   thinking-llm:
     command: >
       /usr/local/bin/llama-server
       -m /models/YOUR_MODEL.gguf
       --host 0.0.0.0 --port 8080
   ```

2. **Download and place in `./models/`:**
   ```bash
   # Example: Use a different Qwen version
   mkdir -p ./models
   wget -O ./models/qwen2-1b.gguf \
     https://huggingface.co/repo/resolve/main/qwen2-1b.gguf
   ```

3. **Restart:**
   ```bash
   docker compose up -d --force-recreate
   ```

## Documentation

- [docker-compose.yml](./docker-compose.yml) - Service configuration
- [build/Dockerfile.llama](./build/Dockerfile.llama) - Llama.cpp build
- [build/Dockerfile.whisper](./build/Dockerfile.whisper) - Whisper.cpp build
- [project/main.py](./project/main.py) - FastAPI routes


## Huggingface
# In dein Projektverzeichnis gehen
cd liva-jetson-ai

# Alte Umgebung löschen
rm -rf .venv

# Neue Umgebung erstellen (jetzt sollte pip automatisch dabei sein)
python3 -m venv .venv

# Aktivieren
source .venv/bin/activate

# Prüfen, ob pip da ist
pip --version
# install
pip install --upgrade pip
pip install "huggingface_hub[cli]"

benchmark_summary_nano_20260527T122223Z.csv
hardware,mode,whisper_model,llm_model,command_id,runs,avg_latency_ms,p95_latency_ms,min_latency_ms,max_latency_ms,avg_cpu_pct,peak_cpu_pct,avg_ram_pct,peak_ram_pct,peak_ram_used_mb,avg_gpu_pct,peak_gpu_pct,peak_gpu_used_mb,error_count
nano,audio,unknown,unknown,factory_check_warehouse,3,1447.87,1454.11,1442.38,1454.98,,,,,,,,,0



cat benchmark_raw_nano_20260527T122223Z.csv
timestamp_utc,hardware,host,os,python,mode,endpoint,base_url,whisper_model,llm_model,command_id,run_index,input_value,audio_file,expected_intent,status_code,latency_ms,cpu_avg_pct,cpu_peak_pct,ram_avg_pct,ram_peak_pct,ram_peak_used_mb,gpu_avg_pct,gpu_peak_pct,gpu_peak_used_mb,response_intent,response_route,response_command_text,response_tts_text,server_dispatch_ms,server_tts_ms,server_total_ms,server_dispatch_offset_ms,error
2026-05-27T12:22:20.596934+00:00,nano,nano,Linux-5.15.148-tegra-aarch64-with-glibc2.35,3.10.12,audio,/api/process-audio,http://127.0.0.1:5000,unknown,unknown,factory_check_warehouse,1,factory_check_warehouse.wav,factory_check_warehouse.wav,Check the warehouse,200,1454.98,,,,,,,,,custom_command,,REST: GET /api/factory/highbay-stock,"High-bay has 3 of 9 slots occupied. Free: 6. Occupied locations: A1, B1, C1",,,,,
2026-05-27T12:22:22.042174+00:00,nano,nano,Linux-5.15.148-tegra-aarch64-with-glibc2.35,3.10.12,audio,/api/process-audio,http://127.0.0.1:5000,unknown,unknown,factory_check_warehouse,2,factory_check_warehouse.wav,factory_check_warehouse.wav,Check the warehouse,200,1442.38,,,,,,,,,custom_command,,REST: GET /api/factory/highbay-stock,"High-bay has 3 of 9 slots occupied. Free: 6. Occupied locations: A1, B1, C1",,,,,
2026-05-27T12:22:23.489351+00:00,nano,nano,Linux-5.15.148-tegra-aarch64-with-glibc2.35,3.10.12,audio,/api/process-audio,http://127.0.0.1:5000,unknown,unknown,factory_check_warehouse,3,factory_check_warehouse.wav,factory_check_warehouse.wav,Check the warehouse,200,1446.24,,,,,,,,,custom_command,,REST: GET /api/factory/highbay-stock,"High-bay has 3 of 9 slots occupied. Free: 6. Occupied locations: A1, B1, C1",,,,,