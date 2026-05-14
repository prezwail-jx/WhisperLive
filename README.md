# WhisperLive

<h2 align="center">
  <a href="https://www.youtube.com/watch?v=0PHWCApIcCI"><img
src="https://img.youtube.com/vi/0PHWCApIcCI/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperLive"></a>
  <a href="https://www.youtube.com/watch?v=0f5oiG4oPWQ"><img
  src="https://img.youtube.com/vi/0f5oiG4oPWQ/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperLive"></a>
  <br><br>A nearly-live implementation of OpenAI's Whisper.
<br><br>
</h2>

This project is a real-time transcription application that uses the OpenAI Whisper model
to convert speech input into text output. It can be used to transcribe both live audio
input from microphone and pre-recorded audio files.

## 中文快速导航（新增）

本节是中文重排版，目标是让你按顺序完成：安装 -> 启动服务 -> 连接客户端。
你要求的「现有内容不删除」已经遵守：本文件后半部分保留了原始英文章节与全部命令。

- [安装教程（放最前）](#安装教程放最前)
- [拉服务使用教程（按平台）](#拉服务使用教程按平台)
- [会议同传 Web 前端](#会议同传-web-前端)
- [其他内容](#其他内容)
- [原始英文内容（完整保留）](#原始英文内容完整保留)

## 安装教程（放最前）

### 1. 通用前置依赖

```bash
bash scripts/setup.sh
```

```bash
pip install whisper-live
```

### 2. macOS 安装

`scripts/setup.sh` 在 macOS 上会使用 Homebrew 安装 `portaudio` 和 `wget`。

```bash
bash scripts/setup.sh
python3 -m venv .venv
source .venv/bin/activate
pip install whisper-live
```

### 3. Windows 安装

Windows 建议手动安装 Python 3.10+ 与 PortAudio 依赖后再安装包。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install whisper-live
```

如果你在 Windows 上使用原生命令行遇到音频依赖问题，建议优先使用 WSL2 或 Docker 方式运行服务。

### 4. Linux 安装

Debian/Ubuntu、Fedora 均可。`scripts/setup.sh` 会自动识别并安装系统依赖。

```bash
bash scripts/setup.sh
python3 -m venv .venv
source .venv/bin/activate
pip install whisper-live
```

Fedora 的 Python 3.12 虚拟环境（原文命令，保留）：

```bash
sudo dnf install -y python3.12 python3.12-pip
python3.12 -m venv whisper_env
source whisper_env/bin/activate
```

## 拉服务使用教程（按平台）

### macOS

服务端（MLX，本地模型路径或 HF 仓库都可以）：

```bash
python3 run_server.py --port 9090 \
                      --backend mlx_whisper \
                      --max_clients 4 \
                      --max_connection_time 600
```

客户端（转写音频文件，不外放声音；这里传你本地的 MLX 模型目录）：

```bash
python3 run_client.py --server localhost --port 9090 \
                      --files <audio-file-name> \
                      --model /Users/xuan/Documents/program/Translate/model/whisper-medium-mlx \
                      --lang en \
                      --mute_audio_playback
```

OpenAI REST 接口（如果你要保留 REST 模式，也建议先用 MLX 后端）：

```bash
python3 run_server.py --port 9090 --backend mlx_whisper --max_clients 4 --max_connection_time 600 --enable_rest --cors-origins="http://localhost:8080,http://127.0.0.1:8080"
python3 client_openai.py $AUDIO_FILE
```

### Windows

服务端（faster_whisper）：

```bash
python run_server.py --port 9090 --backend faster_whisper --max_clients 4 --max_connection_time 600
```

客户端：

```bash
python run_client.py --files <audio-file-name> --server localhost --port 9090
```

说明：Windows 下路径建议使用绝对路径，音频驱动或编译依赖异常时建议改用 Docker 或 WSL2。

### Linux

服务端（faster_whisper）：

```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600
```

控制 OpenMP 线程（原文命令，保留）：

```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --omp_num_threads 4
```

### Docker（推荐用于跨平台部署）

GPU + Faster-Whisper：

```bash
docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
```

CPU + Faster-Whisper：

```bash
docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
```

OpenVINO：

```bash
docker run -it --device=/dev/dri -p 9090:9090 ghcr.io/collabora/whisperlive-openvino
```

## 会议同传 Web 前端

当前项目新增了一个轻量 Web 前端，目录是 `web/`。它用于会议同传场景：

```text
浏览器采集当前电脑麦克风
-> WebSocket 发送到 WhisperLive server
-> server 做语音识别和中英自动互译
-> 网页刷新原文和翻译
```

这时不需要运行 `run_client.py`。`run_client.py` 是命令行客户端，会使用它所在机器的麦克风；Web 前端会使用打开网页那台电脑的麦克风。

### 服务端：跑模型的机器

在 WhisperLive 项目根目录启动后端：

```bash
source venv/bin/activate

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 4 \
  --max_connection_time 600
```

多 Web client 并发测试时建议开启批处理推理，并指定一个本地 ASR 模型作为共享模型：

```bash
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 4 \
  --max_connection_time 600 \
  --batch_inference \
  --batch_max_size 4 \
  --batch_window_ms 100 \
  -fw model/asr/base
```

说明：

- `--batch_inference` 用于多 client 并发时把请求合并成批处理，提升 GPU 利用率。
- `-fw model/asr/base` 会强制所有 client 使用同一个本地模型，批处理模式下更稳定。
- 如果要测更大模型，把 `model/asr/base` 改成 `model/asr/small` 或 `model/asr/large-v3-turbo`。
- `--max_clients` 控制允许连接的 Web client 数量；第 `max_clients + 1` 个 client 会进入等待。

生产环境如果只是想快速拉起固定的多 GPU 服务，可以使用快捷启动脚本：

```bash
scripts/start_prod_gpu.sh start
```
等价于：

```bash

  CUDA_VISIBLE_DEVICES=0 python3 run_server.py \
    --port 9090 \
    --backend faster_whisper \
    --max_clients 12 \
    --max_connection_time 600 \
    --batch_inference \
    --batch_max_size 8 \
    --batch_window_ms 50 \
    -fw model/asr/small


  CUDA_VISIBLE_DEVICES=1 python3 run_server.py \
    --port 9091 \
    --backend faster_whisper \
    --max_clients 12 \
    --max_connection_time 600 \
    --batch_inference \
    --batch_max_size 8 \
    --batch_window_ms 50 \
    -fw model/asr/small
```

常用管理命令：

```bash
scripts/start_prod_gpu.sh status
scripts/start_prod_gpu.sh stop
scripts/start_prod_gpu.sh restart
```

`scripts/start_prod_gpu.sh` 只是对 `run_server.py` 的快捷封装，不是新的服务入口。默认会启动两路服务：

- `CUDA_VISIBLE_DEVICES=0` -> `9090`
- `CUDA_VISIBLE_DEVICES=1` -> `9091`

两路默认都使用 `--backend faster_whisper`、`-fw model/asr/small`、`--max_clients 12` 和 `--batch_inference`。客户端需要自行连接 `ws://host:9090` 或 `ws://host:9091` 做分流。

如果要修改 GPU、端口、模型、并发数或 batch 参数，直接编辑 `scripts/start_prod_gpu.sh` 顶部变量：

```bash
GPUS=(0 1)
PORTS=(9090 9091)
MODEL_PATH="model/asr/small"
MAX_CLIENTS=12
MAX_CONNECTION_TIME=600
BATCH_MAX_SIZE=8
BATCH_WINDOW_MS=50
```

服务端对外提供的核心端口是：

```text
9090/TCP
```

这是 WebSocket 后端端口，前端页面的 `Server` 输入框会连接它。

### 客户端：打开网页的机器

如果客户端机器有 `web/` 目录，在客户端机器上启动静态页面：

```bash
cd web
python3 -m http.server 8080
```

浏览器打开：

```text
http://localhost:8080
```

页面里的 `Server` 填后端地址。

同一台机器测试：

```text
ws://localhost:9090
```

同一局域网内，假设后端机器 IP 是 `192.xxx.xx.x`：

```text
ws://192.xxx.xx.x
```

然后点击“开始”，允许浏览器麦克风权限。

### 异地访问：SSH 隧道推荐

如果服务端在家里主机，客户端在手边电脑，且已经有公网 SSH 入口，例如：

```text
xxxx@ub.xxxxxx.com -p 2233
```

推荐在客户端机器上开 SSH 隧道：

```bash
ssh -p 2233 -L 9090:localhost:9090 xxxx@ub.xxxxxx.com
```

然后客户端本地启动 Web 页面：

```bash
cd web
python3 -m http.server 8080
```

浏览器打开：

```text
http://localhost:8080
```

页面里的 `Server` 填：

```text
ws://localhost:9090
```

如果 Web 页面也跑在服务器主机，可以同时转发 `8080`：

```bash
ssh -p 2233 \
  -L 9090:localhost:9090 \
  -L 8080:localhost:8080 \
  xxxx@ub.xxxxxx.com
```

然后浏览器打开 `http://localhost:8080`。

### 路由器端口映射

临时公网测试时需要映射：

```text
外部 9090/TCP -> 后端机器内网IP:9090
```

如果 Web 页面也在服务端机器上对外提供，再映射：

```text
外部 8080/TCP -> 后端机器内网IP:8080
```

注意：当前 WebSocket 服务没有鉴权，不建议长期直接暴露 `9090` 到公网。生产使用建议用 Tailscale、VPN，或 HTTPS/WSS 反向代理加认证。

### 中英自动互译模型

当前 Web 前端默认开启：

```text
enable_translation = true
target_language = auto
translation_provider = helsinki_zh_en
```

默认本地翻译模型路径：

```text
model/opus-mt-zh-en
model/opus-mt-en-zh
```

后端会根据识别语言自动选择方向：

```text
中文 -> 英文
英文 -> 中文
```

当前实现按本次会话识别出的主语言选择翻译方向。如果同一场会议里频繁中英文混说，后续可以增加 segment 级语言检测来逐句切换翻译方向。

### 会议日志导出与 Word 转换

Web 前端会保留完整会议日志，页面仍然只显示最近 N 段字幕。会议结束后点击页面上的“导出日志”，会下载一个 JSON 文件，里面包含原文和翻译：

```text
source_segments
translation_segments
```

如果要把 JSON 转成 Word：

```bash
pip install -r requirements/docx.txt
python scripts/meeting_log_to_docx.py meeting-log.json --output meeting.docx
```

说明：

- “清空”按钮只清空页面显示。
- “清空日志”会清空完整会议记录。
- 刷新页面会丢失浏览器内存中的会议日志，导出前不要刷新。
- 需要把文件json路径改到项目目录并改名为meeting-log.json。

### 自动下载模型

项目提供统一下载脚本：

```bash
pip install -r requirements/model_download.txt
```

默认下载翻译模型到当前代码使用的目录：

```bash
python scripts/download_models.py --translation
```

下载 ASR 模型：

```bash
python scripts/download_models.py --model large-v3-turbo
```

下载全部内置模型：

```bash
python scripts/download_models.py --all
```

默认目录结构：

```text
model/asr/tiny
model/asr/tiny.en
model/asr/base
model/asr/base.en
model/asr/small
model/asr/small.en
model/asr/medium
model/asr/medium.en
model/asr/large-v3-turbo
model/asr/large-v3
model/opus-mt-zh-en
model/opus-mt-en-zh
```

脚本默认使用 `--source auto`：先尝试魔搭 ModelScope，如果没有配置对应魔搭模型 ID 或下载失败，再使用 Hugging Face 兜底。只使用 Hugging Face：

```bash
python scripts/download_models.py --all --source huggingface
```

只使用魔搭：

```bash
python scripts/download_models.py --all --source modelscope
```

使用：

```bash
python scripts/download_models.py --all --manifest modelscope_models.json
```

Web 前端会把模型选项自动映射到本地 ASR 目录，例如：

```text
large-v3-turbo -> model/asr/large-v3-turbo
```

因此预下载后，启动服务时不需要再临时下载 ASR 模型。

TensorRT（原文命令，保留）：

```bash
docker build . -f docker/Dockerfile.tensorrt -t whisperlive-tensorrt
docker run -p 9090:9090 --runtime=nvidia --entrypoint /bin/bash -it whisperlive-tensorrt

# Build small.en engine
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int8
bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int4
```

## 其他内容

### 支持的服务后端

- `faster_whisper`
- `tensorrt`
- `openvino`
- `mlx_whisper`

### 预下载模型

如果你想在启动服务前预先下载模型（避免首次启动时自动下载），可以使用 `scripts/download_models.py` 脚本：

**下载所有 ASR 模型和翻译模型：**
```bash
python scripts/download_models.py --all
```

**只下载 ASR 模型：**
```bash
python scripts/download_models.py --asr
```

**只下载翻译模型：**
```bash
python scripts/download_models.py --translation
```

**下载指定模型（可多次使用 --model）：**
```bash
python scripts/download_models.py --model tiny --model base --model small
```

**使用自定义模型目录：**
```bash
python scripts/download_models.py --all --model-dir /path/to/models
```

**从指定源下载（auto 会先尝试 ModelScope，再尝试 Hugging Face）：**
```bash
python scripts/download_models.py --asr --source huggingface
```

**支持的模型列表（ASR）：**
- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v3`, `large-v3-turbo`

**支持的翻译模型：**
- `opus-mt-zh-en` (中文 → 英文)
- `opus-mt-en-zh` (英文 → 中文)

### 自定义模型与缓存目录

自定义 faster_whisper 模型与 cache 目录（原文命令，保留）：

```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600 \
                      -fw "/path/to/custom/faster/whisper/model" \
                      -c ~/.cache/whisper-live/
```

### Single model mode（原说明保留）

当不显式指定模型时，服务默认可能会按连接实例化模型；使用 `-trt` 或 `-fw` 自定义模型时通常会复用单模型实例。若不希望如此，可设置 `--no_single_model`。

### 浏览器扩展与 iOS 客户端

- Chrome / Firefox 扩展：见原文 Browser Extensions 部分
- iOS 客户端：见 `Audio-Transcription-iOS/README.md`

### TensorRT 配置说明

请参考 `TensorRT_whisper.md`。

## 原始英文内容（完整保留）

以下为仓库原始 README 内容，保持完整，便于与上面的中文重排版对照使用。

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Running the Server](#running-the-server)
- [Running the Client](#running-the-client)
- [Browser Extensions](#browser-extensions)
- [Whisper Live Server in Docker](#whisper-live-server-in-docker)
- [Future Work](#future-work)
- [Blog Posts](#blog-posts)
- [Contact](#contact)
- [Citations](#citations)

## Installation
- Install PortAudio
```bash
 bash scripts/setup.sh
```

- Install whisper-live from pip
```bash
 pip install whisper-live
```


- Install 3.12 venv on Fedora

```bash
sudo dnf install -y python3.12 python3.12-pip
python3.12 -m venv whisper_env
source whisper_env/bin/activate
```


### OpenAI REST interface

#### Server

```bash
python3 run_server.py --port 9090 --backend faster_whisper --max_clients 4 --max_connection_time 600 --enable_rest --cors-origins="http://localhost:8080,http://127.0.0.1:8080"
```

#### Client

```bash
python3 client_openai.py $AUDIO_FILE
```



### Setting up NVIDIA/TensorRT-LLM for TensorRT backend
- Please follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and for building Whisper-TensorRT engine.

## Getting Started
The server supports 3 backends `faster_whisper`, `tensorrt` and `openvino`. If running `tensorrt` backend follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md)

### Running the Server
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) backend
```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600
  
# running with custom model and cache_dir to save auto-converted ctranslate2 models
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --max_clients 4 \
                      --max_connection_time 600 \
                      -fw "/path/to/custom/faster/whisper/model" \
                      -c ~/.cache/whisper-live/
```

- TensorRT backend. Currently, we recommend to only use the docker setup for TensorRT. Follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) which works as expected. Make sure to build your TensorRT Engines before running the server with TensorRT backend.
```bash
# Run English only model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small_en \
                      --max_clients 4 \
                      --max_connection_time 600

# Run Multilingual model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small \
                      -m \
                      --max_clients 4 \
                      --max_connection_time 600
```
- Use `--max_clients` option to restrict the number of clients the server should allow. Defaults to 4.
- Use `--max_connection_time` options to limit connection time for a client in seconds. Defaults to 600.
- WhisperLive now supports the [OpenVINO](https://github.com/openvinotoolkit/openvino) backend for efficient inference on Intel CPUs, iGPU and dGPUs. Currently, we tested the models uploaded to [huggingface by OpenVINO](https://huggingface.co/OpenVINO?search_models=whisper).
  - > **Docker Recommended:** Running WhisperLive with OpenVINO inside Docker automatically enables GPU support (iGPU/dGPU) without requiring additional host setup.
  - > **Native (non-Docker) Use:** If you prefer running outside Docker, ensure the Intel drivers and OpenVINO runtime are installed and properly configured on your system. Refer to the documentation for [installing OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=PIP#).

```
python3 run_server.py -p 9090 -b openvino
```


#### Controlling OpenMP Threads
To control the number of threads used by OpenMP, you can set the `OMP_NUM_THREADS` environment variable. This is useful for managing CPU resources and ensuring consistent performance. If not specified, `OMP_NUM_THREADS` is set to `1` by default. You can change this by using the `--omp_num_threads` argument:
```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --omp_num_threads 4
```

#### Single model mode
By default, when running the server without specifying a model, the server will instantiate a new whisper model for every client connection. This has the advantage, that the server can use different model sizes, based on the client's requested model size. On the other hand, it also means you have to wait for the model to be loaded upon client connection and you will have increased (V)RAM usage.

When serving a custom TensorRT model using the `-trt` or a custom faster_whisper model using the `-fw` option, the server will instead only instantiate the custom model once and then reuse it for all client connections.

If you don't want this, set `--no_single_model`.


### Running the Client

Use the below command to run the client:
```bash
python3 run_client.py --files <audio-file-name>
```
This will connect to the localhost server running on port 9090 by default. Use flags `--server` and `--port` to use different configurations. The above command will transcribe audio file provided with `--files` flag.


Here are the details of client instance implemented in `run_client.py` script:
  - `lang`: Language of the input audio, applicable only if using a multilingual model.
  - `translate`: If set to `True` then translate from any language to `en`.
  - `model`: Whisper model path or size. This fork defaults to local `model/asr/small` when available.
  - `use_vad`: Whether to use `Voice Activity Detection` on the server.
  - `save_output_recording`: Set to True to save the microphone input as a `.wav` file during live transcription. This option is helpful for recording sessions for later playback or analysis. Defaults to `False`. 
  - `output_recording_filename`: Specifies the `.wav` file path where the microphone input will be saved if `save_output_recording` is set to `True`.
  - `mute_audio_playback`: Whether to mute audio playback when transcribing an audio file. Defaults to False.
  - `enable_translation`: Start translation thread on the server (from any to any).
  - `target_language`: Server translation thread's target translation language.

```python
from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="model/asr/small",                            # also supports hf_model => `Systran/faster-whisper-small`
  use_vad=False,
  save_output_recording=True,                         # Only used for microphone input, False by Default
  output_recording_filename="./output_recording.wav", # Only used for microphone input
  mute_audio_playback=False,                          # Only used for file input, False by Default
  enable_translation=True,
  target_language="hi",
)
```
It connects to the server running on localhost at port 9090. Using a multilingual model, language for the transcription will be automatically detected. You can also use the language option to specify the target language for the transcription, in this case, English ("en"). The translate option should be set to `True` if we want to translate from the source language to English and `False` if we want to transcribe in the source language.

- Transcribe an audio file:
```python
client("tests/jfk.wav")
```

- To transcribe from microphone:
```python
client()
```

- To transcribe from a RTSP stream:
```python
client(rtsp_url="rtsp://admin:admin@192.168.0.1/rtsp")
```

- To transcribe from a HLS stream:
```python
client(hls_url="http://as-hls-ww-live.akamaized.net/pool_904/live/ww/bbc_1xtra/bbc_1xtra.isml/bbc_1xtra-audio%3d96000.norewind.m3u8")
```

## Browser Extensions
- Run the server with your desired backend as shown [here](https://github.com/collabora/WhisperLive?tab=readme-ov-file#running-the-server).
- Transcribe audio directly from your browser using our Chrome or Firefox extensions. Refer to [Audio-Transcription-Chrome](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme) and https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md

## iOS Client

Use WhisperLive on iOS with our native iOS client.  
Refer to [`ios-client`](https://github.com/collabora/WhisperLive/tree/main/Audio-Transcription-iOS) and [`ios-client/README.md`](https://github.com/collabora/WhisperLive/blob/main/Audio-Transcription-iOS/README.md) for setup and usage instructions.


## Whisper Live Server in Docker
- GPU
  - Faster-Whisper
  ```bash
  docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
  ```

  - TensorRT. Refer to [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup and more tensorrt backend configurations.
  ```bash
  docker build . -f docker/Dockerfile.tensorrt -t whisperlive-tensorrt
  docker run -p 9090:9090 --runtime=nvidia --entrypoint /bin/bash -it whisperlive-tensorrt

  # Build small.en engine
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en        # float16
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int8   # int8 weight only quantization
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en int4   # int4 weight only quantization

  # Run server with small.en
  python3 run_server.py --port 9090 \
                        --backend tensorrt \
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_float16"
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_int8"
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en_int4"
  ```

  - OpenVINO
  ```
  docker run -it --device=/dev/dri -p 9090:9090 ghcr.io/collabora/whisperlive-openvino
  ```

- CPU
  - Faster-whisper
  ```bash
  docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
  ```

## Future Work
- [x] Add translation to other languages on top of transcription.

## Blog Posts
- [Transforming speech technology with WhisperLive](https://www.collabora.com/news-and-blog/blog/2024/05/28/transforming-speech-technology-with-whisperlive/)
- [WhisperFusion: Ultra-low latency conversations with an AI chatbot](https://www.collabora.com/news-and-blog/news-and-events/whisperfusion-ultra-low-latency-conversations-with-an-ai-chatbot.html) powered by WhisperLive
- [Breaking language barriers 2.0: Moving closer towards fully reliable, production-ready Hindi ASR](https://www.collabora.com/news-and-blog/news-and-events/breaking-language-barriers-20-moving-closer-production-ready-hindi-asr.html) which is used in WhisperLive for hindi.

## Contact

We are available to help you with both Open Source and proprietary AI projects. You can reach us via the Collabora website or [vineet.suryan@collabora.com](mailto:vineet.suryan@collabora.com) and [marcus.edel@collabora.com](mailto:marcus.edel@collabora.com).


## Citations
```bibtex
@article{Whisper
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  publisher = {arXiv},
  year = {2022},
}
```

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  email = {hello@silero.ai}
}
