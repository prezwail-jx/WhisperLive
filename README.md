# WhisperLive 部署与使用说明

## 外部文档

以下两份文档覆盖生产环境使用和运维：

- [普通用户使用指导](./whisperlive-user-guide.md)：会议使用者入口、WebSocket 填写、热词上传、显示模式、日志导出和总结功能。
- [运维使用指导](./whisperlive-ops-guide.md)：单卡/双卡模式、容器拉起、Nginx 模式切换、WebSocket 路由和 Admin API。

下面内容为项目通用部署说明和功能参考。

## 项目结构

```
whisper_live/server.py         WebSocket、Admin API、客户端管理与服务编排
whisper_live/meeting/           会议热词、日志、总结模板和 LLM 总结
whisper_live/backend/           ASR 与翻译后端
whisper_live/batch_inference.py  批量 GPU 推理调度
whisper_live/vad.py             Silero VAD 语音活动检测
whisper_live/diarization.py     说话人分离
web/                            浏览器前端
scripts/                        启动脚本、压测、Nginx 切换
deploy/nginx/                   Nginx 配置
```

## 端口规划

| 端口 | 用途 |
|------|------|
| 9090 | 后端 WebSocket ASR 服务 |
| 9093 | 浏览器统一入口（前端页面、/ws、/admin/） |
| 9094 | 后端 Admin API（映射到容器内 8000） |

浏览器只访问 `9093`，Nginx 自动将 `/ws` 转到 `9090`、`/admin/` 转到 `9094`。

---

## 1. 快速启动（docker-compose，推荐）

前置条件：已构建 `whisperlive-server:docx` 镜像，已创建 `whisperlive-net` 网络，`model/asr/large-v3-turbo` 模型已下载。

```bash
docker compose -f docker-compose.local.yml up -d
```

这将同时启动：
- **whisperlive-gpu0**：GPU 0 上的 ASR 服务（Whisper），自动执行 `scripts/start_whisper_service.sh`
- **whisperlive-web-gateway**：Nginx 统一入口

访问：
- 前端页面：`http://localhost:9093/`
- 中控页面：`http://localhost:9093/admin.html`
- Admin API：`http://localhost:9094/admin/clients`

### compose 配置说明

```yaml
# 单卡模式（GPU 0），完整挂载项目目录
whisperlive-gpu0:
  image: whisperlive-server:docx
  ports:
    - "9090:9090"    # WebSocket
    - "9094:8000"    # Admin API
  volumes:
    - .:/app
  command: bash -lc "./scripts/start_whisper_service.sh"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]

# Nginx 统一入口
whisperlive-web-gateway:
  image: nginx:alpine
  ports:
    - "9093:80"
  volumes:
    - ./web:/usr/share/nginx/html:ro
    - ./deploy/nginx/whisperlive.conf:/etc/nginx/conf.d/default.conf:ro
```

如需双卡，可在 compose 中增加一个 `whisperlive-gpu1` 服务、改用 `device_ids: ["1"]`，并切换到双卡 Nginx 配置：

```bash
./scripts/switch_nginx_mode.sh dual
```

---

## 2. 构建镜像

在项目根目录执行：

```bash
docker build --network=host -f docker/Dockerfile.server -t whisperlive-server:docx .
```

### 可用 Dockerfile

| Dockerfile | 用途 |
|------------|------|
| `docker/Dockerfile.server` | GPU 生产镜像（CUDA 12.4），含 ASR + 翻译 + 总结 + DOCX |
| `docker/Dockerfile.gpu` | 精简 GPU 镜像 |
| `docker/Dockerfile.cpu` | CPU-only 镜像 |
| `docker/Dockerfile.tensorrt` | TensorRT 后端 |
| `docker/Dockerfile.openvino` | OpenVINO 后端 |
| `docker/Dockerfile.client` | 纯 Web 前端静态页面 |

### 创建 Docker 网络

```bash
docker network create --subnet 172.30.0.0/24 whisperlive-net
```

如果网络已存在（`docker network ls | grep whisperlive-net`），可跳过。

---

## 3. 手动启动方式

### 3.1 Faster-Whisper（推荐）

```bash
# 1. 拉起容器
docker run --rm -it --gpus '"device=0"' \
  --name whisperlive-gpu0 \
  --network whisperlive-net \
  -p 9090:9090 -p 9094:8000 \
  -v "$PWD:/app" -w /app \
  whisperlive-server:docx bash

# 2. 容器内启动服务
./scripts/start_whisper_service.sh
```

等效的手动命令：

```bash
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 60000 \
  --batch_inference --batch_max_size 1 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --cors-origins http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/large-v3-turbo
```

### 3.2 FunASR（Paraformer 流式 + SenseVoice 精修）

需要提前准备模型：

```
model/funasr/paraformer-zh-streaming
model/funasr/SenseVoiceSmall
model/funasr/ct-punc
model/vad/silero_vad.onnx
```

启动（容器内执行 `./scripts/start_funasr_service.sh` 或手动）：

```bash
python run_server.py \
  --port 9090 \
  --backend funasr \
  --funasr_mode paraformer_streaming \
  --funasr_model model/funasr/paraformer-zh-streaming \
  --funasr_final_model model/funasr/SenseVoiceSmall \
  --funasr_punc_model model/funasr/ct-punc \
  --funasr_device cuda \
  --max_clients 12 \
  --max_connection_time 60000 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --cors-origins http://localhost:9093,http://127.0.0.1:9093
```

FunASR 关键参数：

| 参数 | 说明 |
|------|------|
| `--funasr_mode paraformer_streaming` | Paraformer 流式识别，持续刷新字幕 |
| `--funasr_model` | 流式识别模型路径 |
| `--funasr_final_model` | 断句后用完整语音离线精修，提高 final 文本质量 |
| `--funasr_punc_model` | 标点恢复模型 |
| `--funasr_device cuda` | 主识别模型设备 |
| `--funasr_final_device` | 精修模型设备，默认跟随 `--funasr_device` |
| `--disable_funasr_final_refine` | 关闭精修，降低 final 延迟 |

---

## 4. 翻译功能

### 4.1 翻译模型选择

前端支持三种翻译引擎：

| 模型 | 标识 | 说明 |
|------|------|------|
| Helsinki zh-en | `helsinki_zh_en` | 轻量级，实时性好，中英单向 |
| NLLB 600M | `nllb_200_600m` | 多语言，翻译质量更高 |
| NLLB 1.3B | `nllb_200_distilled_1_3b` | 蒸馏版，质量最佳，显存需求更大 |

NLLB 模型路径默认为 `model/NLLB-200-600M`，可在客户端 config 中通过 `nllb_model_path` 指定自定义路径。

服务端通过 `--translation_device` 控制翻译模型设备（`cpu` / `cuda` / `auto`）：

```bash
--translation_device cpu    # 翻译走 CPU，GPU 留给 ASR（推荐）
--translation_device cuda   # 翻译走 GPU
```

### 4.2 翻译方向模式

**手动指定模式**（`specified`）：用户明确选择源语言和目标语言，单向翻译。

**面对面模式**（`interpretation`）：自动识别输入语言并互译（中↔英），适合中英文混合发言。开启后源语言选择会被锁定，系统自动处理翻译方向。

### 4.3 NLLB 残留字符重试

NLLB 模型从中文翻译到英文时，如果译文中仍残留 CJK 字符（比例超过阈值），会自动触发重试，减少翻译中的中文残句。该逻辑仅在 NLLB 模型且源语言为 `zh` 时生效。

### 4.4 翻译警告标记

翻译出错时前端会在对应片段显示 `!` 警告标记（通过 `（翻译出错）` 后缀检测）。

---

## 5. 批量推理（Batch Inference）

通过 `--batch_inference` 启用。多客户端并发时，由 `BatchInferenceWorker` 统一收集请求后批量提交 GPU 推理，减少模型实例化开销和锁竞争。

```bash
--batch_inference          # 启用批量推理
--batch_max_size 8         # 最大 batch 大小（默认 8）
--batch_window_ms 50       # 等待窗口（毫秒）
```

当 `batch_max_size=1` 时，退化为逐条串行推理，但依然共享模型实例，消除多 session 之间的 `SINGLE_MODEL_LOCK` 竞争。

---

## 6. VAD（语音活动检测）

使用 Silero VAD（ONNX Runtime），自动下载到 `model/vad/silero_vad.onnx`。在 faster-whisper 后端中，VAD 用于过滤静音段、减少幻觉。

前端默认配置：

```json
{
  "use_vad": true,
  "vad_parameters": {
    "threshold": 0.5,
    "min_silence_duration_ms": 900,
    "speech_pad_ms": 300
  }
}
```

可通过 client config 自定义 `vad_parameters` 或关闭 `use_vad`。FunASR 后端亦独立使用 VAD 进行语音段分割。

---

## 7. 说话人分离（Diarization）

通过 `--enable_diarization`（客户端 config）启用，依赖 `pyannote.audio` 和 `wespeaker-voxceleb-resnet34-LM` 嵌入模型。

对已完成的语音片段进行说话人识别，基于余弦相似度在线聚类：
- 相似度阈值：`0.55`（可通过 `diarization_threshold` 调整）
- 最大说话人数：`10`

前端设置中勾选"启用说话人识别"即可。该功能目前为轻量级在线分离，后续可接入独立说话人模型，复用现有 `speaker_id` 和校对数据。

---

## 8. 会议热词

热词文件放在 `config/hotwords.d/`，文件名去掉 `.txt` 即为会议号：

```
config/hotwords.d/产品周会.txt  -> 会议号：产品周会
config/hotwords.d/meeting-a.txt -> 会议号：meeting-a
```

文件格式：

```
图灵科技
faster-whisper
张三
# 注释行（以 # 开头）
OpenAI => 开放人工智能    # 固定翻译
```

- 普通行：只作为 ASR 热词
- `source => target`：同时加入 ASR 热词和固定翻译表
- 固定翻译是单向规则；需要反向翻译时应增加反向条目
- 多条规则匹配时优先最长词组

使用规则：
- 启动参数：`--meeting_hotwords_dir config/hotwords.d`
- Admin 页面负责刷新、下拉和预览，不上传/删除文件
- Client 点击开始时读取热词快照并锁定本次会话
- 开始后修改文件只影响下次开始
- 没有对应会议号 txt 时，使用 `--hotwords_file` 的全局默认热词
- 新增或修改热词后无需重启服务，在页面点击刷新即可

---

## 9. 会议日志

点击"导出日志"时：
- 浏览器本地下载 `会议号-*.json`
- 通过 Admin API 保存到服务端 `logs/` 目录

服务端保存目录通过 `--meeting_logs_dir logs` 指定。容器挂载 `-v "$PWD:/app"` 时，日志落在宿主机项目目录下。

文件命名示例：`logs/产品周会-2026-05-28T10-30-15.json`

---

## 10. 会议纪要模板

Web 设置面板支持上传 UTF-8 编码的 `.md` 模板（最多 2 MB）。

流程：
1. 停止会议
2. 在"日志与总结"中上传 `.md` 文件并点击"分析模板"
3. 调整字段名称、类型、说明后保存
4. 在总结模板下拉框的"自定义模板"分组中选择模板并生成总结

支持的字段类型：`文本`、`列表`、`带证据列表`、`表格`。

默认模板库目录通过启动参数指定：

```bash
--summary_templates_dir config/summary_templates
```

---

## 11. 会后转写校对与说话人管理

会议结束后在"会议总结"面板操作：
- 修改识别原文，系统保留 `original_text`
- 新增、重命名、合并说话人，给片段分配说话人
- 修改原文后译文和总结标记为过期；重新生成总结保留旧版本
- 多页面编辑使用修订号（乐观锁）防止覆盖冲突

---

## 12. Nginx 统一入口

docker-compose 已包含 Nginx 网关。手动启动方式：

```bash
docker run --rm -it \
  --name whisperlive-web-gateway \
  --network whisperlive-net \
  --add-host=host.docker.internal:host-gateway \
  -p 9093:80 \
  -v "$PWD/web:/usr/share/nginx/html:ro" \
  -v "$PWD/deploy/nginx/whisperlive.conf:/etc/nginx/conf.d/default.conf:ro" \
  nginx:alpine
```

Nginx 路由规则：

| 路径 | 目标 |
|------|------|
| `/` | 前端页面 |
| `/admin.html` | 中控页面 |
| `/ws` | WebSocket → 后端 9090 |
| `/admin/` | Admin API → 后端 9094（容器内 8000） |

单卡/双卡 Nginx 配置切换：

```bash
./scripts/switch_nginx_mode.sh single
./scripts/switch_nginx_mode.sh dual
```

---

## 13. 命令行 client

在后端容器内执行：

```bash
docker exec -it whisperlive-gpu0 bash
```

中文音频 + 翻译：

```bash
python run_client.py \
  --server 127.0.0.1 --port 9090 \
  --files /app/test_zn.wav --lang zh \
  --enable_translation --target_language en \
  --same_output_threshold 2 --mute_audio_playback
```

英文音频 + 翻译：

```bash
python run_client.py \
  --server 127.0.0.1 --port 9090 \
  --files /app/test_en.wav --lang en \
  --enable_translation --target_language zh \
  --same_output_threshold 2 --mute_audio_playback
```

注意：服务端 `-fw` 决定了实际 ASR 模型，client 传的 `--model` 不覆盖服务端。

---

## 14. 压测

在后端容器内执行：

```bash
docker exec -it whisperlive-gpu0 bash
```

```bash
# 两路中文 + 翻译
python /app/scripts/stress_ws.py \
  --host 127.0.0.1 --port 9090 \
  --audio /app/test_zn.wav --clients 2 \
  --language zh --target_language en \
  --enable_translation --same_output_threshold 2

# 两路英文 + 翻译
python /app/scripts/stress_ws.py \
  --host 127.0.0.1 --port 9090 \
  --audio /app/test_en.wav --clients 2 \
  --language en --target_language zh \
  --enable_translation --same_output_threshold 2
```

日志输出到 `scripts/stress_logs/`。关键字段：
- `success`：是否通过
- `rt_factor`：总耗时/音频时长，越接近 1.0 越实时
- `segments` / `translations`：消息数
- `errors` / `timeout`：连接错误或超时

---

## 15. 启动参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `9090` | WebSocket 端口 |
| `--backend` | `faster_whisper` | 后端类型：`faster_whisper` / `funasr` / `tensorrt` / `openvino` |
| `-fw` | - | Faster-Whisper 模型路径 |
| `--max_clients` | `4` | 最大并发客户端数 |
| `--max_connection_time` | `300` | 最长连接时间（秒） |
| `--batch_inference` | `false` | 启用批量 GPU 推理 |
| `--batch_max_size` | `8` | 批量推理最大 batch |
| `--translation_device` | `cpu` | 翻译模型设备：`cpu` / `cuda` / `auto` |
| `--rest_port` | `8000` | Admin API 端口（容器内） |
| `--cors-origins` | - | CORS 域名（逗号分隔） |
| `--meeting_hotwords_dir` | `config/hotwords.d` | 会议热词目录 |
| `--meeting_logs_dir` | `logs` | 日志保存目录 |
| `--summary_base_url` | `http://127.0.0.1:8001/v1` | LLM 总结 API |
| `--summary_model` | `qwen3-4b-awq` | 总结模型名 |
| `--summary_templates_dir` | `config/summary_templates` | 总结模板目录 |

---

## 16. 常见问题

### 改了代码后需要重新 build 镜像吗

容器使用 `-v "$PWD:/app"` 挂载时，修改代码后只需重启 `run_server.py`，无需重新 build。只有 Dockerfile 中 `COPY` 固化的代码才需要重建镜像。

### 页面能打开但显示连接错误

通常是 `--cors-origins` 没包含页面地址。需要在启动命令中加入：

```bash
--cors-origins http://localhost:9093,http://127.0.0.1:9093,http://你的IP:9093
```

修改后重启服务。

### localhost 容易混淆

远程浏览器访问时 `localhost` 指的是你自己电脑而非服务器。远程访问统一用 `9093`：

```
页面：http://服务器IP:9093/
WebSocket：ws://服务器IP:9093/ws
```

`9090/9094` 是后端端口，通常不直接填到浏览器页面的 WebSocket 地址中。

### FunASR 启动后还在下载模型

通常是启动参数没有指向本地模型目录，或使用了 `iic/` 模型 ID。确认本地有完整模型文件并使用本地路径。

### 中控 `HEAD` 请求返回 405

`curl -I` 发的 `HEAD` 被 `/admin/clients` 拒绝，因为该接口只支持 `GET`。验证请用：

```bash
curl http://127.0.0.1:9093/admin/clients
```

---

## 17. 停止服务

docker-compose 方式：

```bash
docker compose -f docker-compose.local.yml down
```

手动方式：

```bash
docker stop whisperlive-web-gateway whisperlive-gpu0
```

---

## 18. 原始项目地址

```text
https://github.com/collabora/WhisperLive
```
