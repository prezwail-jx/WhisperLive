# WhisperLive 部署与开发说明

本项目提供浏览器实时语音识别、中英翻译、会议热词、后端会议日志、会后校对和大模型会议总结。

## 文档入口

- [普通用户使用指导](./whisperlive-user-guide.md)：业务模式、会议操作、显示、热词、日志校对与总结。
- [生产运维指南](./whisperlive-ops-guide.md)：5090×2 生产架构、路由、GPU 分工、启动、验收和排障。
- [Web 前端说明](./web/README.md)：本地静态页面与安全上下文。

本文档保留项目结构、本机开发启动和后端功能参考。生产操作以运维指南为准。

## 项目结构

```text
run_server.py                    服务启动入口
whisper_live/server.py           WebSocket、Admin API 与服务编排
whisper_live/meeting/            热词、日志、校对、模板与总结
whisper_live/backend/            ASR 与翻译后端
whisper_live/batch_inference.py  批量 GPU 推理
web/                             浏览器前端与中控
scripts/                         启动、模型下载与压测脚本
deploy/                          环境相关 Nginx 配置（Git 忽略）
```

## 环境区分

### 本机：3060 单卡

本机开发默认使用一个 Faster-Whisper 后端：

- ASR 使用 GPU0。
- 翻译默认使用 CPU，避免挤占 ASR 显存。
- Nginx 统一入口为 `http://localhost:9093`。
- 默认只验证 `/ws-standard` 标准业务。当前高精模式固定请求 `cuda:1`，仅暴露一张卡的默认容器不满足该条件，不应直接使用高精模式。

### 部署机：5090×2

生产环境的正式分工：

| 业务入口 | ASR | 翻译 |
| --- | --- | --- |
| `/ws-standard` | GPU0、GPU1 两个 Faster-Whisper 后端分流 | CPU |
| `/ws-accurate` | 固定 GPU0 后端 | 物理 GPU1 |

两个 ASR 后端均使用 `model/asr/large-v3-turbo`。高精同传优先选择 NLLB 3.3B，会议总结使用 `qwen3-32b-awq`。

`/ws`、`/ws-gpu0`、`/ws-gpu1` 只作为兼容和排障路由。普通用户不需要填写路由或选择 GPU，前端会按业务模式自动切换。

生产部署、共享日志/Admin 要求和验收步骤见[生产运维指南](./whisperlive-ops-guide.md)。

## 1. 本机快速启动

### 1.1 前置条件

- 已构建 `whisperlive-server:docx` 镜像。
- 已创建外部 Docker 网络 `whisperlive-net`。
- 已准备 `model/asr/large-v3-turbo`。
- `deploy/nginx/whisperlive.conf` 已配置为本机入口。

```bash
docker network create --subnet 172.30.0.0/24 whisperlive-net
docker compose -f docker-compose.local.yml up -d
```

访问：

```text
用户页面：http://localhost:9093/
中控页面：http://localhost:9093/admin.html
直接 Admin API：http://localhost:9094/admin/clients
```

`docker-compose.local.yml` 会启动 `whisperlive-gpu0` 和 `whisperlive-web-gateway`。项目目录以 volume 挂载到容器，代码修改通常不需要重建镜像。

停止：

```bash
docker compose -f docker-compose.local.yml down
```

## 2. 构建镜像

```bash
docker build --network=host -f docker/Dockerfile.server -t whisperlive-server:docx .
```

| Dockerfile | 用途 |
| --- | --- |
| `docker/Dockerfile.server` | GPU 生产镜像，包含 ASR、翻译、总结与 DOCX |
| `docker/Dockerfile.gpu` | 精简 GPU 镜像 |
| `docker/Dockerfile.cpu` | CPU-only 镜像 |
| `docker/Dockerfile.tensorrt` | TensorRT 后端 |
| `docker/Dockerfile.openvino` | OpenVINO 后端 |
| `docker/Dockerfile.client` | 静态 Web 前端 |

不要把模型、日志、音频、导出文件或镜像包加入 Git。

## 3. 手动启动 Faster-Whisper

本机容器：

```bash
docker run --rm -it --gpus '"device=0"' \
  --name whisperlive-gpu0 \
  --network whisperlive-net \
  -p 9090:9090 -p 9094:8000 \
  -v "$PWD:/app" -w /app \
  whisperlive-server:docx bash
```

容器内可使用通用脚本：

```bash
ASR_DEVICE_INDEX=0 TRANSLATION_DEVICE=cpu ./scripts/start_whisper_service.sh
```

等效核心参数：

```bash
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 60000 \
  --batch_inference \
  --batch_max_size 1 \
  --asr_device_index 0 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --asr_corrections_dir config/asr_corrections.d \
  --asr_corrections_file config/asr_corrections.d/DOMAIN_CORRECTIONS.txt \
  --meeting_logs_dir logs \
  --cors-origins http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/large-v3-turbo
```

新增启动参数时，应同步检查 `run_server.py` 与 `whisper_live/server.py` 中的 `TranscriptionServer.run()`。

### 手动启动 Web 网关

后端容器启动后，可在项目根目录运行：

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

## 4. 可选 FunASR 后端

FunASR 仍受项目支持，但不是当前 5090×2 生产主线。需要提前准备：

```text
model/funasr/paraformer-zh-streaming
model/funasr/SenseVoiceSmall
model/funasr/ct-punc
model/vad/silero_vad.onnx
```

容器内运行：

```bash
./scripts/start_funasr_service.sh
```

该脚本使用 Paraformer 流式识别，并在断句后尝试使用 SenseVoice 精修。精修失败会退回流式文本。排障优先搜索 `final refinement failed`、`CUDA out of memory` 和 `FUNASR_FINAL_REFINE`。

## 5. 浏览器业务模式

| 模式 | 路由 | 翻译默认值 |
| --- | --- | --- |
| 普通同传 | `/ws-standard` | NLLB 600M，CPU |
| 高精同传 | `/ws-accurate` | 优先 NLLB 3.3B，`cuda:1` |
| 对话翻译 | `/ws-standard` | Helsinki，CPU，自适应中英互译 |
| 语音识别 | `/ws-standard` | 关闭翻译 |

高精模型不可用时，前端会按 1.3B、600M、Helsinki 的顺序选择可用模型。实际可用列表来自 `/admin/translation-models`。

前端还支持：

- 自动互译或指定翻译方向。
- 双栏、上下、交错和单栏显示。
- 原文/译文字号和颜色设置。
- 仅字幕全屏。
- 说话人识别。
- 断线自动重连和继续会议。

## 6. 会议热词

浏览器支持上传 UTF-8 的 `.txt` 或 `.md` 文件，一行一个热词：

```text
WhisperLive
大模型
OpenAI => 开放人工智能
```

- 普通行只作为 ASR 热词。
- `source => target` 同时加入 ASR 热词和固定翻译表。
- 固定翻译为单向规则，多条匹配时优先最长词组。
- 客户端开始会议时锁定热词快照，之后的修改只影响下一次开始。

服务端热词目录由 `--meeting_hotwords_dir` 指定。中控负责扫描和预览服务器热词文件，不负责上传或删除。

### Whisper 错词纠正

Faster-Whisper 的标准中译英和双向翻译场景支持 ASR 错词表。纠错仅作用于已完成的中文片段，纠错后的文本会同步用于源字幕、会议日志和翻译输入；实时 partial 字幕、FunASR、英译中和纯转写不受影响。

全局纠错文件由 `--asr_corrections_file` 指定，可用于所有标准中译英和双向翻译会议。会议级纠错文件目录由 `--asr_corrections_dir` 指定，默认 `config/asr_corrections.d`；文件名去掉 `.txt` 后必须与会议名称一致：

```text
config/asr_corrections.d/DOMAIN_CORRECTIONS.txt
config/asr_corrections.d/产品例会.txt
```

文件内容使用字面量替换规则，不支持正则：

```text
# Whisper 错词 => 正确文本
威斯伯 => Whisper
派森 => Python
开放爱爱 => OpenAI
```

多条规则按错词长度从长到短执行；同一个错词重复出现时以最后一条为准。全局规则先加载，会议规则后加载，同名错词以会议规则为准。文件不存在或规则为空时不会启用纠错。

## 7. 会议日志与校对

会议开始时由前端生成 `session_id` 和开始时间，后端按 session 追加保存 ASR 原文和翻译。前端导出按钮下载后端生成的文件，不在浏览器重新拼接日志。

会后支持：

- Markdown、JSON 和 DOCX 日志。
- DOCX 原文及中英对照布局。
- 修改原文并保留 `original_text`。
- 新增、重命名、合并说话人。
- 修订号冲突保护。
- 修改后将旧译文和总结标记为过期。

日志目录由 `--meeting_logs_dir` 指定。双后端生产环境必须确保统一 `/admin/` 能访问所有会议数据，具体要求见运维指南。

## 8. 总结与自定义模板

总结接口默认地址为 `http://127.0.0.1:8001/v1`。通用默认模型是 `qwen3-4b-awq`；5090×2 生产显式使用 `qwen3-32b-awq`。

`scripts/start_summary_llm_service.sh` 的通用默认值是 Qwen3-4B-AWQ。生产 32B 必须同时设置：

```bash
SUMMARY_MODEL_PATH=model/LLM/Qwen3-32B-AWQ \
SUMMARY_MODEL_NAME=qwen3-32b-awq \
bash scripts/start_summary_llm_service.sh
```

自定义模板支持 `.md` 和 `.docx`，流程为分析模板、确认字段、保存、选择模板并生成。总结会保留历史版本，删除模板不会删除已有总结。

## 9. Admin API 与中控

Admin API 始终在 `--rest_port` 启动；`--enable_rest` 仅控制额外的 OpenAI 兼容 REST ASR 接口。

主要能力包括：

- 客户端状态和主动断开。
- ASR/翻译模型预热。
- 可用翻译模型查询。
- 服务器热词列表与预览。
- 会议日志、校对、说话人和总结。
- 自定义总结模板。

中控的 Admin 输入框填写基础地址，不要附加 `/admin/clients`。

## 10. 批量推理、VAD 与说话人识别

批量推理：

```bash
--batch_inference
--batch_max_size 8
--batch_window_ms 50
```

开启后多个 session 共享模型和 BatchInferenceWorker。`batch_max_size=1` 时退化为逐条调度，但仍共享模型实例。

浏览器默认开启 Silero VAD，减少静音幻觉。说话人识别通过客户端 `enable_diarization` 配置启用，其结果会写入会议日志并可在会后校对。

## 11. 启动参数速查

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--port` | `9090` | WebSocket 端口 |
| `--backend` | `faster_whisper` | ASR 后端 |
| `-fw` | 无 | Faster-Whisper 模型路径 |
| `--max_clients` | `4` | 最大并发连接数 |
| `--max_connection_time` | `300` | 最长连接时间，秒 |
| `--batch_inference` | 关闭 | 启用批量推理 |
| `--batch_max_size` | `8` | 最大 batch |
| `--batch_window_ms` | `50` | batch 等待窗口，毫秒 |
| `--asr_device_index` | `0` | Faster-Whisper CUDA 设备索引 |
| `--translation_device` | `cpu` | `cpu`、`cuda`、`cuda:N` 或 `auto` |
| `--rest_port` | `8000` | Admin API 端口 |
| `--meeting_hotwords_dir` | `config/hotwords.d` | 服务器热词目录 |
| `--asr_corrections_dir` | `config/asr_corrections.d` | Whisper 中译英 ASR 错词规则目录 |
| `--asr_corrections_file` | 无 | 全局 Whisper 中译英 ASR 错词规则文件 |
| `--meeting_logs_dir` | `logs` | 会议日志目录 |
| `--summary_base_url` | `http://127.0.0.1:8001/v1` | 总结 API |
| `--summary_model` | `qwen3-4b-awq` | 通用总结模型名 |
| `--summary_templates_dir` | `config/summary_templates` | 总结模板目录 |

## 12. 最小验证

修改 Python、前端或共享接口时按 `AGENTS.md` 选择最小相关检查，并优先在已运行的 `whisperlive-gpu0` 容器内执行。不要在宿主机安装依赖或自行拉起服务。

纯文档修改只需：

```bash
git diff --check
git status --short
git diff
```

## 13. 原始项目

```text
https://github.com/collabora/WhisperLive
```
