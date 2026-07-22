# WhisperLive 生产运维指南：5090 双卡

本文档面向生产运维人员，说明 5090×2 部署机的服务分工、路由要求、启动参数、监控和故障处理。普通用户只选择业务模式，不需要了解 GPU 或 WebSocket 路由。

## 1. 生产入口与架构

```text
用户页面：https://app.cmtbs.com:57890/
中控页面：https://app.cmtbs.com:57890/admin.html
```

| 服务 | 职责 |
| --- | --- |
| `whisperlive-web-gateway` | 静态页面、WebSocket 和 Admin API 的 HTTPS 入口 |
| `whisperlive-gpu0` | Faster-Whisper ASR；高精同传固定连接该后端 |
| `whisperlive-gpu1` | Faster-Whisper ASR；参与普通业务分流 |
| 总结 LLM | 按需提供会议总结，生产模型为 `qwen3-32b-awq` |

两个 ASR 后端均使用 `model/asr/large-v3-turbo`。

## 2. 业务路由与资源分工

### 2.1 正式路由

| 路由 | 业务模式 | ASR | 翻译 |
| --- | --- | --- | --- |
| `/ws-standard` | 普通同传、对话翻译、语音识别 | 在 GPU0、GPU1 后端之间分流 | 普通同传/对话翻译使用 CPU；语音识别不翻译 |
| `/ws-accurate` | 高精同传 | 固定 GPU0 后端 | 使用物理 GPU1 |

前端会根据业务模式自动切换路由，并在高精同传配置中发送 `translation_device=cuda:1`。因此承载 `/ws-accurate` 的 GPU0 容器必须同时看到物理 GPU0 和 GPU1。

GPU1 后端容器只需要看到物理 GPU1；容器内该卡编号为 `cuda:0`，用于本容器的 ASR。

### 2.2 并发策略

高精翻译和 GPU1 上的普通 ASR 允许同时运行，不是独占模式。并发期间重点观察：

- GPU1 显存余量及 `CUDA out of memory`。
- 普通 ASR 的实时因子和断句延迟。
- 高精翻译队列、首条译文时间和失败标记。
- GPU0/GPU1 当前连接数。

出现显存或延迟压力时，优先减少高精会议并发、将新用户切换到普通同传，并确认没有总结模型同时占用相同 GPU。修改 batch、模型大小、翻译设备或常驻模型前必须评估显存和延迟。

### 2.3 兼容路由

`/ws`、`/ws-gpu0`、`/ws-gpu1` 仅用于旧客户端兼容、固定后端测试和排障，不作为普通用户入口。用户页面中的地址会被业务模式改写为正式路由。

## 3. 端口和 Nginx 要求

| 端口 | 用途 |
| --- | --- |
| `9090` | 后端 WebSocket |
| `8000` | 容器内 Admin API |
| `8001` | 总结模型 OpenAI 兼容 API |
| `57890` | 当前生产 HTTPS 对外入口 |

生产 Nginx 配置必须满足：

- `/ws-standard` 在两个 ASR 后端之间分流，并透传 WebSocket Upgrade。
- `/ws-accurate` 固定转发到 GPU0 后端。
- `/admin/` 能访问用户页面所需的客户端、热词、模型、会议日志、模板和总结接口。
- `/admin-gpu0/`、`/admin-gpu1/` 如保留，只用于逐节点检查。
- Admin 上传上限至少为 2 MB，读取超时应覆盖总结生成时间。
- 证书和对外端口以部署机实际配置为准，业务代码不得写死。

`deploy/` 已被 `.gitignore` 排除，仓库中的本地 Nginx 文件不能视为生产配置来源。修改或切换前先备份并检查部署机当前配置。`scripts/switch_nginx_mode.sh` 依赖环境中存在的单/双卡模板；模板缺失时不要执行。

### 3.1 Admin 数据一致性

前端从标准或高精 WebSocket 地址推导出的管理入口均为 `/admin/`。生产环境必须采用以下一种一致方案：

- 两个后端共享会议日志和总结模板目录，`/admin/` 固定到统一管理节点；或
- 由统一 Admin 服务聚合两个后端的数据。

不要把会议日志完全隔离到两个不可互访的目录后，仍期望用户页面通过单一 `/admin/` 找到全部会议。部署后必须分别用 GPU0、GPU1 完成一次会议，并确认两次会议都能在总结列表中出现和下载。

## 4. 容器与 GPU 可见性

以下命令展示生产所需的 GPU 可见性，镜像标签、网络和持久化挂载以部署机实际配置为准。

GPU0 容器需要看到两张物理卡：

```bash
docker run -it -d --gpus '"device=0,1"' \
  --name whisperlive-gpu0 \
  --network whisperlive-net \
  -v "$PWD:/app" -w /app \
  whisperlive-server:32b bash
```

GPU1 容器只需要看到物理 GPU1：

```bash
docker run -it -d --gpus '"device=1"' \
  --name whisperlive-gpu1 \
  --network whisperlive-net \
  -v "$PWD:/app" -w /app \
  whisperlive-server:32b bash
```

拉起前先确认：

```bash
docker network ls
docker images
nvidia-smi
```

不要直接删除或重建已有容器。容器名冲突或配置变化时，先确认当前客户端数、挂载、环境变量和正在运行的进程。

## 5. 启动 ASR 后端

两个后端均以 Faster-Whisper `large-v3-turbo` 启动，服务端默认翻译设备保持为 CPU。高精模式由前端仅对对应连接覆盖为 GPU1。

GPU0 后端示例：

```bash
docker exec -it whisperlive-gpu0 bash

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --asr_device_index 0 \
  --translation_device cpu \
  --batch_inference \
  --batch_max_size 12 \
  --batch_window_ms 20 \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --summary_model qwen3-32b-awq \
  --cors-origins https://app.cmtbs.com:57890,https://app.cmtbs.com \
  -fw model/asr/large-v3-turbo
```

GPU1 后端容器只看到一张卡，因此容器内 ASR 仍使用设备索引 `0`：

```bash
docker exec -it whisperlive-gpu1 bash

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --asr_device_index 0 \
  --translation_device cpu \
  --batch_inference \
  --batch_max_size 12 \
  --batch_window_ms 20 \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --summary_model qwen3-32b-awq \
  --cors-origins https://app.cmtbs.com:57890,https://app.cmtbs.com \
  -fw model/asr/large-v3-turbo
```

如果生产使用共享数据卷，将两个命令中的 `--meeting_logs_dir` 和 `--summary_templates_dir` 指向同一共享路径。`scripts/start_whisper_service.sh` 是通用脚本，默认参数不等同于完整的 5090×2 生产配置。启动后应从进程参数和 Admin API 再次核对。

## 6. 总结服务

生产总结模型为 `qwen3-32b-awq`。`--summary_model` 必须与 vLLM 的 `--served-model-name` 一致。

通用脚本默认使用 4B 模型；生产使用 32B 时必须同时设置模型目录和模型名：

```bash
SUMMARY_MODEL_PATH=model/LLM/Qwen3-32B-AWQ \
SUMMARY_MODEL_NAME=qwen3-32b-awq \
bash scripts/start_summary_llm_service.sh
```

实际模型目录以部署机为准。总结服务可由后端按需启动，并在空闲超时后关闭。生成失败时优先检查：

- `qwen3-32b-awq` 名称是否和 vLLM served model 一致。
- `http://127.0.0.1:8001/v1` 是否可达。
- 模型加载是否超时或出现 CUDA OOM。
- 是否同时存在高精翻译、普通 ASR 和总结任务。
- 会议是否已经处于 `finished` 状态。

不要为了生成总结临时安装依赖、复制模型实例或提高 GPU 内存利用率。

## 7. 中控使用

打开 `https://app.cmtbs.com:57890/admin.html`。“Admin API”输入框填写基础地址，不要附加 `/admin/clients`。双卡环境按用途填写：

| 用途 | Admin API 输入框 |
| --- | --- |
| 默认入口（当前转到 GPU0） | `https://app.cmtbs.com:57890` |
| 显式查看 GPU0 | `https://app.cmtbs.com:57890/admin-gpu0` |
| 查看 GPU1 | `https://app.cmtbs.com:57890/admin-gpu1` |

中控会在输入值后自动拼接 `/admin/clients`、`/admin/hotwords`、`/admin/warmup/status` 等接口路径。一个中控页面一次只查看一个后端；切换 GPU 时先停止当前轮询，替换输入值，再点击“开始”。

按当前生产 Nginx 配置，默认 `/admin/` 固定代理到 GPU0，不是双卡聚合服务。热词查看、服务预热、客户端断开和删除等操作只作用于当前输入地址对应的后端。两个后端未共享会议日志时，默认入口也无法看到 GPU1 独立保存的会议记录。

中控支持：

- 查看在线、空闲、断开和无翻译客户端。
- 查看后端、模型、语言、消息数量、最近原文和译文。
- 查看会议热词文件及固定翻译数量。
- 检查并执行服务预热。
- 删除断开记录。
- 断开在线客户端并删除记录。

点击在线客户端的删除按钮会主动断开连接，页面会要求确认。操作前应确认该客户端不是正在进行的正式会议。

## 8. 启动后验收

### 8.1 基础检查

```bash
docker ps
nvidia-smi
```

确认两个容器内各只有一个预期的 `run_server.py` 进程，并核对 Admin API 返回的 `server_backend`、ASR 模型、客户端计数和最大连接数。

### 8.2 业务验收

1. 普通同传建立多个连接，确认请求能分配到两个 ASR 后端且翻译使用 CPU。
2. 高精同传确认 WebSocket 固定到 GPU0，翻译模型加载在物理 GPU1。
3. 高精与普通会议并发，观察 GPU1 显存、ASR 延迟和翻译延迟。
4. 分别让 GPU0、GPU1 后端承载并结束一次普通会议，确认统一总结列表可见两次会议。
5. 下载 Markdown、DOCX 原文和中英对照。
6. 在无活动会议时生成一次 `qwen3-32b-awq` 总结并下载。

## 9. 常用只读检查

```bash
docker ps
nvidia-smi
docker logs --tail 100 whisperlive-gpu0
docker logs --tail 100 whisperlive-gpu1
docker logs --tail 100 whisperlive-web-gateway
```

日志较大时优先限定时间范围并搜索 `ERROR`、`CUDA out of memory`、`translation`、`summary` 和 `SEGMENT_COMPLETE`。

## 10. 常见故障

### 10.1 页面能打开但无法开始

检查 HTTPS 入口、证书、CORS、WebSocket Upgrade，以及 `/ws-standard`、`/ws-accurate` 是否存在。前端会自动改写路径，不要只检查旧 `/ws`。

### 10.2 普通业务只落到一个后端

检查 `/ws-standard` 的 upstream、两个后端健康状态和最大连接数。不要通过让用户手工填写 GPU 地址来替代修复分流。

### 10.3 高精模式没有使用 GPU1 翻译

确认请求进入 `/ws-accurate` 并固定到 GPU0、GPU0 容器可见两张物理 GPU、客户端包含 `translation_device=cuda:1`，且 NLLB 3.3B 模型被 Admin 模型列表识别。模型缺失时前端会回退到可用版本。

### 10.4 GPU1 OOM 或普通 ASR 变慢

GPU1 同时承担普通 ASR 和高精翻译。先停止或减少高精会议，确认总结模型没有同时占用该卡，再评估 batch、模型和翻译设备配置。

### 10.5 用户找不到刚结束的会议

确认会议状态为 `finished`，检查会议实际落在哪个后端、`/admin/` 指向哪个节点，以及两个节点是否共享日志数据。

### 10.6 中控没有数据

Admin 输入框填写基础地址。验证接口时使用 GET：

```bash
curl https://app.cmtbs.com:57890/admin/clients
```

`curl -I` 发送 HEAD，接口返回 405 不代表 GET 不可用。

## 11. 变更与重启原则

- 修改前确认没有活动客户端，并记录当前容器、进程、挂载和参数。
- 前端和代码通常通过目录挂载同步，不要无依据重建镜像。
- 容器 PID 1 可能只是 Bash，重启容器后必须检查 `run_server.py` 是否真正启动。
- Nginx、ASR、翻译设备、模型或数据目录变更后，按第 8 节重新验收。
- 不使用 `git reset --hard`、`git checkout --` 或强推覆盖部署机改动。
