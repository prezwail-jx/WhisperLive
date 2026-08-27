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

两个 ASR 后端均使用 `model/asr/large-v3-turbo`。生产仅使用 Faster-Whisper 后端，FunASR 已弃用，不作为生产识别方案。

## 2. 业务路由与资源分工

### 2.1 正式路由

| 路由 | 业务模式 | ASR | 翻译 |
| --- | --- | --- | --- |
| `/ws-standard` | 普通同传、对话翻译、语音识别 | 在 GPU0、GPU1 后端之间分流 | 普通同传/对话翻译使用 CPU；语音识别不翻译 |
| `/ws-accurate` | 高精同传 | 固定 GPU0 后端 | 使用物理 GPU1 |

前端会根据业务模式自动切换路由，并在高精同传配置中发送 `translation_device=cuda:1`。因此承载 `/ws-accurate` 的 GPU0 容器必须同时看到物理 GPU0 和 GPU1。现阶段高精同传同一时间只支持一路，具体策略见 2.2 节。

GPU1 后端容器只需要看到物理 GPU1；容器内该卡编号为 `cuda:0`，用于本容器的 ASR。

`/ws`、`/ws-gpu0`、`/ws-gpu1` 仅用于旧客户端兼容、固定后端测试和排障，不作为普通用户入口。用户页面中的地址会被业务模式改写为正式路由。

### 2.2 并发策略

现阶段高精同传同一时间只允许一路：GPU1 同时承载这一路高精翻译和普通 ASR，允许两者并发，但不是多路高精并存。出现第二路高精请求时，应提示暂不可用并让用户改普通同传，不要通过调整参数或降质临时容纳多路高精。

并发期间重点观察：

- GPU1 显存余量及 `CUDA out of memory`。
- 普通 ASR 的实时因子和断句延迟。
- 高精翻译队列、首条译文时间和失败标记。
- GPU0/GPU1 当前连接数。

GPU1 显存或延迟出现压力时，确认不存在第二路高精连接、总结模型没有同时占用该卡，再评估 batch、模型和翻译设备。修改 batch、模型大小、翻译设备或常驻模型前必须评估显存和延迟。

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

前端从标准或高精 WebSocket 地址推导出的管理入口均为 `/admin/`。生产采用共享目录方案：两个后端共享会议日志和总结模板目录，`/admin/` 固定代理到统一管理节点（当前为 GPU0）。迁移或变更挂载后，必须分别用 GPU0、GPU1 完成一次会议，并确认两次会议都能在总结列表中出现和下载。

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

### 5.1 双卡会话亲和与共享日志

普通模式的 `/ws-standard` 必须按浏览器携带的 `session_id` 固定到同一个后端。实际生产 Nginx 配置中使用以下模式，并在修改后执行 `nginx -t`：

```nginx
map $arg_session_id $standard_route_key {
    ""      "$remote_addr|$http_user_agent";
    default $arg_session_id;
}

upstream whisperlive_standard {
    hash $standard_route_key consistent;
    server whisperlive-gpu0:9090 max_fails=0;
    server whisperlive-gpu1:9090 max_fails=0;
}

location /ws-standard {
    proxy_pass http://whisperlive_standard;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_next_upstream off;
    proxy_next_upstream_tries 1;
    proxy_connect_timeout 5s;
    proxy_read_timeout 28800s;
    proxy_send_timeout 28800s;
}
```

`max_fails=0` 防止一致性哈希在被动失败后预先跳过原节点；`proxy_next_upstream off` 禁止连接失败时改投另一张卡。原节点不可用时会话应失败并由浏览器在约 30 秒内重试，而不是换卡。

两个容器已把同一个宿主机目录（部署机 `/srv/whisperlive/shared/logs`）挂载为 `/shared/meeting-logs`，并均使用：

```text
--meeting_logs_dir /shared/meeting-logs --max_connection_time 28800
```

两个后端均以 Faster-Whisper `large-v3-turbo` 启动，服务端默认翻译设备保持为 CPU。高精模式由前端仅对对应连接覆盖为 GPU1。

GPU0 后端示例：

```bash
docker exec -it whisperlive-gpu0 bash

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 28800 \
  --asr_device_index 0 \
  --translation_device cpu \
  --batch_inference \
  --batch_max_size 12 \
  --batch_window_ms 20 \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --asr_corrections_dir config/asr_corrections.d \
  --asr_corrections_file config/asr_corrections.d/DOMAIN_CORRECTIONS.txt \
  --meeting_logs_dir /shared/meeting-logs \
  --summary_templates_dir /shared/meeting-logs/templates \
  --summary_model qwen3-32b-awq \
  --cors-origins https://app.cmtbs.com:57890 \
  -fw model/asr/large-v3-turbo
```
(已经在script中有)
GPU1 后端容器只看到一张卡，因此容器内 ASR 仍使用设备索引 `0`：

```bash
docker exec -it whisperlive-gpu1 bash

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 28800 \
  --asr_device_index 0 \
  --translation_device cpu \
  --batch_inference \
  --batch_max_size 12 \
  --batch_window_ms 20 \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --asr_corrections_dir config/asr_corrections.d \
  --asr_corrections_file config/asr_corrections.d/DOMAIN_CORRECTIONS.txt \
  --meeting_logs_dir /shared/meeting-logs \
  --summary_templates_dir /shared/meeting-logs/templates \
  --summary_model qwen3-32b-awq \
  --cors-origins https://app.cmtbs.com:57890 \
  -fw model/asr/large-v3-turbo
```
(已经在script中有)
上面的两个命令把 `--meeting_logs_dir` 和 `--summary_templates_dir` 都指向共享卷 `/shared/meeting-logs`（总结模板位于其下 `templates/` 子目录，具体布局以部署机共享卷为准）。两个容器必须使用同一挂载点，否则统一 `/admin/` 仍无法跨节点看到全部会议。`scripts/start_whisper_service.sh` 是通用脚本，默认参数不等同于完整的 5090×2 生产配置。启动后应从进程参数和 Admin API 再次核对。

### 5.2 ASR 错词纠正

生产通过 `--asr_corrections_dir` 和 `--asr_corrections_file` 启用错词纠正，用于修正反复出现的错词、同音字和残句。规则在识别文本输出时替换，不新增模型实例、不增加显存占用。

- 生效条件：仅 Faster-Whisper 后端且启用翻译的中译英方向（标准模式源语言为中文、目标为 `en`/`auto`，以及自动互译的英方向）。纯语音识别、英译中、纯英文不生效。
- `--asr_corrections_file`：全局纠错文件，默认 `config/asr_corrections.d/DOMAIN_CORRECTIONS.txt`，对所有满足上述方向的中译英会议生效。
- `--asr_corrections_dir`：会议级纠错目录，默认 `config/asr_corrections.d`；文件名去掉 `.txt` 后必须与会议名称一致，只对该会议生效。

维护注意：

- 规则文件为 UTF-8 文本，一行一条规则，格式 `源文本=>目标文本`，长规则优先匹配。
- 规则在每个连接初始化时读取。新开始或重连的会议会使用最新规则；已建立的会议连接不会热更新，调整规则后请让相关会议重新开始。
- 全局与会议级规则可叠加；同一源文本同时命中两条规则时，以会议级规则为准。
- 纠错是文本替换，不能替代热词对未收录专有名词的引导。观察日志中的 `ASR_TEXT_CORRECTED` 可确认规则是否命中；错词持续存在时先检查规则格式与生效方向（中译英），再考虑调整规则或识别参数。

### 5.3 说话人识别（待完善）

> 待完善：本节为现状整理，生产 5090×2 是否启用、依赖和显存实测仍需在部署机验证后补充。

说话人识别由客户端在连接时通过 `enable_diarization` 开启，不是独立服务。后端 `_create_diarizer` 在客户端请求开启时按需创建 `SpeakerDiarizer`（`whisper_live/diarization.py`），对每个已完成 ASR 片段的音频提取声纹嵌入，按余弦相似度在线聚类，分配 `SPEAKER_00` 之类的标签并写入会议日志，会后可在“总结”面板校对（新增、重命名、合并说话人，重新分配片段）。

运维注意：

- 依赖可选安装 `pyannote.audio`；嵌入模型默认路径 `model/LLM/wespeaker-voxceleb-resnet34-LM`，首次使用时才加载，设备取容器可见 GPU，无 GPU 则退到 CPU。
- 相关参数：`similarity_threshold` 默认 0.55（值越低越容易合并说话人）、`max_speakers` 默认 10（达到上限后新片段分给最相似说话人）、`hf_token`。
- 容器内未安装 `pyannote.audio` 时功能会被静默禁用，日志出现 `pyannote.audio not installed; diarization disabled`；排查时先确认该包和嵌入模型文件是否就位。
- 每个开启的客户端都会懒加载一个嵌入模型实例并参与片段级推理，可能额外占用显存和增加处理延迟，多路同时开启时需评估资源；目前属于开发阶段功能，不作为生产默认能力。

## 6. 总结服务

生产总结模型为 `qwen3-32b-awq`。`--summary_model` 必须与 vLLM 的 `--served-model-name` 一致。

通用脚本默认使用 4B 模型；生产使用 32B 时必须同时设置模型目录和模型名(服务器中已经配置为32b)：

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

## 7. 启动后验收

### 7.1 基础检查

按第 8 节执行常用只读检查，确认两个容器内各只有一个预期的 `run_server.py` 进程，并核对 Admin API 返回的 `server_backend`、ASR 模型、客户端计数和最大连接数。

### 7.2 业务验收

1. 普通同传建立多个连接，确认请求能分配到两个 ASR 后端且翻译使用 CPU。
2. 高精同传确认 WebSocket 固定到 GPU0，翻译模型加载在物理 GPU1。
3. 一路高精与普通会议并发，观察 GPU1 显存、ASR 延迟和翻译延迟。
4. 分别让 GPU0、GPU1 后端承载并结束一次普通会议，确认统一总结列表可见两次会议。
5. 下载 Markdown、DOCX 原文和中英对照。
6. 在无活动会议时生成一次 `qwen3-32b-awq` 总结并下载。

## 8. 常用只读检查

```bash
docker ps
nvidia-smi
docker logs --tail 100 whisperlive-gpu0
docker logs --tail 100 whisperlive-gpu1
docker logs --tail 100 whisperlive-web-gateway
```

日志较大时优先限定时间范围并搜索 `ERROR`、`CUDA out of memory`、`translation`、`summary` 和 `SEGMENT_COMPLETE`。排查会话恢复时关注 `SESSION_RESUME_FAILED`，排查错词纠正时关注 `ASR_TEXT_CORRECTED`，排查翻译草稿合并时关注 `COALESCED`。

## 9. 中控使用

打开 `https://app.cmtbs.com:57890/admin.html`。“Admin API”输入框填写基础地址，不要附加 `/admin/clients`。双卡环境按用途填写：

| 用途 | Admin API 输入框 |
| --- | --- |
| 默认入口（当前转到 GPU0） | `https://app.cmtbs.com:57890` |
| 显式查看 GPU0 | `https://app.cmtbs.com:57890/admin-gpu0` |
| 查看 GPU1 | `https://app.cmtbs.com:57890/admin-gpu1` |

中控会在输入值后自动拼接 `/admin/clients`、`/admin/hotwords`、`/admin/warmup/status` 等接口路径。一个中控页面一次只查看一个后端；切换 GPU 时先停止当前轮询，替换输入值，再点击“开始”。

按当前生产 Nginx 配置，默认 `/admin/` 固定代理到 GPU0，不是双卡聚合服务；热词查看、服务预热、客户端断开和删除等操作只作用于当前输入地址对应的后端。共享日志目录后，默认入口可看到两张卡承载的全部会议记录（见 3.1 节）。

中控支持：

- 顶部汇总显示当前连接、开启翻译、最近活跃和总 Client 数量。
- 查看在线、空闲、断开和无翻译客户端，以及名称、UID、会议、模型、语言、热词状态、ASR/翻译消息数、最近原文和译文。
- 查看会议热词文件列表、固定翻译数量，并可预览具体会议的热词内容。
- 检查并执行服务预热，可选择在有活动客户端时强制预热。
- 删除断开记录。
- 断开在线客户端并删除记录。

点击在线客户端的删除按钮会主动断开连接，页面会要求确认。操作前应确认该客户端不是正在进行的正式会议。

## 10. 常见故障

### 10.1 页面能打开但无法开始

检查 HTTPS 入口、证书、CORS、WebSocket Upgrade，以及 `/ws-standard`、`/ws-accurate` 是否存在。前端会自动改写路径，不要只检查旧 `/ws`。

### 10.2 普通业务只落到一个后端

检查 `/ws-standard` 的 upstream、两个后端健康状态和最大连接数。不要通过让用户手工填写 GPU 地址来替代修复分流。

### 10.3 高精模式没有使用 GPU1 翻译

确认请求进入 `/ws-accurate` 并固定到 GPU0、GPU0 容器可见两张物理 GPU、客户端包含 `translation_device=cuda:1`，且 NLLB 3.3B 模型被 Admin 模型列表识别。模型缺失时前端会回退到可用版本。

### 10.4 GPU1 OOM 或普通 ASR 变慢

GPU1 同时承担普通 ASR 和唯一一路高精翻译。先确认没有第二路高精连接、总结模型没有同时占用该卡，再评估 batch、模型和翻译设备配置。

### 10.5 用户找不到刚结束的会议

确认会议状态为 `finished`，检查会议实际落在哪个后端、`/admin/` 指向哪个节点，以及两个节点是否共享日志数据。

### 10.6 中控没有数据

Admin 输入框填写基础地址。验证接口时使用 GET：

```bash
curl https://app.cmtbs.com:57890/admin/clients
```

`curl -I` 发送 HEAD，接口返回 405 不代表 GET 不可用。

### 10.7 断线重连后字幕不继续或找不到会议

普通模式按 `session_id` 会话亲和路由，重连会回到初始后端；原后端不可用时不会换卡，连接在浏览器约 30 秒内重试失败，页面提示继续或结束中断会议。检查重连时不要期望 `/ws-standard` 把会话改投另一张卡，先确认分配到的后端进程存活、共享日志目录可写。恢复失败时日志会出现 `SESSION_RESUME_FAILED`。

### 10.8 中控里断开客户端一会儿就消失

断开连接的客户端在中控以轻量快照保留约 10 分钟（容量有限）后自动清除，这是正常行为，不是状态丢失。需要确认断开详情请在快照有效期内操作，或从共享日志目录核对对应 session 记录。

### 10.9 会议在 8 小时后自动断开

生产每个 WebSocket 的连接上限为 28800 秒（8 小时）。到达上限后后端会主动断开，浏览器用同一 `session_id` 自动重连到原后端，时间轴接续。这是预期行为；如果重连后未续上，按 10.7 排查。

### 10.10 标准断句出现残句或翻译不完整

标准（非高精）模式会合并约 700ms 窗口内的短片段，并用完整句缓冲提高中译英翻译完整性，极端情况下可能出现短暂滞留后刷新。先确认原文行是否最终稳定、是否属于静音幻觉过滤或翻译草稿 25 秒 TTL 丢弃；再结合 ASR 错词纠正（5.2 节）和批处理/显存压力（10.4）判断。不要把前端滚动、显示层改动当作 ASR 输出变化。

## 11. 后续稳定性优化方向

在长时间连续运行测试中，已观察到服务运行较久后翻译异常出现概率上升，部分片段可能出现 `Translation unavailable` 或 `translation_exception`；同时也曾出现容器在长时间运行或重启后无法正常识别 GPU 的情况。上述现象目前尚未定位为单一根因，建议作为后续生产稳定性优化的重点持续跟踪。

- **翻译链路稳定性**：重点观察翻译队列长度、任务积压、内存与显存变化、模型/工作线程生命周期以及异常后的恢复情况；后续可评估队列和历史长度上限、完整异常日志、失败重试、模型健康检查与必要时的自动恢复机制。
- **GPU 可见性与容器恢复**：在容器启动、服务重启和长时间运行场景中增加 GPU 可用性自检，核对 NVIDIA Container Runtime、设备映射和 CUDA 初始化状态；若容器无法读取 GPU，应先区分宿主机、容器运行时和应用进程三个层级后再决定是否重启。
- **长时间稳定性压测**：建议增加 4 小时、8 小时及更长时间的连续运行测试，并在普通同传、高精同传和并发场景下记录翻译失败率、翻译队列、GPU 显存、容器内存、客户端数量和 GPU 可见性，以便定位问题出现前后的资源变化。

## 12. 变更与重启原则

- 修改前确认没有活动客户端，并记录当前容器、进程、挂载和参数。
- 前端和代码通常通过目录挂载同步，不要无依据重建镜像。
- 容器 PID 1 可能只是 Bash，重启容器后必须检查 `run_server.py` 是否真正启动。
- Nginx、ASR、翻译设备、模型或数据目录变更后，按第 7 节重新验收。
- 不使用 `git reset --hard`、`git checkout --` 或强推覆盖部署机改动。
