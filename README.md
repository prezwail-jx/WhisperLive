# WhisperLive 部署与使用说明

本项目当前主要用途：

```text
Docker 镜像拉起 WhisperLive 服务
-> GPU 跑 faster-whisper ASR
-> CPU 跑中英翻译
-> 浏览器中控页面查看 client 状态
-> Web 页面或脚本连接服务做实时识别/翻译
```

## 端口规划

推荐固定使用这几个端口：

```text
9090  后端 WebSocket ASR 服务
9093  浏览器统一入口：Web 前端、中控、/ws、/admin/
9094  后端 Admin API，映射到容器内 8000
```

下面示例使用测试服务器 IP：

```text
192.168.1.100
```

实际部署到别的机器时，把命令里的 `192.168.1.100` 替换成你的服务器 IP 或内网地址。

## 1. 构建镜像

在项目根目录执行：

```bash
docker build --network=host -f docker/Dockerfile.server -t whisperlive-server .
```

构建 Web client 静态页面镜像，备用测试时使用：

```bash
docker build -f docker/Dockerfile.client -t whisperlive-client:latest .
```

如果已经有对应镜像，可以跳过这一步。

## 2. 创建 Docker 网络

推荐使用固定网段的 Docker 网络，避免默认 `bridge` 网络和 VPN/内网路由冲突。

```bash
docker network create --subnet 172.30.0.0/24 whisperlive-net
```

如果提示网络已存在，可以忽略。

## 3. 启动后端容器

在项目根目录执行：

```bash
docker run --rm -it --gpus '"device=0"' \
  --name whisperlive-server \
  --network whisperlive-net \
  -p 9090:9090 \
  -p 9094:8000 \
  -v "$PWD:/app" \
  -w /app \
  whisperlive-server bash
```

进入容器后启动服务：

```bash
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --cors-origins http://192.168.1.100:9093,http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/whisper-small-zh_tw-ct2/
```

说明：

- `-fw` 指定服务端实际使用的 ASR 模型。
- `--translation_device cpu` 表示翻译模型走 CPU，GPU 优先留给 ASR。
- `--rest_port 8000` 是容器内 Admin API 端口，通过 `-p 9094:8000` 暴露到宿主机。
- `--meeting_hotwords_dir` 指定服务端会议热词目录，目录内 `会议号.txt` 会自动出现在 Client 和中控下拉列表。
- Web client 默认发送 `min_segment_rms=0.0015`，用于过滤极低音量静音段里的热词幻觉；需要关闭时可在 client payload 中设为 `0`。
- 浏览器推荐只访问 `9093`；`9093` 会把 `/ws` 转到 `9090`，把 `/admin/` 转到 `9094`。
- `--cors-origins` 必须包含中控页面地址，否则浏览器会显示连接错误。

如果要换模型，例如 small：

```bash
-fw model/asr/small
```

如果要用中文 int8 模型：

```bash
-fw model/asr/faster-whisper-belle-large-v3-turbo-zh-int8
```

## 4. 推荐：统一 Web 入口

推荐生产和多人使用时采用统一入口：client 用户只打开一个地址，不需要在 client 机器上安装 Docker。

统一入口 nginx 负责：

```text
/             -> Web 同传页面
/admin.html   -> 中控页面
/ws           -> 反代到后端 WebSocket 9090
/admin/       -> 反代到后端 Admin API 9094
```

先按第 3 节启动后端容器和 `run_server.py`。如果统一入口页面使用 `http://192.168.1.100:9093`，服务启动命令里的 CORS 可以写：

```bash
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --cors-origins http://192.168.1.100:9093,http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/whisper-small-zh_tw-ct2/
```

另开一个 server 端宿主机终端，启动统一入口 nginx：

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

client 用户打开：

```text
http://192.168.1.100:9093/
```

页面会默认连接：

```text
ws://192.168.1.100:9093/ws
```

页面里的 `会议号` 同时作为 client 可读名称和热词归属。Client 页面可以从下拉选择服务端已有热词文件，也可以手动填写会议号；点击开始时会自动加载同名 txt 热词文件并锁定本次连接。

中控打开：

```text
http://192.168.1.100:9093/admin.html
```

中控会默认请求：

```text
http://192.168.1.100:9093/admin/clients
```

这种方式下，多台 client 机器都只需要浏览器打开同一个地址。多路并发仍由后端 `--max_clients`、模型大小和 GPU 性能决定。

如果要让远程浏览器稳定使用麦克风，生产建议给统一入口配置 HTTPS，然后页面会自动使用：

```text
wss://你的域名/ws
```


## 5. 会议热词表

热词文件提前放在 server 机器的 `config/hotwords.d/` 目录中。文件名去掉 `.txt` 后就是会议号，例如：

```text
config/hotwords.d/产品周会.txt  -> 会议号：产品周会
config/hotwords.d/meeting-a.txt -> 会议号：meeting-a
```

文件格式：

```text
图灵科技
faster-whisper
张三
# 这行是注释
```

使用规则：

- 服务启动参数使用 `--meeting_hotwords_dir config/hotwords.d`。
- Admin 页面只负责刷新、下拉选择和预览服务器已有热词文件，不在网页上传或删除文件。
- Client 页面可以从下拉选择已有会议热词，也可以手动填写会议号。
- Client 点击开始时会按会议号读取 `config/hotwords.d/会议号.txt` 的快照并锁定本次连接。
- 开始后再修改服务器 txt 文件，只影响下一次开始。
- 如果会议号没有对应 txt，服务端才会使用 `--hotwords_file` 的全局默认热词。
- 中控 Client 列表会显示会议号、热词文件名、热词是否锁定和热词数量。

新增或修改热词文件后，不需要重启 ASR 服务；在 Client 或 Admin 页面点击刷新即可看到最新列表。

## 6. 备用：client 端 Docker 拉 Web 页面

如果暂时没有统一入口或 HTTPS，可以让 client 机器自己拉起静态 Web 页面容器。client 机器只负责页面，不跑 ASR、不跑翻译，也不需要 GPU。

```bash
docker run --rm -it \
  --name whisperlive-client \
  -p 8080:80 \
  whisperlive-client:latest
```

client 机器浏览器打开：

```text
http://localhost:8080
```

页面里的 `Server` 手动填远程 server 的 WebSocket 地址：

```text
ws://192.168.1.100:9090
```

这种方式仍然支持多路并发：每台 client 的浏览器都会建立独立 WebSocket 连接到 server。

## 7. 命令行 client 使用

推荐在后端容器内执行，依赖最完整。

进入容器：

```bash
docker exec -it whisperlive-server bash
```

测试中文音频并开启翻译：

```bash
python run_client.py \
  --server 127.0.0.1 \
  --port 9090 \
  --files /app/test_zn.wav \
  --lang zh \
  --enable_translation \
  --target_language en \
  --same_output_threshold 2 \
  --mute_audio_playback
```

测试英文音频并翻译成中文：

```bash
python run_client.py \
  --server 127.0.0.1 \
  --port 9090 \
  --files /app/test_en.wav \
  --lang en \
  --enable_translation \
  --target_language zh \
  --same_output_threshold 2 \
  --mute_audio_playback
```

注意：如果服务端启动时已经用了 `-fw`，实际 ASR 模型由服务端 `-fw` 决定，client 传的 `--model` 不会覆盖服务端固定模型。

## 8. 压测脚本

压测脚本用于模拟多路 WebSocket client 实时推流。

进入后端容器：

```bash
docker exec -it whisperlive-server bash
```

两路中文 + 翻译压测：

```bash
python /app/scripts/stress_ws.py \
  --host 127.0.0.1 \
  --port 9090 \
  --audio /app/test_zn.wav \
  --clients 2 \
  --language zh \
  --target_language en \
  --enable_translation \
  --same_output_threshold 2
```

两路英文 + 翻译压测：

```bash
python /app/scripts/stress_ws.py \
  --host 127.0.0.1 \
  --port 9090 \
  --audio /app/test_en.wav \
  --clients 2 \
  --language en \
  --target_language zh \
  --enable_translation \
  --same_output_threshold 2
```

脚本默认会把日志写到：

```text
scripts/stress_logs/
```

常看字段：

- `success`：本次压测是否通过。
- `ready=2/2`：成功连接服务的 client 数。
- `rt_factor`：总耗时 / 音频时长，越接近 `1.0` 越接近实时。
- `segments`：ASR 消息数。
- `translations`：翻译消息数。
- `errors` / `timeout`：连接错误或等待超时。

## 9. 常见问题

### 页面能打开，但中控一直显示连接错误

通常是 `--cors-origins` 没包含页面地址。

服务端启动命令里需要包含：

```bash
--cors-origins http://192.168.1.100:9093,http://localhost:9093,http://127.0.0.1:9093
```

修改后需要重启 `run_server.py`。

### 访问 `http://192.168.1.100:9093/admin.html` 打不开

先在服务器上检查：

```bash
curl -I http://127.0.0.1:9093/admin.html
```

如果服务器本机通，但外部浏览器不通，检查防火墙、安全组或路由器是否放行 `9093`。

如果服务器本机也不通，确认前端容器是否在 `whisperlive-net`：

```bash
docker ps
```

前端容器启动命令应包含：

```bash
--network whisperlive-net
```

### `HEAD /admin/clients 405 Method Not Allowed`

这个不是服务错误。`curl -I` 发的是 `HEAD` 请求，而 `/admin/clients` 只支持 `GET`。

验证统一入口的 Admin API 请用：

```bash
curl http://127.0.0.1:9093/admin/clients
```

如果要直接验证后端裸端口，也可以用 `http://127.0.0.1:9094/admin/clients`。

### `localhost` 容易混淆

如果你是在自己电脑浏览器访问远程服务器：

```text
localhost
```

指的是你自己的电脑，不是 SSH 服务器。

远程浏览器访问统一入口时只用 `9093`：

```text
Web 页面：http://192.168.1.100:9093/
中控页面：http://192.168.1.100:9093/admin.html
WebSocket：ws://192.168.1.100:9093/ws
Admin API：http://192.168.1.100:9093
```

`9090/9094` 是后端真实端口，通常不直接填到浏览器页面里。

## 10. 停止服务

停止前端容器：

```bash
docker stop whisperlive-web-gateway
```

停止后端容器：

```bash
docker stop whisperlive-server
```

## 11. 原始项目地址

原 fork / 上游 README 可参考：

```text
https://github.com/collabora/WhisperLive
```
