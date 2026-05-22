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
9090  WebSocket ASR 服务
9093  Web 前端和中控页面
9094  Admin API，映射到容器内 8000
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

如果已经有 `whisperlive-server` 镜像，可以跳过这一步。

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
  --cors-origins http://192.168.1.100:9093,http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/whisper-small-zh_tw-ct2/
```

说明：

- `-fw` 指定服务端实际使用的 ASR 模型。
- `--translation_device cpu` 表示翻译模型走 CPU，GPU 优先留给 ASR。
- `--rest_port 8000` 是容器内 Admin API 端口，通过 `-p 9094:8000` 暴露到宿主机。
- `--cors-origins` 必须包含中控页面地址，否则浏览器会显示连接错误。

如果要换模型，例如 small：

```bash
-fw model/asr/small
```

如果要用中文 int8 模型：

```bash
-fw model/asr/faster-whisper-belle-large-v3-turbo-zh-int8
```

## 4. Server 端：启动中控页面

server 端机器负责跑 ASR、翻译、Admin API 和中控页面。另开一个 server 端宿主机终端，在项目根目录执行：

```bash
docker run --rm -it \
  --name whisperlive-admin-web \
  --network whisperlive-net \
  -p 9093:80 \
  -v "$PWD/web:/usr/share/nginx/html:ro" \
  nginx:alpine
```

中控页面地址：

```text
http://192.168.1.100:9093/admin.html
```

中控页面里的 `Admin API` 填 server 端 Admin API 地址：

```text
http://192.168.1.100:9094
```

如果是在 server 机器本机浏览器打开中控，也可以用：

```text
http://localhost:9093/admin.html
```

对应 `Admin API` 填：

```text
http://localhost:9094
```

## 5. Client 端：打开 Web 同传页面

Web 同传页面使用打开网页那台机器的麦克风。client 端可以和 server 端是同一台机器，也可以是另一台机器。

### 情况 A：client 和 server 是同一台机器

浏览器打开：

```text
http://localhost:9093/
```

页面里的 `Server` 填：

```text
ws://localhost:9090
```

### 情况 B：client 和 server 不是同一台机器，直接使用 server 提供的 Web 页面

client 机器浏览器打开 server 地址：

```text
http://192.168.1.100:9093/
```

页面里的 `Server` 填 server 的 WebSocket 地址：

```text
ws://192.168.1.100:9090
```

这种方式下，client 机器不需要启动 Docker，也不需要运行 `run_client.py`。浏览器会采集 client 机器的麦克风，然后把音频通过 WebSocket 发到 server 端。

注意：浏览器麦克风权限通常要求安全上下文。`localhost` 一般可以直接用；如果通过 `http://192.168.1.100:9093/` 访问远程页面，部分浏览器可能禁止麦克风。这种情况建议用下面的情况 C，或者给 Web 页面配置 HTTPS。

### 情况 C：client 机器自己启动 Web 页面，连接远程 server

如果 client 机器本地也有本项目代码，可以只在 client 机器启动静态 Web 页面：

```bash
cd web
python3 -m http.server 8080
```

client 机器浏览器打开：

```text
http://localhost:8080
```

页面里的 `Server` 填远程 server 的 WebSocket 地址：

```text
ws://192.168.1.100:9090
```

这种方式的好处是页面运行在 client 本机的 `localhost`，浏览器更容易允许麦克风权限；server 端只需要开放 `9090` WebSocket 和 `9094` Admin API，中控页面仍然可以在 server 端打开。

### 完整推荐流程：server 和 client 不同机器

server 端执行：

```bash
# 1. 启动后端容器
docker run --rm -it --gpus '"device=0"' \
  --name whisperlive-server \
  --network whisperlive-net \
  -p 9090:9090 \
  -p 9094:8000 \
  -v "$PWD:/app" \
  -w /app \
  whisperlive-server bash

# 2. 容器内启动服务
python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --translation_device cpu \
  --rest_port 8000 \
  --cors-origins http://192.168.1.100:9093,http://localhost:8080,http://127.0.0.1:8080 \
  -fw model/asr/whisper-small-zh_tw-ct2/
```

server 端另开终端启动中控页面：

```bash
docker run --rm -it \
  --name whisperlive-admin-web \
  --network whisperlive-net \
  -p 9093:80 \
  -v "$PWD/web:/usr/share/nginx/html:ro" \
  nginx:alpine
```

server 端打开中控：

```text
http://localhost:9093/admin.html
```

中控里的 `Admin API` 填：

```text
http://localhost:9094
```

client 端启动 Web 页面：

```bash
cd web
python3 -m http.server 8080
```

client 端浏览器打开：

```text
http://localhost:8080
```

client 页面里的 `Server` 填：

```text
ws://192.168.1.100:9090
```

然后点击开始，允许浏览器麦克风权限。

## 6. 命令行 client 使用

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

## 7. 压测脚本

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

## 8. 常见问题

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

验证 Admin API 请用：

```bash
curl http://127.0.0.1:9094/admin/clients
```

### `localhost` 容易混淆

如果你是在自己电脑浏览器访问远程服务器：

```text
localhost
```

指的是你自己的电脑，不是 SSH 服务器。

远程访问请使用：

```text
http://192.168.1.100:9093/admin.html
http://192.168.1.100:9094
ws://192.168.1.100:9090
```

## 9. 停止服务

停止前端容器：

```bash
docker stop whisperlive-admin-web
```

停止后端容器：

```bash
docker stop whisperlive-server
```

## 10. 原始项目地址

原 fork / 上游 README 可参考：

```text
https://github.com/collabora/WhisperLive
```
