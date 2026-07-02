# WhisperLive 运维使用指导：单卡 / 双卡模式

本文档面向运维人员，说明 WhisperLive 在生产环境中如何按单卡或双卡模式运行，如何切换 Nginx 模式，以及前端 WebSocket 和中控 Admin API 应该如何填写。

## 1. 生产访问入口

生产环境统一使用带端口的访问地址：

```text
前端页面：
https://app.cmtbs.com:57890/

中控页面：
https://app.cmtbs.com:57890/admin.html
```

注意：当前生产环境必须带 `:57890`。如果不带端口，可能访问到其他入口，或者 WebSocket / Admin API 无法连通。

## 2. 服务组成

| 服务 | 说明 |
|---|---|
| `whisperlive-web-gateway` | Nginx 网关，负责前端页面、WebSocket、Admin API 转发 |
| `whisperlive-gpu0` | GPU0 后端服务，可运行 Whisper 或 FunASR |
| `whisperlive-gpu1` | GPU1 后端服务，可运行 Whisper 或 FunASR |

GPU 编号不固定绑定模型。也就是说：

- `gpu0` 可能跑 Whisper，也可能跑 FunASR。
- `gpu1` 可能跑 Whisper，也可能跑 FunASR。
- 实际模型以该容器内启动的脚本和 Admin API 返回结果为准。

## 3. 单卡模式与双卡模式

### 3.1 单卡模式

单卡模式只启用一个后端服务，可能是 `whisperlive-gpu0`，也可能是 `whisperlive-gpu1`。

适合场景：

- 只有一张 GPU 可用。
- 当前只需要一种识别模型。
- 排查问题时希望减少变量。
- 只给用户暴露一个统一 WebSocket 地址。

单卡模式下，用户前端通常填写：

```text
wss://app.cmtbs.com:57890/ws
```

中控 Admin API 通常填写：

```text
https://app.cmtbs.com:57890
```

### 3.2 双卡模式

双卡模式同时启用 `whisperlive-gpu0` 和 `whisperlive-gpu1`。

适合场景：

- 两张 GPU 同时提供服务。
- 两张卡可能运行不同模型。
- 用户或运维需要明确选择连接 GPU0 或 GPU1。
- 需要分别观察 GPU0 / GPU1 的客户端状态、日志和负载。

双卡模式下，WebSocket 有三种入口：

```text
自动分流：
wss://app.cmtbs.com:57890/ws

固定 GPU0：
wss://app.cmtbs.com:57890/ws-gpu0

固定 GPU1：
wss://app.cmtbs.com:57890/ws-gpu1
```

如果两张卡运行的是不同模型，不建议普通用户使用 `/ws` 自动分流。应明确填写 `/ws-gpu0` 或 `/ws-gpu1`，避免用户不知道自己连到了哪个模型。

## 4. 切换 Nginx 单卡 / 双卡模式

项目提供 Nginx 模式切换脚本：

```bash
./scripts/switch_nginx_mode.sh single
./scripts/switch_nginx_mode.sh dual
```

该脚本会切换当前生效的 Nginx 配置，并在 `whisperlive-web-gateway` 正在运行时自动重启该容器。

### 4.1 切换到单卡模式

在项目根目录执行：

```bash
./scripts/switch_nginx_mode.sh single
```

执行后应看到类似输出：

```text
[OK] Switched Nginx config to SINGLE GPU mode.
[OK] Restarted whisperlive-web-gateway.
```

### 4.2 切换到双卡模式

在项目根目录执行：

```bash
./scripts/switch_nginx_mode.sh dual
```

执行后应看到类似输出：

```text
[OK] Switched Nginx config to DUAL GPU mode.
[OK] Restarted whisperlive-web-gateway.
```

### 4.3 切换后检查容器

执行：

```bash
docker ps
```

单卡模式至少应看到：

```text
whisperlive-web-gateway
whisperlive-gpu0 或 whisperlive-gpu1
```

双卡模式应看到：

```text
whisperlive-web-gateway
whisperlive-gpu0
whisperlive-gpu1
```

## 5. 首次拉起容器

本节用于从宿主机拉起 GPU 后端容器和前端 Nginx 网关。以下命令需要在项目根目录执行。

### 5.1 前置检查

确认 Docker 网络存在：

```bash
docker network ls | grep whisperlive-net
```

如果不存在，先创建：

```bash
docker network create whisperlive-net
```

确认镜像存在：

```bash
docker images | grep whisperlive-server
```

当前生产示例使用镜像：

```text
whisperlive-server:32b
```

确认 HTTPS 证书目录存在：

```bash
ls /etc/letsencrypt/live/app.cmtbs.com
```

前端 Nginx 容器会以只读方式挂载 `/etc/letsencrypt`。如果证书目录不存在，Nginx HTTPS 启动会失败。

### 5.2 拉起 GPU0 后端容器

```bash
docker run -it --gpus '"device=0"' \
  --name whisperlive-gpu0 \
  --network whisperlive-net \
  -v "$PWD:/app" \
  -w /app \
  whisperlive-server:32b bash
```

进入容器后，根据实际需要启动 Whisper 或 FunASR：

```bash
./scripts/start_whisper_service.sh
```

或：

```bash
./scripts/start_funasr_service.sh
```

### 5.3 拉起 GPU1 后端容器

```bash
docker run -it --gpus '"device=1"' \
  --name whisperlive-gpu1 \
  --network whisperlive-net \
  -v "$PWD:/app" \
  -w /app \
  whisperlive-server:32b bash
```

进入容器后，根据实际需要启动 Whisper 或 FunASR：

```bash
./scripts/start_whisper_service.sh
```

或：

```bash
./scripts/start_funasr_service.sh
```

### 5.4 拉起前端 Nginx 网关

启动前先确认 `deploy/nginx/whisperlive.conf` 已经是目标模式配置。需要切换时先执行：

```bash
./scripts/switch_nginx_mode.sh single
```

或：

```bash
./scripts/switch_nginx_mode.sh dual
```

然后拉起 Nginx：

```bash
docker run --rm -it \
  --name whisperlive-web-gateway \
  --network whisperlive-net \
  -p 57890:443 \
  -v "$PWD/web:/usr/share/nginx/html:ro" \
  -v "$PWD/deploy/nginx/whisperlive.conf:/etc/nginx/conf.d/default.conf:ro" \
  -v "/etc/letsencrypt:/etc/letsencrypt:ro" \
  nginx:alpine
```

外部访问地址为：

```text
https://app.cmtbs.com:57890/
```

### 5.5 容器已存在时的处理

如果 `docker run` 提示容器名已存在，先查看当前容器：

```bash
docker ps -a | grep whisperlive
```

如果确认旧容器不再使用，可以停止并删除：

```bash
docker stop whisperlive-gpu0 whisperlive-gpu1 whisperlive-web-gateway
```

```bash
docker rm whisperlive-gpu0 whisperlive-gpu1 whisperlive-web-gateway
```

如果只重启某一个容器，只处理对应容器即可。

### 5.6 拉起后检查

查看容器：

```bash
docker ps
```

查看 GPU：

```bash
nvidia-smi
```

检查 GPU0 Admin API：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

检查 GPU1 Admin API：

```text
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

检查前端页面：

```text
https://app.cmtbs.com:57890/
```

## 6. 启动后端服务

### 6.1 启动 Whisper 服务

进入目标 GPU 容器。下面以 GPU0 为例：

```bash
docker exec -it whisperlive-gpu0 bash
```

启动 Whisper：

```bash
./scripts/start_whisper_service.sh
```

如果要在 GPU1 上启动 Whisper，则进入 GPU1：

```bash
docker exec -it whisperlive-gpu1 bash
./scripts/start_whisper_service.sh
```

### 6.2 启动 FunASR 服务

进入目标 GPU 容器。下面以 GPU1 为例：

```bash
docker exec -it whisperlive-gpu1 bash
```

启动 FunASR：

```bash
./scripts/start_funasr_service.sh
```

如果要在 GPU0 上启动 FunASR，则进入 GPU0：

```bash
docker exec -it whisperlive-gpu0 bash
./scripts/start_funasr_service.sh
```

### 6.3 注意事项

- GPU 编号和模型没有固定绑定关系。
- 同一张 GPU 上不建议同时运行多个大模型服务，除非确认显存和并发压力可控。
- 启动完成后，应通过 Admin API 确认该容器实际运行的后端类型。

## 7. 前端 WebSocket 填写方式

用户打开前端页面：

```text
https://app.cmtbs.com:57890/
```

在设置中的 WebSocket 地址填写以下之一。

### 8.1 单卡模式推荐填写

```text
wss://app.cmtbs.com:57890/ws
```

说明：

- `/ws` 会转发到当前单卡 Nginx 配置指定的后端。
- 用户不需要知道实际使用 GPU0 还是 GPU1。
- 实际模型由该后端容器当前启动的服务决定。

### 7.2 双卡模式自动分流

```text
wss://app.cmtbs.com:57890/ws
```

说明：

- Nginx 会自动分配到 GPU0 或 GPU1。
- 如果两张卡跑的是不同模型，不建议使用自动分流。
- 自动分流适合两张卡跑相同模型，或用户不关心具体模型的场景。

### 7.3 双卡模式固定 GPU0

```text
wss://app.cmtbs.com:57890/ws-gpu0
```

说明：

- 固定连接 `whisperlive-gpu0`。
- GPU0 当前跑 Whisper 还是 FunASR，需要通过 Admin API 确认。

### 7.4 双卡模式固定 GPU1

```text
wss://app.cmtbs.com:57890/ws-gpu1
```

说明：

- 固定连接 `whisperlive-gpu1`。
- GPU1 当前跑 Whisper 还是 FunASR，需要通过 Admin API 确认。

## 8. 中控 Admin API 填写方式

打开中控页面：

```text
https://app.cmtbs.com:57890/admin.html
```

页面顶部有 `Admin API` 输入框。

注意：这里填写的是 Admin API 基础地址，不要填写完整 `/admin/clients`。

### 8.1 单卡模式

填写：

```text
https://app.cmtbs.com:57890
```

中控实际访问：

```text
https://app.cmtbs.com:57890/admin/clients
```

### 8.2 双卡模式查看 GPU0

填写：

```text
https://app.cmtbs.com:57890/admin-gpu0
```

中控实际访问：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

也可以直接在浏览器打开：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

### 8.3 双卡模式查看 GPU1

填写：

```text
https://app.cmtbs.com:57890/admin-gpu1
```

中控实际访问：

```text
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

也可以直接在浏览器打开：

```text
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

## 9. 如何确认某张卡正在跑什么模型

### 9.1 查看 GPU0

打开：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

查看返回 JSON 中的字段：

```json
"server_backend": "faster_whisper"
```

表示 GPU0 当前运行 Whisper / faster-whisper。

如果看到：

```json
"server_backend": "funasr"
```

表示 GPU0 当前运行 FunASR。

### 9.2 查看 GPU1

打开：

```text
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

同样查看：

```json
"server_backend"
```

客户端记录中也可能看到：

```json
"backend": "faster_whisper"
"model": "model/asr/large-v3-turbo"
```

或：

```json
"backend": "funasr"
```

## 10. 推荐操作流程

### 10.1 单卡启动流程

1. 切换 Nginx 到单卡模式：

```bash
./scripts/switch_nginx_mode.sh single
```

2. 选择实际使用的 GPU 容器，例如 GPU0：

```bash
docker exec -it whisperlive-gpu0 bash
```

3. 根据需要启动 Whisper 或 FunASR：

```bash
./scripts/start_whisper_service.sh
```

或：

```bash
./scripts/start_funasr_service.sh
```

4. 前端填写：

```text
wss://app.cmtbs.com:57890/ws
```

5. 中控填写：

```text
https://app.cmtbs.com:57890
```

6. 检查服务状态：

```bash
docker ps
nvidia-smi
```

### 10.2 双卡启动流程

1. 切换 Nginx 到双卡模式：

```bash
./scripts/switch_nginx_mode.sh dual
```

2. 进入 GPU0 容器，启动需要的模型：

```bash
docker exec -it whisperlive-gpu0 bash
./scripts/start_whisper_service.sh
```

或：

```bash
./scripts/start_funasr_service.sh
```

3. 进入 GPU1 容器，启动需要的模型：

```bash
docker exec -it whisperlive-gpu1 bash
./scripts/start_whisper_service.sh
```

或：

```bash
./scripts/start_funasr_service.sh
```

4. 运维确认 GPU0 当前模型：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

5. 运维确认 GPU1 当前模型：

```text
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

6. 用户如果要固定连接 GPU0，填写：

```text
wss://app.cmtbs.com:57890/ws-gpu0
```

7. 用户如果要固定连接 GPU1，填写：

```text
wss://app.cmtbs.com:57890/ws-gpu1
```

## 11. 常用检查命令

### 11.1 查看容器

```bash
docker ps
```

### 11.2 查看 GPU

```bash
nvidia-smi
```

### 11.3 查看 GPU0 日志

```bash
docker logs --tail 100 whisperlive-gpu0
```

### 11.4 查看 GPU1 日志

```bash
docker logs --tail 100 whisperlive-gpu1
```

### 11.5 查看 Nginx 网关日志

```bash
docker logs --tail 100 whisperlive-web-gateway
```

## 12. 常见问题

### 12.1 WebSocket 连不上

确认地址必须带端口：

```text
wss://app.cmtbs.com:57890/ws
wss://app.cmtbs.com:57890/ws-gpu0
wss://app.cmtbs.com:57890/ws-gpu1
```

不要写成：

```text
wss://app.cmtbs.com/ws
```

### 12.2 中控没有数据显示

Admin API 输入框不要填完整 `/admin/clients`。

正确：

```text
https://app.cmtbs.com:57890/admin-gpu0
```

错误：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
```

因为中控会自动拼接 `/admin/clients`。

### 12.3 不确定 GPU0 / GPU1 当前跑的是什么

直接打开：

```text
https://app.cmtbs.com:57890/admin-gpu0/admin/clients
https://app.cmtbs.com:57890/admin-gpu1/admin/clients
```

查看：

```json
"server_backend"
```

### 12.4 双卡不同模型时是否推荐用 `/ws`

不推荐。

如果 GPU0 和 GPU1 跑的是不同模型，用户应该明确填写：

```text
wss://app.cmtbs.com:57890/ws-gpu0
```

或：

```text
wss://app.cmtbs.com:57890/ws-gpu1
```

只有两张卡跑相同模型，或者用户不关心具体模型时，才建议使用：

```text
wss://app.cmtbs.com:57890/ws
```

