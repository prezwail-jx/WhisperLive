# WhisperLive Web 前端

`web/` 是纯静态浏览器前端，负责麦克风采集、实时字幕展示、连接配置，以及下载后端生成的会议日志和总结。会议日志的拼接、清洗和落盘由后端完成。

## 推荐访问方式

本机开发使用项目的 Nginx 统一入口：

```text
用户页面：http://localhost:9093/
中控页面：http://localhost:9093/admin.html
```

生产入口：

```text
https://app.cmtbs.com:57890/
```

前端会根据“普通同传、高精同传、对话翻译、语音识别”自动选择 `/ws-standard` 或 `/ws-accurate`。普通用户应保持 WebSocket 地址为默认值，不需要选择 GPU 或直接连接后端端口。

## 仅用于静态页面调试

可以在 `web/` 目录启动已有环境中的静态文件服务器：

```bash
python3 -m http.server 8080
```

然后访问 `http://localhost:8080`。此方式只提供静态页面，仍需填写可访问的 WhisperLive 网关地址，并确保后端 CORS 允许该来源。

不要把远程后端 `9090` 当作面向用户的生产入口；远程访问应经过配置了 WebSocket 和 Admin API 转发的 Nginx 网关。

## 浏览器安全上下文

浏览器麦克风要求安全上下文：

- `localhost` 可使用 HTTP。
- 远程域名或 IP 通常需要 HTTPS 和有效证书。
- HTTPS 页面必须连接 `wss://`，不能连接 `ws://`。

修改前端 JavaScript 后只检查实际改动文件；前端静态改动不需要自行启动 Nginx、ASR 或 Admin API。
