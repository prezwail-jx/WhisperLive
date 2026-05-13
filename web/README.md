# WhisperLive Web Panel

这个目录是一个无需构建的浏览器前端，用当前电脑的麦克风连接 WhisperLive WebSocket server，并实时显示原文和中英自动翻译。

## 启动

在当前电脑上进入 `web` 目录并启动静态服务：

```bash
python -m http.server 8080
```

打开：

```text
http://localhost:8080
```

如果 server 在远程主机，把页面里的 Server 改成：

```text
ws://远程主机IP:9090
```

## 注意

浏览器麦克风需要安全上下文。`localhost` 
 localhost 域名，需要配置 HTTPS。
