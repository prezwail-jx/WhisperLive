# AGENTS.md

## 基本规则

- 始终使用简体中文回复。
- 修改前先阅读相关代码、脚本和现有文档，优先沿用当前项目风格。
- 不要回滚用户已有改动；如果工作区已有无关修改，保持原样。
- 不要随意重构无关模块，改动应聚焦当前需求。
- 不要提交或新增模型、日志、音频、压缩包、镜像包等大文件。
- 测试时不能自己拉起服务或安装依赖，除非用户明确要求；优先使用语法检查、静态检查等现有环境可执行的验证。

## 项目结构与职责

- `run_server.py` 是服务启动入口，新增服务参数时要同步检查 `whisper_live/server.py` 的 `TranscriptionServer.run()`。
- `whisper_live/` 是后端核心，包含 ASR、翻译、WebSocket 服务、Admin API、会议热词和会议日志逻辑。
- `web/` 是浏览器前端，只负责音频采集、实时展示、连接配置和下载后端生成的日志。
- 前端不要重新实现会议日志拼接、清洗、落盘；会议日志由后端 Python 按 session 记录。
- `scripts/start_whisper_service.sh` 和 `scripts/start_funasr_service.sh` 是服务拉起脚本，修改启动参数时优先同步这里。
- `deploy/` 存放部署和 Nginx 相关配置；除非需求明确，不要顺手改部署文件。

## 运行与部署约定

- 端口不要写死在新逻辑里，以启动脚本、`run_server.py` 参数、Nginx 配置和实际服务器开放端口为准。
- WebSocket、Admin API、Web/Nginx 入口可能在不同环境映射到不同端口；修改前先检查当前脚本和部署配置。
- `/ws`、`/admin/` 等路径转发规则以当前 Nginx 配置为准，不要假设固定公网端口。
- 如果涉及 Cloudflare、运营商反代或 HTTPS，只修改必要的服务参数和 Nginx 配置，不要把环境特定端口写进业务代码。

## 数据目录约定

- 会议热词目录通常由 `--meeting_hotwords_dir` 指定，默认习惯是 `config/hotwords.d`。
- 会议日志目录通常由 `--meeting_logs_dir` 指定，默认习惯是 `logs`。
- 模型通常放在 `model/`，不要把模型文件纳入 Git。
- `logs/`、`deploy/`、导出的会议文件、`.tar`、`.tar.gz`、音频样本和镜像包都要谨慎处理，提交前检查 `.gitignore` 和 `git status`。

## 当前会议日志设计

- 前端开始连接时生成 `session_id` 和 `session_started_at`，并通过 WebSocket 配置发给后端。
- 后端按 session 记录 ASR 原文片段和翻译片段，并生成 JSON 与 Markdown。
- 前端导出按钮只从后端下载 Markdown，不再本地拼接会议日志。
- 后端日志逻辑要尽量保持可追加、可去重，并方便后续接入大模型总结。
- 如果要改日志格式，优先保持 JSON 结构化数据可用，再调整 Markdown 渲染。

## ASR 与翻译注意事项

- ASR 边界去重、静音幻觉过滤、翻译缓冲去重属于低延迟保护逻辑，修改时要小心不要增加明显延迟。
- Whisper/faster-whisper、FunASR、翻译模型的设备选择由启动参数控制，不要在业务代码里硬编码 GPU 或 CPU。
- 多路并发性能问题优先从 batch、模型大小、VAD、翻译设备和前端测试方式排查，不要只看显存占用。
- 当前交错模式以原文片段为稳定主行，翻译只更新对应行；分栏和单语模式不要跟随交错模式需求一起重构。
- FunASR 最终精修失败时会退回流式文本。排查错词、残句时先搜索 `final refinement failed`、`CUDA out of memory` 和 `FUNASR_FINAL_REFINE`，再考虑调整识别参数。

## 近期已知环境与排查捷径

- 当前宿主机运行 FunASR/翻译单元测试时可能缺少 `numpy`、`torch`。若导入阶段明确报缺依赖，记录一次后停止，不重复运行、不安装依赖；改做 `py_compile`、`node --check` 和 `git diff --check`。
- 若工具报 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，这是执行沙箱故障，不是项目错误。普通命令和 `apply_patch` 各确认一次即可，随后使用已授权命令继续；写文件只能做精确局部替换，并立即检查 `git diff`。
- 查看容器日志时不要直接输出大段 `docker logs`。优先使用时间范围、关键词和 `tail`，例如筛选 `ERROR`、`FINAL_REFINE`、`SEGMENT_COMPLETE`、`CUDA out of memory`。
- `whisperlive-server` 容器的 PID 1 可能只是 `bash`，`docker restart` 后不会自动拉起 ASR。除非用户明确要求，不要自行重启；确需重启时先确认无活动客户端，并在重启后检查 Python 进程和 Admin API。
- 前端与容器通常通过目录挂载实时同步。修改前端后先比较宿主机与容器内文件哈希，不要无依据重建镜像或重启 Nginx。
- 识别效果分析优先查看后端 JSON/Markdown 原文和精修日志。前端显示、合并、滚动改动不会改变 ASR 模型输出。

## 验证命令

- 修改 Python 后优先运行：

```bash
python3 -m py_compile whisper_live/server.py run_server.py
```

- 修改前端后优先运行：

```bash
node --check web/app.js
```

- 修改测试后运行相关 unittest 或 pytest。若当前环境缺少 `fastapi`、`torch`、`pytest` 等依赖，要在回复中明确说明是环境依赖缺失。
- 测试若明确因缺少依赖、GPU、容器、系统工具或版本冲突而无法运行，确认一次原因后立即停止，不为同一环境问题反复尝试、启动临时容器或注入 stub。
- 不要为了通过验证临时安装、升级、降级或卸载项目依赖，除非用户明确要求处理环境；改用语法检查、静态检查等现有环境可执行的验证，并在最终回复中说明未运行的测试及原因。
- 输出验证结果时只汇报关键成功项和首个明确失败原因，不粘贴重复堆栈、进度条或大段容器日志。

## Git 与交付

- 提交前查看：

```bash
git status --short
git diff
```

- 不要使用 `git reset --hard`、`git checkout --` 等破坏性命令，除非用户明确要求。
- 如果远端和本地分支分叉，先说明差异和推荐合并方式，不要强推。
- 最终回复要说明改了什么、验证了什么、哪些验证因为环境原因没跑。
