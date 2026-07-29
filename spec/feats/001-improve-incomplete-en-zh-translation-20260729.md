# 改善高精英译中残句漏译

2026-07-29

## 变更点

- 调整英译中翻译缓冲刷新顺序，避免 `translation_context_seconds` 在未完成英文结尾前提前触发翻译。
- 为未完成英文结尾增加独立等待预算，默认高精模式最多等待 4 秒。
- 将英译中漏译覆盖率阈值从 120% 调整为 100%，降低完整短译误报。
- 扩充英文未完成结尾短语识别，覆盖 `I started`、`this is how`、`I want really` 等跨段残句。
- 更新前端配置和 `app.js` 缓存版本，确保浏览器加载新参数。

## 测试计划

- 容器内对改动 Python 文件运行 `py_compile`。
- 运行 `tests.test_translation_backend` 和 `tests.test_server_extended`。
- 运行 `node --check web/app.js`，若容器缺少 node 则记录环境原因。
- 运行 `git diff --check` 并检查 `git status --short`。

## 假设与风险

- 未完成英文等待最多 4 秒，续句超过 4 秒才到达时仍会输出当前最佳译文和 warning。
- 不新增模型实例、不增加并发模型调用，显存峰值不因本改动扩大。
- 浏览器若未刷新缓存，仍可能使用旧 `app.js` 参数。

## 结果总结

- 已调整翻译缓冲刷新顺序，未完成英文结尾会优先等待，不再被 `translation_context_seconds` 提前刷新。
- 已新增 `translation_incomplete_max_wait_seconds` 参数，高精前端发送 4 秒，服务端透传到翻译客户端。
- 已扩充英文残句短语检测，并支持三词及以上结尾短语。
- 已将英译中漏译覆盖率阈值从 120% 调整为 100%。
- 已更新 `web/index.html` 中 `app.js` 版本参数，降低浏览器继续使用旧参数的风险。
- 验证已通过：Python `py_compile`、`tests.test_translation_backend`、`tests.test_server_extended`、`git diff --check`。
- `node --check web/app.js` 未执行成功，原因是容器内缺少 `node`。

## 后续跟进

- 待部署机真实音频复测后确认是否还存在 10 秒旧配置或残句漏译。
- 复测时检查 WebSocket 首包是否包含 `translation_incomplete_max_wait_seconds: 4.0`。
- 复测时重点核对 `we really feel that we are / at the frontier` 是否合并翻译。
