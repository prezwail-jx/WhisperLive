#!/usr/bin/env bash
set -e

cd /app

python run_server.py \
  --port 9090 \
  --backend faster_whisper \
  --max_clients 12 \
  --max_connection_time 600 \
  --translation_device cuda \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --cors-origins http://ub.tuitukj.com:9093,http://localhost:9093,http://127.0.0.1:9093 \
  -fw model/asr/whisper-small-zh_tw-ct2/
