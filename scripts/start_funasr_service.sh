#!/usr/bin/env bash
set -e

cd /app

python run_server.py \
  --port 9090 \
  --backend funasr \
  --funasr_mode paraformer_streaming \
  --funasr_model model/funasr/paraformer-zh-streaming \
  --funasr_final_model model/funasr/SenseVoiceSmall \
  --funasr_punc_model model/funasr/ct-punc \
  --funasr_device cuda \
  --max_clients 12 \
  --max_connection_time 600 \
  --translation_device cpu \
  --rest_port 8000 \
  --meeting_hotwords_dir config/hotwords.d \
  --meeting_logs_dir logs \
  --cors-origins http://ub.tuitukj.com:9093,http://localhost:9093,http://127.0.0.1:9093
