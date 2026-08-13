#!/usr/bin/env bash
# AX650 板端编译：kantts_tts <model_dir> <resource_dir> <symbols.txt> <out.wav>
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -ffast-math -std=c++17 \
    -I include -I axrt/include \
    src/ax_engine.cpp src/kantts.cpp src/main.cpp \
    -L axrt/lib -lax_engine -lax_sys -lpthread \
    -o kantts_tts
echo "[build] OK: $(pwd)/kantts_tts"
