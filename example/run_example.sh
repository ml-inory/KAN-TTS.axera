#!/usr/bin/env bash
# 板端示例：编译 SDK 并合成一句话。
set -e
cd "$(dirname "$0")/.."
cd sdk
./build.sh
cd ..
cat > example/symbols.txt <<'SYM'
0	{b_c$tone3$s_begin$word_begin$emotion_neutral$F7} {ei_c$tone3$s_end$word_middle$emotion_neutral$F7} {j_c$tone1$s_begin$word_middle$emotion_neutral$F7} {ing_c$tone1$s_end$word_end$emotion_neutral$F7} {#1$tone_none$s_none$word_none$emotion_none$F7} {j_c$tone1$s_begin$word_begin$emotion_neutral$F7} {in_c$tone1$s_end$word_middle$emotion_neutral$F7} {t_c$tone1$s_begin$word_middle$emotion_neutral$F7} {ian_c$tone1$s_end$word_end$emotion_neutral$F7} {#1$tone_none$s_none$word_none$emotion_none$F7} {t_c$tone1$s_begin$word_begin$emotion_neutral$F7} {ian_c$tone1$s_end$word_middle$emotion_neutral$F7} {q_c$tone4$s_begin$word_middle$emotion_neutral$F7} {i_c$tone4$s_end$word_end$emotion_neutral$F7} {#1$tone_none$s_none$word_none$emotion_none$F7} {z_c$tone3$s_begin$word_begin$emotion_neutral$F7} {en_c$tone3$s_end$word_middle$emotion_neutral$F7} {m_c$tone5$s_begin$word_middle$emotion_neutral$F7} {e_c$tone5$s_end$word_middle$emotion_neutral$F7} {y_c$tone4$s_begin$word_middle$emotion_neutral$F7} {ang_c$tone4$s_end$word_end$emotion_neutral$F7} {#4$tone_none$s_none$word_none$emotion_none$F7}
SYM
LD_LIBRARY_PATH=/soc/lib:sdk/axrt/lib ./sdk/kantts_tts model model/resource example/symbols.txt example/out.wav
echo "[example] 完成：example/out.wav"
