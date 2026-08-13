# KAN-TTS · AX650 交付包（C++ SDK）

将 [modelscope/KAN-TTS](https://github.com/modelscope/KAN-TTS)（SamBERT + HiFi-GAN，中文 16k，发音人 zhitian_emo）部署到 AX650（NPU3）的 C++ 推理包。

## 架构（混合部署）

| 模块 | 位置 | 说明 |
|------|------|------|
| 前端（文本→符号） | 主机 x86 | ttsfrd，产出 symbols.txt |
| AM 编码器 enc | 板端 NPU | am_enc.axmodel（U16 PTQ） |
| 前端 embedding/位置编码 | 板端 CPU | 查表生成 x_emb/attn_mask/mask_f |
| 韵律/时长/PNCA 解码/Postnet | 板端 CPU | 自研 C++（float32） |
| 声码器 voc | 板端 NPU | voc.axmodel（INT8） |

> enc 曾尝试 INT8/SmoothQuant（text_hid cosine 0.99）与 QAT（0.9996，但 Pulsar2
> QuantONNX 导出有工具链缺陷无法编译）。最终采用剥离 embedding/mask 后的纯
> transformer 图 + U16 PTQ（text_hid cosine 0.9901），实际听感自然（RTF 0.58）。
> 句末自动拼接 0.3s 静音，避免尾音听不清。

## 目录

```
kantts-tts-ax650/
  model/            am_enc.axmodel + voc.axmodel + host_weights/ + resource/ + am_config.yaml
  sdk/              C++ 源码 + axrt 库（ax_engine/ax_sys）+ build.sh
  sdk/tools/        text_to_symbols.py（主机 x86，ttsfrd）
  example/
```

## 板端编译与运行

1. 拷贝整个包到 AX650 板，进入 `sdk/`：

   ```bash
   cd sdk && ./build.sh        # 产出 kantts_tts
   ```

2. 生成符号文件（在主机 x86 上）：

   ```bash
   python3 sdk/tools/text_to_symbols.py <resource_dir> "北京今天天气怎么样" symbols.txt
   ```

   `symbols.txt` 每行一个 ttsfrd `gen_tacotron_symbols` 输出（含 utt id 前缀亦可，程序自动跳过）。

3. 运行：

   ```bash
   LD_LIBRARY_PATH=/soc/lib ./sdk/kantts_tts model model/resource symbols.txt out.wav
   ```

   示例输出：`输出 out.wav（1.57s 音频，合成 1.09s，RTF=0.69）`

## 板端依赖

- AX650 / NPU3，`/soc/lib` 提供 ax_engine/ax_sys 运行时（或用包内 `sdk/axrt/lib`）
- `sdk/axrt/lib/libonnxruntime.so.1.23.2`（aarch64，来自 onnxruntime 1.23.2 wheel）

## 精度验证

（测试文本："北京今天天气怎么样"）

| 阶段 | 指标 |
|------|------|
| ling 编码 | 与官方 ttsfrd 逐项一致（T=22） |
| enc text_hid | cosine 1.0（CPU FP32） |
| durations / memory | 与官方一致（M=42，lr_len=126） |
| dec / postnet mel | cosine 1.0 / rms 0.8041=0.8139 |
| 最终 wav | cosine 0.973 / 频谱 0.996（voc INT8） |

## 示例音频

<audio controls src="example/out.wav"></audio>

[example/out.wav](example/out.wav)（"北京今天天气怎么样"，16kHz，句末含 0.3s 静音）

## 已知限制

- 前端（文本→符号）依赖主机 ttsfrd，不在板端运行；embedding/位置编码查表在板端 CPU
- enc 为 U16 PTQ（text_hid cosine 0.9901）、voc 为 INT8（0.973 cosine）；
  如需更高精度可走 QAT（见 qat/ 目录，待 Pulsar2 修复 QuantONNX 导出）
- 句末自动拼接 0.3s 静音（16k）
