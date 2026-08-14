# KAN-TTS · AX650 交付包（C++ SDK）

将 [modelscope/KAN-TTS](https://github.com/modelscope/KAN-TTS)（SamBERT + HiFi-GAN，中文 16k，发音人 zhitian_emo）部署到 AX650（NPU3）的 C++ 推理包。

## 架构（除 PNCA 解码外全 NPU）

| 模块 | 位置 | 说明 |
|------|------|------|
| 前端（文本→符号） | 主机 x86 | ttsfrd，产出 symbols.txt |
| AM 编码器 enc | 板端 NPU | am_enc.axmodel（U16 PTQ） |
| 前端 embedding/位置编码 | 板端 CPU | 查表生成 x_emb/attn_mask/mask_f |
| 韵律 pitch/energy | 板端 NPU | pitch_energy.axmodel |
| 时长 duration | 板端 NPU | duration.axmodel |
| PNCA 解码 | 板端 CPU | 自研 C++（float32，保持精度） |
| Postnet | 板端 NPU | postnet.axmodel |
| 声码器 voc | 板端 NPU | voc.axmodel（INT8） |

> am_dec（PNCA 自回归解码）曾尝试 PTQ 与 QAT 上 NPU：PTQ AR 循环精度不稳定；
> QAT（S16 对称 + 闭环序列训练）单步 dec_out cosine 0.979，但 kv 状态误差在
> 42 步 AR 循环中累积，闭环 mel 达不到可交付听感，故解码保持 CPU（float32）。
> 其余模块（pitch/energy/duration/postnet）NPU 化已板端验证无质量退化
> （对齐后 mel cosine 0.98，RTF 0.32）。
> 句末自动拼接 0.3s 静音，避免尾音听不清。

## 目录

```
kantts-tts-ax650/
  model/            am_enc/pitch_energy/duration/postnet/voc .axmodel + host_weights/ + resource/ + am_config.yaml
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

   默认配置：enc/韵律/时长/postnet/voc 走 NPU，PNCA 解码走 CPU。
   示例输出：`输出 out.wav（1.84s 音频，合成 0.58s，RTF=0.32）`

   环境变量开关（实验）：
   - `KANTTS_NPU_DEC=1`：PNCA 解码也走 NPU（am_dec.axmodel，QAT 实验产物，精度未达交付标准）
   - `KANTTS_CPU_PRED=1`：韵律/时长回退 CPU（float32）

## 板端依赖

- AX650 / NPU3，`/soc/lib` 提供 ax_engine/ax_sys 运行时（或用包内 `sdk/axrt/lib`）
- `sdk/axrt/lib/libonnxruntime.so.1.23.2`（aarch64，来自 onnxruntime 1.23.2 wheel）

## 精度验证

（测试文本："北京今天天气怎么样"）

| 模块 | 板端对分 |
|------|------|
| enc text_hid | cosine 0.9901（U16 PTQ，听感自然） |
| pitch / energy | cosine 0.996 / 0.962 |
| log_dur | cosine 0.9999 |
| postnet mel | cosine 0.993 |
| 整链 mel（CPU vs NPU 预测器，DTW 对齐） | cosine 0.982 |
| voc（INT8） | wav cosine 0.973 |
| RTF | 0.32（解码 CPU，其余 NPU） |

## 示例音频

<audio controls src="example/out.wav"></audio>

[example/out.wav](example/out.wav)（"北京今天天气怎么样"，16kHz，句末含 0.3s 静音）

## 已知限制

- 前端（文本→符号）依赖主机 ttsfrd，不在板端运行；embedding/位置编码查表在板端 CPU
- PNCA 解码在 CPU（float32），其余全 NPU；RTF 0.32
- 时长 NPU 化后个别音节帧数可能与 CPU 参考差 1（取整敏感），听感为节奏微差、内容一致
- enc 为 U16 PTQ、voc 为 INT8；如需更高精度可走 QAT（见 qat/ 目录）
- 句末自动拼接 0.3s 静音（16k）
