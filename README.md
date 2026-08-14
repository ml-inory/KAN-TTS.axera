# KAN-TTS · AX650 交付包（C++ SDK）

将 [modelscope/KAN-TTS](https://github.com/modelscope/KAN-TTS)（SamBERT + HiFi-GAN，中文 16k，发音人 zhitian_emo）部署到 AX650（NPU3）的 C++ 推理包。

## 架构（全 NPU，除 PNCA 解码）

| 模块 | 位置 | 说明 |
|------|------|------|
| 前端（文本→符号） | 主机 x86 | ttsfrd，产出 symbols.txt |
| AM 编码器 enc（含 embedding/位置编码） | 板端 NPU | am_enc.axmodel（INT8，输入 ling/emo/spk/len） |
| 韵律 pitch/energy | 板端 NPU | pitch_energy.axmodel（INT8） |
| 时长 duration | 板端 NPU | duration.axmodel（INT8，非自回归单次前向） |
| PNCA 解码 dec | 板端 CPU | 自研 C++（float32，host_weights） |
| Postnet | 板端 NPU | postnet.axmodel（INT8） |
| 声码器 voc | 板端 NPU | voc.axmodel（INT8） |

> 校准集为 104 句真实中文语料（覆盖数字/长句/标点，含"八十万对六十万，优势在我"等），
> 由 pypinyin+jieba 生成 ttsfrd 风格符号（`tools/gen_calib.py`）。
> 关键修复：`ax_engine.cpp` 增加 `AX_SYS_MflushCache / AX_SYS_MinvalidateCache`，
> 修复多模型串行推理时缓存一致性问题（此前 NPU 输出逐次漂移）；
> 移除"标点强制停顿"补丁（该补丁会把停顿帧喂给解码器导致音频损坏）。
> 长句（>22 符号）按标点切分成 ≤22 子段，每段独立走全管线后拼接。
> 句末自动拼接 0.3s 静音，避免尾音听不清；语速默认 1.4x（`KANTTS_SPEED` 可调）。

## 目录

```
kantts-tts-ax650/
  model/            am_enc/pitch_energy/duration/postnet/voc.axmodel + host_weights/ + resource/ + am_config.yaml
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

   示例：`输出 out.wav（2.44s 音频，合成 0.72s，RTF=0.30）`（北京例句）

## 板端依赖

- AX650 / NPU3，`/soc/lib` 提供 ax_engine/ax_sys 运行时（或用包内 `sdk/axrt/lib`）
- `sdk/axrt/lib/libonnxruntime.so.1.23.2`（aarch64，来自 onnxruntime 1.23.2 wheel）

## 精度与效果验证（2026-08-14 重建后）

（测试文本："北京今天天气怎么样" / "八十万对六十万，优势在我！"）

| 阶段 | 指标 |
|------|------|
| ling 编码 | 与官方 ttsfrd 逐项一致（T=22） |
| enc text_hid | 板端 NPU vs 浮点 cosine 0.990 |
| 重建模型编译校验 | am_enc/pitch_energy/duration/voc 余弦 ≥ 0.9999 |
| 最终 wav | 北京句可识别；蒋介石台词 ASR = "80萬對60萬,優勢在我" |

## 已知限制

- 前端（文本→符号）依赖主机 ttsfrd，不在板端运行
- postnet 保留原始编译版本（新导出 ONNX 含 ConstantOfShape/Pad 组合，
  Pulsar2 无法编译，onnxsim 简化会破坏精度；原始模型精度正常）
- dec 为 CPU（PNCA），如需 NPU 解码可用 `KANTTS_NPU_DEC=1`（实验性，
  需要 am_dec.axmodel，QAT 导出在 Pulsar2 6.0 下因 per-channel DequantizeLinear 无法编译）
- 句末自动拼接 0.3s 静音（16k）
