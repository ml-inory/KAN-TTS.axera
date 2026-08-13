#!/usr/bin/env python3
"""导出 KAN-TTS 三个 ONNX 子图并做 ORT 对分验证。

产物: export/am_enc.onnx / export/am_dec.onnx / export/voc.onnx
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK / "export"))

from am_models import AMDec, AMEnc, Voc, load_fsnet, MAX_M, MAX_MEL, MAX_T, NHEAD, DHEAD

AM_CKPT = TASK / "origin/speech_sambert-hifigan_tts_zh-cn_16k/voices/zhitian_emo/am/ckpt/checkpoint_0.pth"
VOC_CKPT = TASK / "origin/speech_sambert-hifigan_tts_zh-cn_16k/voices/zhitian_emo/voc/ckpt/checkpoint_1.pth"
VOC_CFG = TASK / "origin/speech_sambert-hifigan_tts_zh-cn_16k/voices/zhitian_emo/voc/config.yaml"
OUT = TASK / "export"


def cosine(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_refs():
    z = np.load(OUT / "ref_am.npz", allow_pickle=False)
    utts = {}
    for u in ["utt0", "utt1", "utt2", "utt3"]:
        d = {}
        for k in z.files:
            if k.startswith(u + "__"):
                d[k[len(u) + 2:]] = z[k]
        utts[u] = d
    return utts


def export_model(model, example_inputs, input_names, output_names, path, opset=17):
    last = None
    for ops in [opset, 13, 11]:
        try:
            torch.onnx.export(
                model, example_inputs, str(path),
                input_names=input_names, output_names=output_names,
                opset_version=ops, dynamo=False, do_constant_folding=True,
            )
            print(f"  [export ok] opset={ops} -> {path.name}")
            return ops
        except Exception as e:
            last = e
            print(f"  [export fail] opset={ops}: {str(e)[:300]}")
    raise RuntimeError(f"全部 opset 导出失败: {last}")


def check_static(path, use_simplify=True):
    import onnx
    from onnxsim import simplify
    m = onnx.load(str(path))
    if use_simplify:
        ms, ok = simplify(m)
        if ok:
            onnx.save(ms, str(path))
            m = ms
    onnx.checker.check_model(m)
    m = onnx.shape_inference.infer_shapes(m)
    dyn = []
    for vi in list(m.graph.input) + list(m.graph.output):
        for i, d in enumerate(vi.type.tensor_type.shape.dim):
            if not d.HasField("dim_value") or d.dim_value <= 0:
                dyn.append((vi.name, i))
    return not dyn, dyn


def validate_am_enc(ort_session, refs):
    print("  AMEnc 对分:")
    ok = True
    for u, d in refs.items():
        T = d["inputs_ling"].shape[1]
        ling = np.zeros((1, MAX_T, 4), np.int32)
        emo = np.zeros((1, MAX_T), np.int32)
        spk = np.zeros((1, MAX_T), np.int32)
        ling[:, :T] = d["inputs_ling"]
        emo[:, :T] = d["inputs_emo"]
        spk[:, :T] = d["inputs_spk"]
        length = np.array([T], np.int32)
        feed = {"inputs_ling": ling, "inputs_emo": emo, "inputs_spk": spk, "inputs_len": length}
        o = ort_session.run(None, feed)
        th, sh, eh = o[0], o[1], o[2]
        c = [
            cosine(th[0, :T], d["text_hid"][0]),
            cosine(sh[0, :T], d["spk_hid"][0]),
            cosine(eh[0, :T], d["emo_hid"][0]),
        ]
        print(f"    {u}: text={c[0]:.6f} spk={c[1]:.6f} emo={c[2]:.6f}")
        ok &= all(x >= 0.99 for x in c)
    return ok


def validate_am_dec(ort_session, refs):
    import onnxruntime as ort
    print("  AMDec 对分（逐帧循环复刻 host 逻辑）:")
    ok = True
    enc_sess = ort.InferenceSession(str(OUT / "am_enc.onnx"), providers=["CPUExecutionProvider"])
    for u, d in refs.items():
        T = d["inputs_ling"].shape[1]
        ling = np.zeros((1, MAX_T, 4), np.int32); ling[:, :T] = d["inputs_ling"]
        emo = np.zeros((1, MAX_T), np.int32); emo[:, :T] = d["inputs_emo"]
        spk = np.zeros((1, MAX_T), np.int32); spk[:, :T] = d["inputs_spk"]
        th, sh, eh = enc_sess.run(None, {"inputs_ling": ling, "inputs_emo": emo, "inputs_spk": spk,
                                         "inputs_len": np.array([T], np.int32)})
        # 用捕获的真实中间张量，避免依赖 numpy host 实现（host 实现单独验证）
        memory = np.zeros((1, MAX_M, 160), np.float32)
        memory[:, : d["memory"].shape[1]] = d["memory"]
        M = d["memory"].shape[1]
        x_band = d["x_band"][0]
        h_band = x_band
        xk = np.zeros((1, 12, NHEAD, MAX_M, DHEAD), np.float32)
        xv = np.zeros((1, 12, NHEAD, MAX_M, DHEAD), np.float32)
        dec_outs = []
        step = 0
        mel_frame = np.zeros((1, 1, 80), np.float32)
        all_cos = []
        while step < M:
            mem_step = memory[:, step:step + 1, :]
            feed = {"mel_frame": mel_frame, "memory_step": mem_step, "memory": memory,
                    "x_k": xk, "x_v": xv, "step": np.array([step], np.int32),
                    "x_band": np.array([x_band], np.int32), "h_band": np.array([h_band], np.int32),
                    "mem_len": np.array([M], np.int32)}
            o = ort_session.run(None, feed)
            dec_out = o[0]
            for lidx in range(12):
                kv = np.asarray(o[1 + lidx]).reshape(1, NHEAD, 1, DHEAD)
                xk[0, lidx, :, step, :] = kv[0, :, 0, :]
                vv = np.asarray(o[1 + 12 + lidx]).reshape(1, NHEAD, 1, DHEAD)
                xv[0, lidx, :, step, :] = vv[0, :, 0, :]
            ref_step = d[f"step_{step}_out"]
            c = cosine(dec_out, ref_step)
            all_cos.append(c)
            dec_outs.append(dec_out)
            mel_frame = dec_out[:, :, -80:]
            step += 1
        dec_stack = np.concatenate(dec_outs, axis=1).reshape(1, -1, 80)
        c_dec = cosine(dec_stack[:, : d["valid"][0]], d["dec_outputs"][0, : d["valid"][0]])
        c_min = min(all_cos)
        print(f"    {u}: M={M} 逐帧min={c_min:.6f} dec_valid cosine={c_dec:.6f}")
        ok &= c_min >= 0.99 and c_dec >= 0.99
    return ok


def validate_voc(ort_session, refs):
    import torch
    print("  Voc 对分:")
    ok = True
    voc = Voc(voc_ckpt=VOC_CKPT, voc_config=VOC_CFG)
    voc.eval()
    C, O = MAX_MEL, 40
    for u, d in refs.items():
        mel = d["postnet_out"][0, : d["valid"][0], :]  # (T,80)
        mel = np.ascontiguousarray(mel.T[np.newaxis], np.float32)  # (1,80,T)
        # 分块推理（块长 C，重叠 O）——与 SDK 一致
        wav_parts = []
        ref_parts = []
        start = 0
        T = mel.shape[2]
        while start < T:
            end = min(start + C, T)
            cs = max(0, start - O)
            chunk = np.zeros((1, 80, C), np.float32)
            chunk[:, :, : end - cs] = mel[:, :, cs:end]
            with torch.no_grad():
                ref_chunk = voc(torch.from_numpy(chunk)).numpy()
            out_chunk = ort_session.run(None, {"mel": chunk})[0]
            keep0 = (start - cs) * 200
            keepn = (end - start) * 200
            wav_parts.append(out_chunk[0, 0, keep0 : keep0 + keepn])
            ref_parts.append(ref_chunk[0, 0, keep0 : keep0 + keepn])
            start = end
        wav = np.concatenate(wav_parts)
        ref_wav = np.concatenate(ref_parts)
        n = T * 200
        c = cosine(wav[:n], ref_wav[:n])
        with torch.no_grad():
            ref_full = voc.gen(torch.from_numpy(mel)).numpy()[0, 0, :n]
        c_full = cosine(wav[:n], ref_full)
        print(f"    {u}: valid={d['valid'][0]} chunk_ort_vs_torch={c:.6f} chunk_vs_full={c_full:.6f}")
        ok &= c >= 0.99 and c_full >= 0.99
    return ok


def main():
    import onnxruntime as ort
    refs = load_refs()
    print("== AMEnc ==")
    fsnet = load_fsnet(AM_CKPT)
    enc = AMEnc(fsnet)
    enc_ex = (
        torch.zeros((1, MAX_T, 4), dtype=torch.int32),
        torch.zeros((1, MAX_T), dtype=torch.int32),
        torch.zeros((1, MAX_T), dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    ops1 = export_model(enc, enc_ex,
                        ["inputs_ling", "inputs_emo", "inputs_spk", "inputs_len"],
                        ["text_hid", "spk_hid", "emo_hid"], OUT / "am_enc.onnx")
    static, dyn = check_static(OUT / "am_enc.onnx")
    assert static, f"am_enc 动态维度: {dyn}"
    print("  am_enc 静态检查 OK")
    enc_sess = ort.InferenceSession(str(OUT / "am_enc.onnx"), providers=["CPUExecutionProvider"])
    ok1 = validate_am_enc(enc_sess, refs)

    print("== AMDec ==")
    dec = AMDec(fsnet)
    mem = torch.zeros((1, MAX_M, 160))
    dec_ex = (
        torch.zeros((1, 1, 80)),
        torch.zeros((1, 1, 160)),
        mem,
        torch.zeros((1, 12, NHEAD, MAX_M, DHEAD)),
        torch.zeros((1, 12, NHEAD, MAX_M, DHEAD)),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    out_names = ["dec_out"] + [f"k{l}" for l in range(12)] + [f"v{l}" for l in range(12)]
    ops2 = export_model(dec, dec_ex,
                        ["mel_frame", "memory_step", "memory", "x_k", "x_v", "step", "x_band", "h_band", "mem_len"],
                        out_names, OUT / "am_dec.onnx")
    static, dyn = check_static(OUT / "am_dec.onnx")
    assert static, f"am_dec 动态维度: {dyn}"
    print("  am_dec 静态检查 OK")
    dec_sess = ort.InferenceSession(str(OUT / "am_dec.onnx"), providers=["CPUExecutionProvider"])
    ok2 = validate_am_dec(dec_sess, refs)

    print("== Voc ==")
    voc = Voc(voc_ckpt=VOC_CKPT, voc_config=VOC_CFG)
    voc_ex = (torch.zeros((1, 80, MAX_MEL)),)
    ops3 = export_model(voc, voc_ex, ["mel"], ["wav"], OUT / "voc.onnx")
    static, dyn = check_static(OUT / "voc.onnx", use_simplify=False)
    assert static, f"voc 动态维度: {dyn}"
    print("  voc 静态检查 OK")
    voc_sess = ort.InferenceSession(str(OUT / "voc.onnx"), providers=["CPUExecutionProvider"])
    ok3 = validate_voc(voc_sess, refs)

    meta = {
        "model_name": "KAN-TTS (zhitian_emo)",
        "framework": "pytorch",
        "submodels": {
            "am_enc": {"file": "am_enc.onnx", "opset": ops1, "inputs": ["inputs_ling int32 (1,128,4)", "inputs_emo int32 (1,128)", "inputs_spk int32 (1,128)", "inputs_len int32 (1,)"],
                       "outputs": ["text_hid fp32 (1,128,32)", "spk_hid fp32 (1,128,32)", "emo_hid fp32 (1,128,32)"]},
            "am_dec": {"file": "am_dec.onnx", "opset": ops2, "inputs": ["mel_frame fp32 (1,1,80)", "memory_step fp32 (1,1,160)", "memory fp32 (1,270,160)", "x_k fp32 (1,12,8,270,16)", "x_v fp32 (1,12,8,270,16)", "step int32 (1,)", "x_band int32 (1,)", "h_band int32 (1,)", "mem_len int32 (1,)"],
                       "outputs": ["dec_out fp32 (1,1,240)", "k_steps (1,8,1,16)x12", "v_steps (1,8,1,16)x12"]},
            "voc": {"file": "voc.onnx", "opset": ops3, "inputs": ["mel fp32 (1,80,810)"], "outputs": ["wav fp32 (1,162000)"]},
        },
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n结论: am_enc={'OK' if ok1 else 'FAIL'} am_dec={'OK' if ok2 else 'FAIL'} voc={'OK' if ok3 else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
