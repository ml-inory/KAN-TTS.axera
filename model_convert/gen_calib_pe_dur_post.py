#!/usr/bin/env python3
"""为 pitch_energy / duration / postnet 生成校准数据（104 句浮点中间张量）。"""
import os
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "/workspace")
from export_pe_dur_post import W, PitchEnergy, Duration  # noqa: E402

HW = os.environ["HW_DIR"]
WS = Path("/workspace")
OUT = WS / "export" / "calib_data"
MAX_T, MAX_M, NHEAD, DHEAD = 128, 270, 8, 16

# ---- 前端编码（复刻 C++ Frontend）----
_res = os.environ["RES_DIR"]
_xml = open(os.path.join(_res, "PinYin/PhoneSet.xml"), encoding="utf-8").read()
_phones = ["@" + n for n in re.findall(r"<name>([^<]+)</name>", _xml)]
for _i in range(1, 5):
    _phones.append("@#" + str(_i))
_tones = []
for _raw in open(os.path.join(_res, "PinYin/tonelist.txt"), encoding="utf-8"):
    _line = _raw.strip()
    _tones.append("tone_none" if not _line else "tone" + _line)
_SYLL = ["s_begin", "s_end", "s_none", "s_both", "s_middle"]
_WS = ["word_begin", "word_end", "word_middle", "word_both", "word_none"]
_EMO = ["emotion_none", "emotion_neutral", "emotion_angry", "emotion_disgust", "emotion_fear",
        "emotion_happy", "emotion_sad", "emotion_surprise", "emotion_calm", "emotion_gentle",
        "emotion_relax", "emotion_lyrical", "emotion_serious", "emotion_disgruntled",
        "emotion_satisfied", "emotion_disappointed", "emotion_excited", "emotion_anxiety",
        "emotion_jealousy", "emotion_hate", "emotion_pity", "emotion_pleasure", "emotion_arousal",
        "emotion_dominance", "emotion_placeholder1", "emotion_placeholder2", "emotion_placeholder3",
        "emotion_placeholder4", "emotion_placeholder5", "emotion_placeholder6", "emotion_placeholder7",
        "emotion_placeholder8", "emotion_placeholder9"]
_SPK = ["F7", "F74", "FBYN", "FRXL", "M7", "xiaoyu"]


def _mkv(items):
    return list(items) + ["_", "~", "@[MASK]"]


def encode_sym(symbol_seq):
    sy_v = _mkv(_phones)
    tone_v = _mkv(_tones)
    syll_v = _mkv(_SYLL)
    ws_v = _mkv(_WS)
    emo_v = _mkv(_EMO)
    spk_v = _mkv(_SPK)
    toks = symbol_seq.split()
    parts = [[] for _ in range(6)]
    for t in toks:
        inner = t[1:] if t.startswith("{") else t
        inner = inner[:-1] if inner.endswith("}") else inner
        fs = inner.split("$")
        for i in range(min(6, len(fs))):
            parts[i].append(fs[i])

    def enc_sy(parts_):
        ids = [sy_v.index("@" + p) if "@" + p in sy_v else 0 for p in parts_]
        ids.append(sy_v.index("~"))
        return ids

    def enc_cat(vocab, parts_):
        ids = [vocab.index(p) if p in vocab else 0 for p in parts_]
        ids.append(vocab.index("~"))
        return ids

    sy = enc_sy(parts[0])
    tone = enc_cat(tone_v, parts[1])
    syll = enc_cat(syll_v, parts[2])
    ws = enc_cat(ws_v, parts[3])
    emo = enc_cat(emo_v, parts[4])
    spk = enc_cat(spk_v, parts[5])
    T = len(sy) - 1
    ling = np.zeros((128, 4), np.int32)
    emo_a = np.zeros(128, np.int32)
    spk_a = np.zeros(128, np.int32)
    for i in range(T):
        ling[i] = [sy[i], tone[i], syll[i], ws[i]]
        emo_a[i] = emo[i]
        spk_a[i] = spk[i]
    return ling, emo_a, spk_a, T


def split_sym(sym, k=22):
    toks = sym.split()
    if len(toks) <= k:
        return [sym]
    has_end = toks[-1] == "~"
    if has_end:
        toks.pop()

    def is_punct(t):
        return t.startswith("{#")

    blocks, cur = [], []
    for t in toks:
        cur.append(t)
        if is_punct(t):
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    out, merged, mv = [], [], 0
    for blk in blocks:
        if merged and mv + len(blk) > k:
            out.append(" ".join(merged))
            merged, mv = [], 0
        merged += blk
        mv += len(blk)
    if merged:
        out.append(" ".join(merged))
    return out


def pack(name, samples):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        np.save(d / ("%04d.npy" % i), np.ascontiguousarray(s))
    with tarfile.open(OUT / ("%s.tar.gz" % name), "w:gz") as tar:
        for f in sorted(d.glob("*.npy")):
            tar.add(f, arcname=f.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = W(HW)
    pe = PitchEnergy(w).eval()
    dur = Duration(w).eval()
    enc = ort.InferenceSession("/workspace/export/am_enc.onnx", providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession("/workspace/export/am_dec.onnx", providers=["CPUExecutionProvider"])

    pw = torch.tensor(w.data["pitch_emb_w"])
    pb = torch.tensor(w.data["pitch_emb_b"])
    ew = torch.tensor(w.data["energy_emb_w"])
    eb = torch.tensor(w.data["energy_emb_b"])

    pe_in, dur_in, post_in = [], [], []
    lines = open(WS / "corpus_symbols.txt", encoding="utf-8").read().strip().splitlines()
    for text, sym in [l.split("\t", 1) for l in lines]:
        for seg in split_sym(sym):
            ling, emo_a, spk_a, T = encode_sym(seg)
            r = enc.run(None, {
                "inputs_ling": ling[None].astype(np.int32),
                "inputs_emo": emo_a[None].astype(np.int32),
                "inputs_spk": spk_a[None].astype(np.int32),
                "inputs_len": np.array([T], np.int32),
            })
            text_hid, spk_hid, emo_hid = r[0][0][:T], r[1][0][:T], r[2][0][:T]
            var_in = np.concatenate([text_hid, spk_hid, emo_hid], axis=-1)  # (T,96)
            var_pad = np.zeros((MAX_T, 96), np.float32)
            var_pad[:T] = var_in
            pe_in.append(var_pad[None])

            with torch.no_grad():
                p_full, e_full = pe(torch.tensor(var_pad[None]))
            p, e = p_full[0].numpy()[:T], e_full[0].numpy()[:T]

            def conv1d_same(x, wt, b):
                xi = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
                pad = (wt.shape[2] - 1) // 2
                y = torch.nn.functional.conv1d(xi, wt, bias=b, padding=pad)
                return y[0].transpose(0, 1).numpy()

            pe_c = conv1d_same(p, pw, pb)
            ee_c = conv1d_same(e, ew, eb)
            aug = text_hid + pe_c + ee_c
            cond = np.concatenate([aug, spk_hid, emo_hid], axis=-1)  # (T,96)
            cond22 = np.zeros((22, 96), np.float32)
            cond22[:T] = cond
            dur_in.append(cond22[None])

            with torch.no_grad():
                log_dur = dur(torch.tensor(cond22[None])).numpy()[0, :T]
            durations = np.exp(log_dur) - 1.0
            reps = np.round(durations).astype(int)
            reps = np.maximum(reps, 0)
            # 标点位置不强制停顿（与修复后管线一致）
            sum_r = int(reps.sum())
            pad = (3 - sum_r % 3) % 3
            P = sum_r + pad

            def expand(src, rr):
                dst = np.zeros((P, src.shape[1]), np.float32)
                pos = 0
                for t in range(T):
                    for _ in range(rr[t]):
                        dst[pos] = src[t]
                        pos += 1
                return dst

            lr_text = expand(aug, reps)
            lr_emo = expand(emo_hid, reps)
            lr_spk = expand(spk_hid, reps)
            rc = np.concatenate([[0], np.cumsum(reps)])
            lr_pos = np.zeros((P, 32), np.float32)
            for pp in range(P):
                ph = 0
                for t in range(T):
                    if rc[t] <= pp < rc[t + 1]:
                        ph = pp - rc[t] + 1
                        break
                for c in range(32):
                    inv = 10000.0 ** (2.0 * (c // 2) / 32.0)
                    v = ph / inv
                    lr_pos[pp, c] = np.sin(v) if c % 2 == 0 else np.cos(v)
            lr_text += lr_pos
            M = P // 3
            mem_rows = []
            for m in range(M):
                mem_rows.append(np.concatenate([
                    lr_text[m * 3:(m + 1) * 3].reshape(-1), lr_spk[m * 3], lr_emo[m * 3],
                ]))
            memory = np.stack(mem_rows)

            # decoder 循环（ORT QAT dec）
            mem_pad = np.zeros((MAX_M, 160), np.float32)
            mem_pad[:M] = memory
            xk = np.zeros((12, NHEAD, MAX_M, DHEAD), np.float32)
            xv = np.zeros((12, NHEAD, MAX_M, DHEAD), np.float32)
            frame = np.zeros((1, 1, 80), np.float32)
            x_band = int(max(durations) / 3.0 + 0.5)
            dec_all = np.zeros((M * 3, 80), np.float32)
            for s in range(M):
                out = dec.run(None, {
                    "mel_frame": frame,
                    "memory_step": mem_pad[s:s + 1][None],
                    "memory": mem_pad[None],
                    "x_k": xk[None],
                    "x_v": xv[None],
                    "step": np.array([[s]], np.int32),
                    "x_band": np.array([[x_band]], np.int32),
                    "h_band": np.array([[x_band]], np.int32),
                    "mem_len": np.array([[M]], np.int32),
                })
                dec_all[s * 3:(s + 1) * 3] = out[0][0, 0].reshape(3, 80)
                frame = out[0][:, :, -80:]
                for lidx in range(12):
                    kk = out[1 + lidx].reshape(NHEAD, 1, DHEAD)
                    vv = out[1 + 12 + lidx].reshape(NHEAD, 1, DHEAD)
                    xk[lidx, :, s:s + 1, :] = kk[:, 0:1, :]
                    xv[lidx, :, s:s + 1, :] = vv[:, 0:1, :]
            Tf = M * 3
            for start in range(0, Tf, 128):
                n = min(128, Tf - start)
                chunk = np.zeros((128, 80), np.float32)
                chunk[:n] = dec_all[start:start + n]
                post_in.append(chunk[None])

    pack("pe_var_in", pe_in)
    pack("dur_cond", dur_in)
    pack("post_dec", post_in)
    print("pe samples:", len(pe_in), "dur:", len(dur_in), "post:", len(post_in))


if __name__ == "__main__":
    main()
