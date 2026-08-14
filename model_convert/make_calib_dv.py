#!/usr/bin/env python3
"""为 am_dec / voc 生成校准数据（沿用官方参考的中间张量）。"""
import tarfile
from pathlib import Path

import numpy as np

OUT = Path("/data/yangrongzhao/kantts_recompile/export/calib_data")
MAX_M = 270


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
    z = np.load(Path("/data/yangrongzhao/kantts_recompile/export/ref_am.npz"), allow_pickle=False)
    utts = [{k[len(u) + 2:]: z[k] for k in z.files if k.startswith(u + "__")}
            for u in ["utt%d" % i for i in range(8)]]

    d0 = utts[0]
    mem = np.zeros((1, MAX_M, 160), np.float32)
    mem[:, : d0["memory"].shape[1]] = d0["memory"]
    M = d0["memory"].shape[1]
    xb = int(d0["x_band"][0])
    steps = [0, 10, 20, 30]
    dec_inputs = {n: [] for n in ["mel_frame", "memory_step", "memory", "x_k", "x_v",
                                  "step", "x_band", "h_band", "mem_len"]}
    frame = np.zeros((1, 1, 80), np.float32)
    for s in steps:
        if s >= M:
            continue
        st_k = np.zeros((1, 12, 8, MAX_M, 16), np.float32)
        st_v = np.zeros((1, 12, 8, MAX_M, 16), np.float32)
        for lidx in range(12):
            kk = d0["step_%d_l%d_k" % (s, lidx)]
            vv = d0["step_%d_l%d_v" % (s, lidx)]
            st_k[0, lidx, :, : s + 1, :] = kk[:, : s + 1, :]
            st_v[0, lidx, :, : s + 1, :] = vv[:, : s + 1, :]
        dec_inputs["mel_frame"].append(frame.astype(np.float32))
        dec_inputs["memory_step"].append(mem[:, s : s + 1].astype(np.float32))
        dec_inputs["memory"].append(mem.astype(np.float32))
        dec_inputs["x_k"].append(st_k.astype(np.float32))
        dec_inputs["x_v"].append(st_v.astype(np.float32))
        dec_inputs["step"].append(np.array([[s]], np.int32))
        dec_inputs["x_band"].append(np.array([[xb]], np.int32))
        dec_inputs["h_band"].append(np.array([[xb]], np.int32))
        dec_inputs["mem_len"].append(np.array([[M]], np.int32))
    for n, samples in dec_inputs.items():
        pack("dec_" + n, samples)

    voc_mels = []
    for d in utts:
        T = d["valid"][0]
        mel_full = d["postnet_out"][0, :T].T
        C, O = 200, 40
        start = 0
        while start < T:
            end = min(start + C, T)
            cs = max(0, start - O)
            chunk = np.zeros((1, 80, C), np.float32)
            chunk[:, :, : end - cs] = mel_full[:, cs:end]
            voc_mels.append(chunk)
            start = end
    pack("voc_mel", voc_mels)
    print("dec/voc calib done")


if __name__ == "__main__":
    main()
