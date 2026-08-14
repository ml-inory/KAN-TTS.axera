#!/usr/bin/env python3
"""KAN-TTS NPU 模型 Pulsar2 编译驱动（NPU3，真实校准数据）。

交付架构：enc/pitch_energy/duration/postnet/voc 全 NPU，PNCA 解码（am_dec）走 CPU。
"""
import json
import sys
from pathlib import Path

import onnx

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "magnetar"))

from magnetar.docker_util import docker_pulsar2  # noqa: E402

IMAGE = "pulsar2:7.0"
TARGET = "AX650"
NPU_MODE = "NPU3"


def onnx_io(path):
    m = onnx.load(str(path))
    return ([i.name for i in m.graph.input], [o.name for o in m.graph.output])


def shapes_str(path):
    m = onnx.load(str(path))
    return ",".join(
        f"{i.name}:" + "x".join(str(d.dim_value) for d in i.type.tensor_type.shape.dim)
        for i in m.graph.input
    )


def build_config(name, onnx_name, calib_prefix, calib_size=4, use_input_shapes=True,
                 disable_opt=False, layer_configs=None):
    onnx_path = TASK / "export" / onnx_name
    ins, _ = onnx_io(onnx_path)
    input_configs = []
    for n in ins:
        input_configs.append({
            "tensor_name": n,
            "calibration_dataset": f"/workspace/export/calib_data/{calib_prefix}_{n}.tar.gz",
            "calibration_format": "Numpy",
            "calibration_size": calib_size,
            "calibration_mean": [],
            "calibration_std": [],
        })
    int_inputs = {"am_enc": ["inputs_ling", "inputs_emo", "inputs_spk", "inputs_len"]}.get(name, [])
    processors = []
    for n in ins:
        processors.append({
            "tensor_name": n,
            "tensor_format": "RGB",
            "tensor_layout": "NCHW",
            "src_format": "RGB",
            "src_dtype": "S32" if n in int_inputs else "FP32",
            "src_layout": "NCHW",
        })
    quant = {
        "input_configs": input_configs,
        "calibration_method": "MinMax",
        "precision_analysis": True,
        "precision_analysis_method": "EndToEnd",
        "highest_mix_precision": False,
    }
    if layer_configs:
        quant["layer_configs"] = layer_configs
    return {
        "input": f"/workspace/export/{onnx_name}",
        "output_dir": "/workspace/compile",
        "output_name": f"{name}.axmodel",
        "work_dir": f"/workspace/compile/work_{name}",
        "model_type": "ONNX",
        "target_hardware": TARGET,
        "npu_mode": NPU_MODE,
        "input_shapes": shapes_str(onnx_path) if use_input_shapes else "",
        "onnx_opt": {
            "disable_onnx_optimization": disable_opt,
            "enable_onnxsim": False,
            "model_check": True,
            "disable_transformation_check": True,
        },
        "quant": quant,
        "compiler": {"check": 3, "enable_tile_mode": name == "voc"},
        "input_processors": processors,
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    # 2026-08-14 重建：新校准集（104 句）+ 官方类导出 ONNX（pitch_energy/duration/postnet）
    models = [
        ("am_enc", "am_enc.onnx", "enc", 64),
        ("pitch_energy", "pitch_energy.onnx", "pe", 64),
        ("duration", "duration.onnx", "dur", 64),
        ("postnet", "postnet.onnx", "post", 64),
        ("voc", "voc.onnx", "voc", 4),
    ]
    for name, onnx_name, calib_prefix, calib_size in models:
        if only and name != only:
            continue
        cfg = build_config(name, onnx_name, calib_prefix, calib_size=calib_size,
                           use_input_shapes=False, disable_opt=True)
        cfg_path = TASK / "compile" / f"pulsar2_{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[compile] {name}: pulsar2 build --config compile/pulsar2_{name}.json")
        docker_pulsar2(
            IMAGE, str(TASK),
            f"pulsar2 build --config /workspace/compile/pulsar2_{name}.json",
            timeout=3600,
            log_file=TASK / "compile" / f"compile_{name}.log",
        )
        ax = TASK / "compile" / f"{name}.axmodel"
        if not ax.is_file():
            raise RuntimeError(f"{name} 编译未生成 axmodel，见 compile/compile_{name}.log")
        print(f"[compile] {name} OK: {ax.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
