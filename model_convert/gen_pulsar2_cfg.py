#!/usr/bin/env python3
"""生成 am_enc / am_dec / voc 的 Pulsar2 编译配置（新校准集）。"""
import json
from pathlib import Path

import onnx

TASK = Path(__file__).resolve().parents[1]  # 仓库根
IMAGE = "docker-registry.aitsw.axera-tech.com/pulsar2:6.0"
TARGET = "AX650"
NPU_MODE = "NPU3"


def onnx_io(path):
    m = onnx.load(str(path))
    return ([i.name for i in m.graph.input], [o.name for o in m.graph.output])


def shapes_str(path):
    m = onnx.load(str(path))
    return ",".join(
        "%s:" + "x".join(str(d.dim_value) for d in i.type.tensor_type.shape.dim)
        for i in m.graph.input
    )


def build_config(name, onnx_name, calib_prefix, calib_size, int_inputs=(),
                 disable_opt=True, fp32_bias=False, smooth_quant=False):
    onnx_path = TASK / "export" / onnx_name
    ins, _ = onnx_io(onnx_path)
    input_configs = []
    for n in ins:
        input_configs.append({
            "tensor_name": n,
            "calibration_dataset": "/workspace/export/calib_data/%s_%s.tar.gz" % (calib_prefix, n),
            "calibration_format": "Numpy",
            "calibration_size": calib_size,
            "calibration_mean": [],
            "calibration_std": [],
        })
    processors = []
    for n in ins:
        processors.append({
            "tensor_name": n,
            "tensor_layout": "NCHW",
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
    if fp32_bias:
        quant["conv_bias_data_type"] = "FP32"
        quant["disable_auto_refine_scale"] = True
    if smooth_quant:
        quant["enable_smooth_quant"] = True
    cfg = {
        "input": "/workspace/export/%s" % onnx_name,
        "output_dir": "/workspace/compile",
        "output_name": "%s.axmodel" % name,
        "work_dir": "/workspace/compile/work_%s" % name,
        "model_type": "ONNX",
        "target_hardware": TARGET,
        "npu_mode": NPU_MODE,
        "input_shapes": "",
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
    return cfg


def main():
    cfgs = {
        "am_enc": build_config("am_enc", "am_enc.onnx", "enc", 64,
                               int_inputs=["inputs_ling", "inputs_emo", "inputs_spk", "inputs_len"]),
        "am_dec": build_config("am_dec", "am_dec.onnx", "dec", 4,
                               int_inputs=["step", "x_band", "h_band", "mem_len"]),
        "voc": build_config("voc", "voc.onnx", "voc", 4),
        "pitch_energy": build_config("pitch_energy", "pitch_energy.onnx", "pe", 64),
        "duration": build_config("duration", "duration.onnx", "dur", 64,
                                 fp32_bias=True, smooth_quant=True),
        "postnet": build_config("postnet", "postnet.onnx", "post", 64,
                                fp32_bias=True, smooth_quant=True),
    }
    for name, cfg in cfgs.items():
        p = TASK / "compile" / ("pulsar2_%s.json" % name)
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("written", p)
    print(IMAGE)


if __name__ == "__main__":
    main()
