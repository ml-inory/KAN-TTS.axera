#!/usr/bin/env python3
"""主机侧前端工具：中文文本 → 符号序列文件（供板端 kantts_tts 使用）。
依赖 ttsfrd（x86 Python3.10 可用）；板端无需此工具。
"""
import sys

import ttsfrd


def main():
    resource_dir = sys.argv[1]
    text = sys.argv[2]
    out = sys.argv[3]
    fe = ttsfrd.TtsFrontendEngine()
    assert fe.initialize(resource_dir)
    fe.set_lang_type("zh-cn")
    res = fe.gen_tacotron_symbols(text.strip())
    open(out, "w", encoding="utf-8").write(res)
    print(f"{out}: {len(res.strip().splitlines())} 句")


if __name__ == "__main__":
    main()
