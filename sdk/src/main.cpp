#include "ax_engine.hpp"
#include "kantts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace kantts;

static void WriteWav(const std::string& path, const std::vector<float>& audio, int sr = 16000) {
    std::vector<int16_t> pcm(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        float v = audio[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    std::ofstream f(path, std::ios::binary);
    auto wr = [&](const void* p, size_t n) { f.write((const char*)p, n); };
    uint32_t data = pcm.size() * 2;
    uint32_t rate = sr;
    uint16_t ch = 1, bits = 16;
    wr("RIFF", 4);
    uint32_t riff_size = 36 + data;
    wr(&riff_size, 4);
    wr("WAVEfmt ", 8);
    uint32_t hdr = 16;
    uint16_t fmt = 1;
    wr(&hdr, 4);
    wr(&fmt, 2);
    wr(&ch, 2);
    wr(&rate, 4);
    uint32_t bps = rate * ch * bits / 8;
    wr(&bps, 4);
    uint16_t ba = ch * bits / 8;
    wr(&ba, 2);
    wr(&bits, 2);
    wr("data", 4);
    wr(&data, 4);
    wr(pcm.data(), pcm.size() * 2);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "用法: kantts_tts <model_dir> <resource_dir> <symbols.txt> <out.wav>\n"
                     "symbols.txt: 每行一个 ttsfrd gen_tacotron_symbols 输出（见 tools/text_to_symbols.py）\n");
        return 1;
    }
    try {
        AxRuntimeInit();
        KanttsPipeline pipe(argv[1], argv[2], std::string(argv[1]) + "/am_config.yaml");
        std::ifstream sf(argv[3]);
        std::vector<std::string> symbols;
        std::string line;
        while (std::getline(sf, line)) {
            auto tab = line.find('\t');
            symbols.push_back(tab == std::string::npos ? line : line.substr(tab + 1));
        }
        auto t0 = std::chrono::steady_clock::now();
        auto audio = pipe.SynthesizeSymbols(symbols);
        auto t1 = std::chrono::steady_clock::now();
        WriteWav(argv[4], audio);
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double dur = audio.size() / 16000.0;
        std::printf("输出 %s（%.2fs 音频，合成 %.2fs，RTF=%.2f）\n", argv[4], dur, sec,
                    dur > 0 ? sec / dur : 0);
        AxRuntimeDeinit();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "错误: %s\n", e.what());
        return 1;
    }
    return 0;
}
