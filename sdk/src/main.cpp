#include "ax_engine.hpp"
#include "kantts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace kantts;

// duration 模型固定输入 22 帧：把超过 22 个有效符号的长句按标点切成子段，
// 每段独立走完整管线（enc/韵律/时长/解码/voc），音频按段拼接。
static const int kMaxDurT = 22;

static bool IsPunct(const std::string& tok) {
    return tok.size() >= 2 && tok[0] == '{' && tok[1] == '#';
}

static std::string JoinSymbols(const std::vector<std::string>& toks, bool with_end) {
    std::string s;
    for (const auto& t : toks) {
        if (!s.empty()) s += ' ';
        s += t;
    }
    if (with_end) {
        if (!s.empty()) s += ' ';
        s += "~";
    }
    return s;
}

static std::vector<std::string> SplitLongSentence(const std::string& line) {
    std::vector<std::string> toks;
    std::stringstream ss(line);
    std::string t;
    while (ss >> t) toks.push_back(t);
    if (toks.empty()) return {};
    // 末尾 ~ 结束符单独保留
    bool has_end = false;
    if (toks.back() == "~") {
        has_end = true;
        toks.pop_back();
    }
    // duration 模型的 T = Encode 后的符号数（含标点），上限 kMaxDurT
    int total_valid = (int)toks.size();
    if (total_valid <= kMaxDurT) {
        return {line};
    }
    // 按标点切块
    std::vector<std::vector<std::string>> blocks;
    std::vector<std::string> cur;
    for (const auto& s : toks) {
        cur.push_back(s);
        if (IsPunct(s)) {
            blocks.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) blocks.push_back(cur);
    // 贪心合并块，使每段有效符号 ≤ kMaxDurT
    std::vector<std::string> out;
    std::vector<std::string> merged;
    int merged_valid = 0;
    for (const auto& blk : blocks) {
        int blk_valid = (int)blk.size();
        if (!merged.empty() && merged_valid + blk_valid > kMaxDurT) {
            out.push_back(JoinSymbols(merged, has_end));
            merged.clear();
            merged_valid = 0;
        }
        merged.insert(merged.end(), blk.begin(), blk.end());
        merged_valid += blk_valid;
    }
    if (!merged.empty()) out.push_back(JoinSymbols(merged, has_end));
    return out;
}

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
            std::string sym = tab == std::string::npos ? line : line.substr(tab + 1);
            auto parts = SplitLongSentence(sym);
            symbols.insert(symbols.end(), parts.begin(), parts.end());
            if (parts.size() > 1) {
                std::fprintf(stderr, "[stage] 长句切分为 %zu 段\n", parts.size());
            }
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
