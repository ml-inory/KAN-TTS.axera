#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kantts {

class ModelSession;
class EncOrt;

// 权重容器：读 models/host_weights/manifest.json + *.bin。
class Weights {
public:
    void Load(const std::string& dir);
    const std::vector<float>& Get(const std::string& name) const;
    const std::vector<int64_t>& Shape(const std::string& name) const;
    bool Has(const std::string& name) const { return data_.count(name) != 0; }

private:
    std::map<std::string, std::vector<float>> data_;
    std::map<std::string, std::vector<int64_t>> shapes_;
};

// 前端：符号串 → am_enc 输入（对齐 KanTtsLinguisticUnit）。
class Frontend {
public:
    explicit Frontend(const std::string& resource_dir, const std::string& am_config);
    // symbols: ttsfrd gen_tacotron_symbols 输出（如 "{b_c$tone3$...} {...}"）
    // 输出 4 个 int32 数组（ling/emo/spk 各 pad 到 128）+ 真实长度。
    struct EncInput {
        std::vector<int32_t> ling;  // 1*128*4
        std::vector<int32_t> emo;   // 1*128
        std::vector<int32_t> spk;   // 1*128
        std::vector<int32_t> len;   // 1
        int T;
    };
    EncInput Encode(const std::string& symbol_seq);

private:
    void BuildVocab();
    std::vector<std::string> phones_;
    std::vector<std::string> tones_;
    std::vector<std::string> syllable_flags_;
    std::vector<std::string> word_segments_;
    std::vector<std::string> emotion_types_;
    std::vector<std::string> speakers_;
};

// 完整 TTS 管线。
class KanttsPipeline {
public:
    KanttsPipeline(const std::string& model_dir, const std::string& resource_dir,
                   const std::string& am_config);
    ~KanttsPipeline();

    // symbols 列表（每句一个）→ 16kHz float32 波形（PCM 幅度）。
    std::vector<float> SynthesizeSymbols(const std::vector<std::string>& symbols);

private:
    std::unique_ptr<ModelSession> enc_, voc_, pe_, dur_, post_, dec_;
    std::string model_dir_;
    Weights w_;
    std::unique_ptr<Frontend> frontend_;
};

}  // namespace kantts
