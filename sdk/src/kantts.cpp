#include "kantts.hpp"

#include "ax_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <arm_neon.h>

namespace kantts {
namespace {

std::vector<float> LoadBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() % 4 != 0) throw std::runtime_error("bad bin size " + path);
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

void Matmul(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& b,
            int n, int in_d, int out_d, std::vector<float>& y) {
    y.assign(n * out_d, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* xp = x.data() + i * in_d;
        for (int o = 0; o < out_d; ++o) {
            float acc = b.empty() ? 0.0f : b[o];
            const float* wp = w.data() + o * in_d;
            int k = 0;
            float32x4_t v = vdupq_n_f32(0.0f);
            for (; k + 4 <= in_d; k += 4)
                v = vfmaq_f32(v, vld1q_f32(xp + k), vld1q_f32(wp + k));
            acc += vaddvq_f32(v);
            for (; k < in_d; ++k) acc += xp[k] * wp[k];
            y[i * out_d + o] = acc;
        }
    }
}

void Matmul(const float* x, const float* w, const float* b, int n, int in_d, int out_d,
            std::vector<float>& y) {
    y.assign(n * out_d, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* xp = x + i * in_d;
        for (int o = 0; o < out_d; ++o) {
            float acc = b ? b[o] : 0.0f;
            const float* wp = w + o * in_d;
            int k = 0;
            float32x4_t v = vdupq_n_f32(0.0f);
            for (; k + 4 <= in_d; k += 4)
                v = vfmaq_f32(v, vld1q_f32(xp + k), vld1q_f32(wp + k));
            acc += vaddvq_f32(v);
            for (; k < in_d; ++k) acc += xp[k] * wp[k];
            y[i * out_d + o] = acc;
        }
    }
}

void LayerNorm(const std::vector<float>& x, const std::vector<float>& g,
               const std::vector<float>& b, int n, int d, std::vector<float>& y) {
    y.resize(n * d);
    for (int i = 0; i < n; ++i) {
        float mean = 0, var = 0;
        for (int k = 0; k < d; ++k) mean += x[i * d + k];
        mean /= d;
        for (int k = 0; k < d; ++k) var += (x[i * d + k] - mean) * (x[i * d + k] - mean);
        var /= d;
        float inv = 1.0f / std::sqrt(var + 1e-6f);
        for (int k = 0; k < d; ++k) y[i * d + k] = (x[i * d + k] - mean) * inv * g[k] + b[k];
    }
}

void LayerNorm(const float* x, const float* g, const float* b, int n, int d,
               std::vector<float>& y) {
    y.resize(n * d);
    for (int i = 0; i < n; ++i) {
        float mean = 0, var = 0;
        for (int k = 0; k < d; ++k) mean += x[i * d + k];
        mean /= d;
        for (int k = 0; k < d; ++k) var += (x[i * d + k] - mean) * (x[i * d + k] - mean);
        var /= d;
        float inv = 1.0f / std::sqrt(var + 1e-6f);
        for (int k = 0; k < d; ++k) y[i * d + k] = (x[i * d + k] - mean) * inv * g[k] + b[k];
    }
}

void Conv1dSame(const std::vector<float>& x, const std::vector<float>& wgt,
                const std::vector<float>& bias, int T, int C, int O, int K,
                std::vector<float>& y) {
    int pad = (K - 1) / 2;
    y.assign(T * O, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < O; ++o) {
            float acc = bias.empty() ? 0.0f : bias[o];
            for (int k = 0; k < K; ++k) {
                int tt = t - pad + k;
                if (tt < 0 || tt >= T) continue;
                for (int c = 0; c < C; ++c)
                    acc += x[tt * C + c] * wgt[(o * C + c) * K + k];
            }
            y[t * O + o] = acc;
        }
}

void DepthwiseShift(const std::vector<float>& x, const std::vector<float>& wgt,
                    int T, int C, int K, int lp, int rp, std::vector<float>& y) {
    y.assign(T * C, 0.0f);
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < T; ++t) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                int tt = t - lp + k;
                if (tt < 0 || tt >= T) continue;
                acc += x[tt * C + c] * wgt[c * K + k];
            }
            y[t * C + c] = acc;
        }
}

void LstmCell(const std::vector<float>& x, const std::vector<float>& w_ih,
              const std::vector<float>& w_hh, const std::vector<float>& b_ih,
              const std::vector<float>& b_hh, std::vector<float>& h, std::vector<float>& c,
              int units) {
    std::vector<float> gates(4 * units, 0.0f);
    for (int g = 0; g < 4 * units; ++g) {
        float acc = b_ih[g] + b_hh[g];
        for (int k = 0; k < (int)x.size(); ++k) acc += x[k] * w_ih[g * x.size() + k];
        for (int k = 0; k < units; ++k) acc += h[k] * w_hh[g * units + k];
        gates[g] = acc;
    }
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    for (int u = 0; u < units; ++u) {
        float i = sig(gates[u]);
        float f = sig(gates[units + u]);
        float g = std::tanh(gates[2 * units + u]);
        float o = sig(gates[3 * units + u]);
        c[u] = f * c[u] + i * g;
        h[u] = o * std::tanh(c[u]);
    }
}

void Blstm(const std::vector<float>& x, const Weights& w, const std::string& pre,
           int T, int in_d, int units, std::vector<float>& y) {
    const auto& wih = w.Get(pre + "_blstm_w_ih");
    const auto& whh = w.Get(pre + "_blstm_w_hh");
    const auto& bih = w.Get(pre + "_blstm_b_ih");
    const auto& bhh = w.Get(pre + "_blstm_b_hh");
    const auto& wihr = w.Get(pre + "_blstm_w_ih_r");
    const auto& whhr = w.Get(pre + "_blstm_w_hh_r");
    const auto& bihr = w.Get(pre + "_blstm_b_ih_r");
    const auto& bhhr = w.Get(pre + "_blstm_b_hh_r");
    std::vector<float> hf(units, 0), cf(units, 0), hb(units, 0), cb(units, 0);
    y.assign(T * 2 * units, 0.0f);
    std::vector<float> xi(in_d);
    for (int t = 0; t < T; ++t) {
        std::copy(x.begin() + t * in_d, x.begin() + (t + 1) * in_d, xi.begin());
        LstmCell(xi, wih, whh, bih, bhh, hf, cf, units);
        std::copy(hf.begin(), hf.end(), y.begin() + t * 2 * units);
    }
    for (int t = T - 1; t >= 0; --t) {
        std::copy(x.begin() + t * in_d, x.begin() + (t + 1) * in_d, xi.begin());
        LstmCell(xi, wihr, whhr, bihr, bhhr, hb, cb, units);
        std::copy(hb.begin(), hb.end(), y.begin() + t * 2 * units + units);
    }
}

std::vector<float> FsmnEncoder(const std::vector<float>& x, const Weights& w,
                               const std::string& pre, int T, int C, const std::vector<int>& shift) {
    std::vector<float> cur = x;
    int layers = 0;
    while (w.Has(pre + "_ffn" + std::to_string(layers) + "_w1")) ++layers;
    for (int i = 0; i < layers; ++i) {
        int mid = (int)w.Shape(pre + "_ffn" + std::to_string(i) + "_w1")[0];
        std::vector<float> c1;
        Conv1dSame(cur, w.Get(pre + "_ffn" + std::to_string(i) + "_w1"),
                   w.Get(pre + "_ffn" + std::to_string(i) + "_b1"), T, C, mid, 1, c1);
        for (auto& v : c1) v = std::max(v, 0.0f);
        int out_c = (int)w.Shape(pre + "_ffn" + std::to_string(i) + "_w2")[0];
        std::vector<float> c2;
        Conv1dSame(c1, w.Get(pre + "_ffn" + std::to_string(i) + "_w2"),
                   w.Get(pre + "_ffn" + std::to_string(i) + "_b2"), T, mid, out_c, 1, c2);
        int fsize = (int)w.Shape(pre + "_mem" + std::to_string(i) + "_conv")[2];
        int sh = shift.empty() ? 0 : shift[i];
        int lp = (fsize - 1) / 2 + (sh > 0 ? sh : 0);
        int rp = (fsize - 1) / 2 - (sh > 0 ? sh : 0);
        std::vector<float> mem;
        DepthwiseShift(c2, w.Get(pre + "_mem" + std::to_string(i) + "_conv"), T, out_c, fsize,
                       lp, rp, mem);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < out_c; ++c) mem[t * out_c + c] += c2[t * out_c + c];
        if (out_c == C)
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < out_c; ++c) mem[t * out_c + c] += cur[t * C + c];
        cur = mem;
        C = out_c;
    }
    return cur;
}

std::vector<float> VarFsmnRnnPredictor(const std::vector<float>& x, const Weights& w,
                                       const std::string& pre, int T, int in_d) {
    std::vector<float> h = FsmnEncoder(x, w, pre, T, in_d, {0, 0, 0});
    std::vector<float> bh;
    Blstm(h, w, pre, T, 128, 128, bh);
    std::vector<float> out(T);
    const auto& fw = w.Get(pre + "_fc_w");
    const auto& fb = w.Get(pre + "_fc_b");
    for (int t = 0; t < T; ++t) {
        float acc = fb[0];
        for (int k = 0; k < 256; ++k) acc += bh[t * 256 + k] * fw[0 * 256 + k];
        out[t] = acc;
    }
    return out;
}

std::vector<float> DurationAr(const std::vector<float>& cond, const Weights& w, int T, int in_d) {
    std::vector<float> h0(128, 0), c0(128, 0), h1(128, 0), c1(128, 0);
    std::vector<float> x(1, 0.0f), out(T);
    const auto& p0w = w.Get("dur_pre0_w");
    const auto& p0b = w.Get("dur_pre0_b");
    const auto& p1w = w.Get("dur_pre1_w");
    const auto& p1b = w.Get("dur_pre1_b");
    const auto& fw = w.Get("dur_fc_w");
    const auto& fb = w.Get("dur_fc_b");
    std::vector<float> inp, tmp;
    for (int t = 0; t < T; ++t) {
        Matmul(x, p0w, p0b, 1, 1, 128, inp);
        for (auto& v : inp) v = std::max(v, 0.0f);
        Matmul(inp, p1w, p1b, 1, 128, 128, tmp);
        for (auto& v : tmp) v = std::max(v, 0.0f);
        std::vector<float> xin(tmp);
        xin.insert(xin.end(), cond.begin() + t * in_d, cond.begin() + (t + 1) * in_d);
        LstmCell(xin, w.Get("dur_lstm_w_ih0"), w.Get("dur_lstm_w_hh0"), w.Get("dur_lstm_b_ih0"),
                 w.Get("dur_lstm_b_hh0"), h0, c0, 128);
        LstmCell(h0, w.Get("dur_lstm_w_ih1"), w.Get("dur_lstm_w_hh1"), w.Get("dur_lstm_b_ih1"),
                 w.Get("dur_lstm_b_hh1"), h1, c1, 128);
        float acc = fb[0];
        for (int k = 0; k < 128; ++k) acc += h1[k] * fw[0 * 128 + k];
        x[0] = std::max(acc, 0.0f);
        out[t] = x[0];
    }
    return out;
}

}  // namespace

void Weights::Load(const std::string& dir) {
    std::ifstream f(dir + "/manifest.json");
    if (!f) throw std::runtime_error("missing weights manifest");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    std::regex entry_re("\"([A-Za-z0-9_]+)\"\\s*:\\s*\\[([0-9,\\s]*)\\]");
    for (std::sregex_iterator it(text.begin(), text.end(), entry_re), end; it != end; ++it) {
        std::string name = it->str(1);
        std::vector<int64_t> shape;
        std::stringstream dims(it->str(2));
        std::string d;
        while (std::getline(dims, d, ',')) {
            d.erase(std::remove_if(d.begin(), d.end(), ::isspace), d.end());
            if (!d.empty()) shape.push_back(std::stoll(d));
        }
        data_[name] = LoadBin(dir + "/" + name + ".bin");
        shapes_[name] = shape;
    }
}

const std::vector<float>& Weights::Get(const std::string& name) const {
    auto it = data_.find(name);
    if (it == data_.end()) throw std::runtime_error("missing weight " + name);
    return it->second;
}

const std::vector<int64_t>& Weights::Shape(const std::string& name) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) throw std::runtime_error("missing shape " + name);
    return it->second;
}

Frontend::Frontend(const std::string& resource_dir, const std::string& am_config) {
    // phones from PhoneSet.xml
    std::ifstream pf(resource_dir + "/PinYin/PhoneSet.xml");
    if (!pf) throw std::runtime_error("cannot open PhoneSet.xml");
    std::string xml((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    std::regex name_re("<name>([^<]+)</name>");
    for (std::sregex_iterator it(xml.begin(), xml.end(), name_re), end; it != end; ++it)
        phones_.push_back("@" + it->str(1));
    // 官方 parse_phoneset 在 PhoneSet.xml 音素后追加 #1..#4（静音/停顿标记）
    for (int i = 1; i <= 4; ++i) phones_.push_back("@#" + std::to_string(i));
    std::ifstream tf(resource_dir + "/PinYin/tonelist.txt");
    std::string line;
    while (std::getline(tf, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        tones_.push_back(line.empty() ? "tone_none" : "tone" + line);
    }
    syllable_flags_ = {"s_begin", "s_end", "s_none", "s_both", "s_middle"};
    word_segments_ = {"word_begin", "word_end", "word_middle", "word_both", "word_none"};
    emotion_types_ = {
        "emotion_none", "emotion_neutral", "emotion_angry", "emotion_disgust", "emotion_fear",
        "emotion_happy", "emotion_sad", "emotion_surprise", "emotion_calm", "emotion_gentle",
        "emotion_relax", "emotion_lyrical", "emotion_serious", "emotion_disgruntled",
        "emotion_satisfied", "emotion_disappointed", "emotion_excited", "emotion_anxiety",
        "emotion_jealousy", "emotion_hate", "emotion_pity", "emotion_pleasure", "emotion_arousal",
        "emotion_dominance", "emotion_placeholder1", "emotion_placeholder2", "emotion_placeholder3",
        "emotion_placeholder4", "emotion_placeholder5", "emotion_placeholder6",
        "emotion_placeholder7", "emotion_placeholder8", "emotion_placeholder9"};
    std::ifstream cf(am_config);
    while (std::getline(cf, line)) {
        if (line.find("speaker_list") != std::string::npos) {
            auto pos = line.find(':');
            std::string list = line.substr(pos + 1);
            std::stringstream ls(list);
            std::string s;
            while (std::getline(ls, s, ',')) {
                s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
                if (!s.empty()) speakers_.push_back(s);
            }
            break;
        }
    }
}

static std::vector<std::string> MakeVocab(const std::vector<std::string>& items) {
    std::vector<std::string> v;  // list("") == []，无前导空串
    v.insert(v.end(), items.begin(), items.end());
    v.push_back("_");
    v.push_back("~");
    v.push_back("@[MASK]");
    return v;
}

static void EncodeCategory(const std::vector<std::string>& vocab,
                           const std::vector<std::string>& parts, std::vector<int>& ids) {
    for (const auto& p : parts) {
        auto it = std::find(vocab.begin(), vocab.end(), p);
        ids.push_back(it == vocab.end() ? 0 : (int)(it - vocab.begin()));
    }
    auto eit = std::find(vocab.begin(), vocab.end(), "~");
    ids.push_back((int)(eit - vocab.begin()));
}

static void EncodeSy(const std::vector<std::string>& vocab,
                     const std::vector<std::string>& parts, std::vector<int>& ids) {
    for (const auto& p : parts) {
        std::string key = "@" + p;  // 原版按 ARPAbet 处理：{x} → "@x"
        auto it = std::find(vocab.begin(), vocab.end(), key);
        if (it != vocab.end()) ids.push_back((int)(it - vocab.begin()));
    }
    auto eit = std::find(vocab.begin(), vocab.end(), "~");
    ids.push_back((int)(eit - vocab.begin()));
}

Frontend::EncInput Frontend::Encode(const std::string& symbol_seq) {
    auto sy_v = MakeVocab(phones_);
    auto tone_v = MakeVocab(tones_);
    auto syll_v = MakeVocab(syllable_flags_);
    auto ws_v = MakeVocab(word_segments_);
    auto emo_v = MakeVocab(emotion_types_);
    auto spk_v = MakeVocab(speakers_);
    std::vector<std::string> tokens;
    {
        std::stringstream ss(symbol_seq);
        std::string t;
        while (ss >> t) tokens.push_back(t);
    }
    std::vector<std::vector<std::string>> parts(6);
    for (const auto& t : tokens) {
        std::string inner = t;
        if (!inner.empty() && inner.front() == '{') inner.erase(inner.begin());
        if (!inner.empty() && inner.back() == '}') inner.pop_back();
        std::stringstream ps(inner);
        std::string p;
        int idx = 0;
        while (std::getline(ps, p, '$') && idx < 6) parts[idx++].push_back(p);
    }
    std::vector<int> sy, tone, syll, ws, emo, spk;
    EncodeSy(sy_v, parts[0], sy);
    EncodeCategory(tone_v, parts[1], tone);
    EncodeCategory(syll_v, parts[2], syll);
    EncodeCategory(ws_v, parts[3], ws);
    EncodeCategory(emo_v, parts[4], emo);
    EncodeCategory(spk_v, parts[5], spk);
    int T = (int)sy.size() - 1;  // 去掉末尾 ~
    const int MT = 128;
    EncInput out;
    out.ling.assign(MT * 4, 0);
    out.emo.assign(MT, 0);
    out.spk.assign(MT, 0);
    for (int i = 0; i < T; ++i) {
        out.ling[i * 4 + 0] = sy[i];
        out.ling[i * 4 + 1] = tone[i];
        out.ling[i * 4 + 2] = syll[i];
        out.ling[i * 4 + 3] = ws[i];
        out.emo[i] = emo[i];
        out.spk[i] = spk[i];
    }
    out.len = {T};
    out.T = T;
    return out;
}

namespace {

// build_memory：text/spk/emo (T,32) → memory (M,160) + lr_len + durations
void BuildMemory(const std::vector<float>& text_hid, const std::vector<float>& spk_hid,
                 const std::vector<float>& emo_hid, const Weights& w, int T,
                 std::vector<float>& memory, int& lr_len, std::vector<float>& durations) {
    std::vector<float> var_in(T * 96);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 32; ++c) {
            var_in[t * 96 + c] = text_hid[t * 32 + c];
            var_in[t * 96 + 32 + c] = spk_hid[t * 32 + c];
            var_in[t * 96 + 64 + c] = emo_hid[t * 32 + c];
        }
    auto pitch = VarFsmnRnnPredictor(var_in, w, "pitch", T, 96);
    auto energy = VarFsmnRnnPredictor(var_in, w, "energy", T, 96);
    if (std::getenv("KANTTS_DUMP_ENC")) {
        {
            std::ofstream f("/tmp/kt/enc_pitch.bin", std::ios::binary);
            f.write((const char*)pitch.data(), pitch.size() * 4);
        }
        {
            std::ofstream f("/tmp/kt/enc_energy.bin", std::ios::binary);
            f.write((const char*)energy.data(), energy.size() * 4);
        }
    }
    std::vector<float> pe, ee;
    Conv1dSame(pitch, w.Get("pitch_emb_w"), w.Get("pitch_emb_b"), T, 1, 32, 9, pe);
    Conv1dSame(energy, w.Get("energy_emb_w"), w.Get("energy_emb_b"), T, 1, 32, 9, ee);
    std::vector<float> aug(T * 32);
    for (int t = 0; t < T * 32; ++t) aug[t] = text_hid[t] + pe[t] + ee[t];
    if (std::getenv("KANTTS_DUMP_ENC")) {
        std::ofstream f("/tmp/kt/enc_aug.bin", std::ios::binary);
        f.write((const char*)aug.data(), aug.size() * 4);
    }
    std::vector<float> cond(T * 96);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 32; ++c) {
            cond[t * 96 + c] = aug[t * 32 + c];
            cond[t * 96 + 32 + c] = spk_hid[t * 32 + c];
            cond[t * 96 + 64 + c] = emo_hid[t * 32 + c];
        }
    auto log_dur = DurationAr(cond, w, T, 96);
    if (std::getenv("KANTTS_DUMP_ENC")) {
        std::ofstream f("/tmp/kt/enc_logdur.bin", std::ios::binary);
        f.write((const char*)log_dur.data(), log_dur.size() * 4);
    }
    durations.resize(T);
    int sum = 0;
    std::vector<int> reps(T);
    for (int t = 0; t < T; ++t) {
        durations[t] = std::exp(log_dur[t]) - 1.0f;
        reps[t] = (int)(durations[t] + 0.5f);
        sum += reps[t];
    }
    int pad = 3 - sum % 3;
    if (pad == 3) pad = 0;
    int P = sum + pad;
    // LR text/spk/emo
    auto expand = [&](const std::vector<float>& src, std::vector<float>& dst) {
        dst.assign(P * 32, 0.0f);
        int pos = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < reps[t]; ++r) {
                std::copy(src.begin() + t * 32, src.begin() + (t + 1) * 32,
                          dst.begin() + (pos++) * 32);
            }
    };
    std::vector<float> lr_text, lr_emo, lr_spk;
    expand(aug, lr_text);
    expand(emo_hid, lr_emo);
    expand(spk_hid, lr_spk);
    // dur position encoder
    std::vector<float> rc(T + 1, 0);
    for (int t = 0; t < T; ++t) rc[t + 1] = rc[t] + reps[t];
    std::vector<float> lr_pos(P * 32, 0.0f);
    for (int p = 0; p < P; ++p) {
        int ph = 0;
        for (int t = 0; t < T; ++t)
            if (rc[t] <= p && p < rc[t + 1]) { ph = p - rc[t] + 1; break; }
        for (int c = 0; c < 32; ++c) {
            float inv = std::pow(10000.0f, 2.0f * (c / 2) / 32.0f);
            float v = ph / inv;
            lr_pos[p * 32 + c] = (c % 2 == 0) ? std::sin(v) : std::cos(v);
        }
    }
    for (int i = 0; i < P * 32; ++i) lr_text[i] += lr_pos[i];
    int M = P / 3;
    memory.assign(M * 160, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int c = 0; c < 96; ++c) memory[m * 160 + c] = lr_text[m * 96 + c];
        for (int c = 0; c < 32; ++c) {
            memory[m * 160 + 96 + c] = lr_spk[m * 96 + c];
            memory[m * 160 + 128 + c] = lr_emo[m * 96 + c];
        }
    }
    lr_len = sum;
}

// PNCA 单步解码（host）
struct Decoder {
    struct LW {
        const float* ln_w; const float* ln_b;
        const float* xqkv_w; const float* xqkv_b;
        const float* hkv_w; const float* hkv_b;
        const float* fcx_w; const float* fcx_b;
        const float* fch_w; const float* fch_b;
        const float* pln_w; const float* pln_b;
        const float* p1_w; const float* p1_b;
        const float* p2_w; const float* p2_b;
    } lw[12];
    const float* pre0_w; const float* pre0_b;
    const float* pre1_w; const float* pre1_b;
    const float* pre2_w; const float* pre2_b;
    const float* proj_w; const float* proj_b;
    const float* ln_w; const float* ln_b;
    const float* out_w; const float* out_b;
    int M;
    std::vector<float> hk[12], hv[12];

    explicit Decoder(const Weights& w) : M(0) {
        auto P = [&](const char* n) { return w.Get(n).data(); };
        pre0_w = P("pre0_w"); pre0_b = P("pre0_b");
        pre1_w = P("pre1_w"); pre1_b = P("pre1_b");
        pre2_w = P("pre2_w"); pre2_b = P("pre2_b");
        proj_w = P("proj_w"); proj_b = P("proj_b");
        ln_w = P("ln_w"); ln_b = P("ln_b");
        out_w = P("out_w"); out_b = P("out_b");
        for (int li = 0; li < 12; ++li) {
            std::string p = "l" + std::to_string(li) + "_";
            lw[li].ln_w = w.Get(p + "ln_w").data(); lw[li].ln_b = w.Get(p + "ln_b").data();
            lw[li].xqkv_w = w.Get(p + "xqkv_w").data(); lw[li].xqkv_b = w.Get(p + "xqkv_b").data();
            lw[li].hkv_w = w.Get(p + "hkv_w").data(); lw[li].hkv_b = w.Get(p + "hkv_b").data();
            lw[li].fcx_w = w.Get(p + "fcx_w").data(); lw[li].fcx_b = w.Get(p + "fcx_b").data();
            lw[li].fch_w = w.Get(p + "fch_w").data(); lw[li].fch_b = w.Get(p + "fch_b").data();
            lw[li].pln_w = w.Get(p + "pln_w").data(); lw[li].pln_b = w.Get(p + "pln_b").data();
            lw[li].p1_w = w.Get(p + "p1_w").data(); lw[li].p1_b = w.Get(p + "p1_b").data();
            lw[li].p2_w = w.Get(p + "p2_w").data(); lw[li].p2_b = w.Get(p + "p2_b").data();
        }
    }

    void Prepare(const std::vector<float>& memory) {
        M = (int)memory.size() / 160;
        std::vector<float> mem_p(270 * 160, 0.0f);
        std::copy(memory.begin(), memory.end(), mem_p.begin());
        for (int li = 0; li < 12; ++li) {
            std::vector<float> hkv(270 * 256);
            Matmul(mem_p.data(), lw[li].hkv_w, lw[li].hkv_b, 270, 160, 256, hkv);
            hk[li].assign(8 * 270 * 16, 0.0f);
            hv[li].assign(8 * 270 * 16, 0.0f);
            for (int m = 0; m < 270; ++m)
                for (int h = 0; h < 8; ++h)
                    for (int d = 0; d < 16; ++d) {
                        hk[li][(h * 270 + m) * 16 + d] = hkv[(m * 2 + 0) * 128 + h * 16 + d];
                        hv[li][(h * 270 + m) * 16 + d] = hkv[(m * 2 + 1) * 128 + h * 16 + d];
                    }
        }
    }

    void Step(const std::vector<float>& frame, const std::vector<float>& mem_step,
              std::vector<float>& xk, std::vector<float>& xv, int s, int xb,
              std::vector<float>& out) {
        std::vector<float> x(256);
        for (int o = 0; o < 256; ++o) {
            float acc = pre0_b[o];
            for (int k = 0; k < 80; ++k) acc += frame[k] * pre0_w[o * 80 + k];
            x[o] = std::max(acc, 0.0f);
        }
        std::vector<float> x1(256);
        for (int o = 0; o < 256; ++o) {
            float acc = pre1_b[o];
            for (int k = 0; k < 256; ++k) acc += x[k] * pre1_w[o * 256 + k];
            x1[o] = std::max(acc, 0.0f);
        }
        x = std::move(x1);
        std::vector<float> x2(128);
        for (int o = 0; o < 128; ++o) {
            float acc = pre2_b[o];
            for (int k = 0; k < 256; ++k) acc += x[k] * pre2_w[o * 256 + k];
            x2[o] = acc;
        }
        x = std::move(x2);
        x.resize(128);
        x.insert(x.begin(), mem_step.begin(), mem_step.end());  // (1,288)
        std::vector<float> xp(128);
        for (int o = 0; o < 128; ++o) {
            float acc = proj_b[o];
            for (int k = 0; k < 288; ++k) acc += x[k] * proj_w[o * 288 + k];
            xp[o] = acc * std::sqrt(128.0f);
        }
        x = std::move(xp);
        int xs0 = std::max(s - xb, 0);
        int he = std::min(s + xb + 1, M);
        std::vector<float> xmask(270, -1e9f), hmask(270, -1e9f);
        for (int m = xs0; m <= s; ++m) xmask[m] = 0.0f;
        for (int m = s; m < he; ++m) hmask[m] = 0.0f;
        for (int li = 0; li < 12; ++li) {
            std::vector<float> residual = x;
            std::vector<float> lnx;
            LayerNorm(x.data(), lw[li].ln_w, lw[li].ln_b, 1, 128, lnx);
            std::vector<float> qkv;
            Matmul(lnx.data(), lw[li].xqkv_w, lw[li].xqkv_b, 1, 128, 384, qkv);
            auto qp = [&](int off) {
                std::vector<float> q(8 * 16);
                for (int h = 0; h < 8; ++h)
                    for (int d = 0; d < 16; ++d) q[h * 16 + d] = qkv[off + h * 16 + d];
                return q;
            };
            std::vector<float> q = qp(0), k = qp(128), v = qp(256);
            // 原位写入本步 kv（等价参考实现的 append），注意力直接读状态
            for (int h = 0; h < 8; ++h)
                for (int d = 0; d < 16; ++d) {
                    xk[(li * 8 + h) * 270 * 16 + s * 16 + d] = k[h * 16 + d];
                    xv[(li * 8 + h) * 270 * 16 + s * 16 + d] = v[h * 16 + d];
                }
            // x attention
            std::vector<float> wx(8 * 270), ox(128, 0.0f);
            for (int h = 0; h < 8; ++h) {
                float mx = -1e30f;
                for (int m = xs0; m <= s; ++m) {
                    float acc = 0;
                    for (int d = 0; d < 16; ++d)
                        acc += q[h * 16 + d] * xk[(li * 8 + h) * 270 * 16 + m * 16 + d];
                    wx[h * 270 + m] = acc / 4.0f;
                    mx = std::max(mx, wx[h * 270 + m]);
                }
                float sum = 0;
                for (int m = xs0; m <= s; ++m) {
                    wx[h * 270 + m] = std::exp(wx[h * 270 + m] - mx);
                    sum += wx[h * 270 + m];
                }
                for (int d = 0; d < 16; ++d) {
                    float acc = 0;
                    for (int m = xs0; m <= s; ++m)
                        acc += wx[h * 270 + m] / sum * xv[(li * 8 + h) * 270 * 16 + m * 16 + d];
                    ox[h * 16 + d] = acc;
                }
            }
            std::vector<float> oxl;
            Matmul(ox.data(), lw[li].fcx_w, lw[li].fcx_b, 1, 128, 128, oxl);
            // h attention（band 上限截到有效 memory 行数）
            std::vector<float> oh(128, 0.0f);
            for (int h = 0; h < 8; ++h) {
                float mx = -1e30f;
                std::vector<float> wh(270);
                for (int m = s; m < he; ++m) {
                    float acc = 0;
                    for (int d = 0; d < 16; ++d)
                        acc += q[h * 16 + d] * hk[li][(h * 270 + m) * 16 + d];
                    wh[m] = acc / 4.0f;
                    mx = std::max(mx, wh[m]);
                }
                float sum = 0;
                for (int m = s; m < he; ++m) {
                    wh[m] = std::exp(wh[m] - mx);
                    sum += wh[m];
                }
                for (int d = 0; d < 16; ++d) {
                    float acc = 0;
                    for (int m = s; m < he; ++m)
                        acc += wh[m] / sum * hv[li][(h * 270 + m) * 16 + d];
                    oh[h * 16 + d] = acc;
                }
            }
            std::vector<float> ohl;
            Matmul(oh.data(), lw[li].fch_w, lw[li].fch_b, 1, 128, 128, ohl);
            for (int c = 0; c < 128; ++c) x[c] = oxl[c] + ohl[c] + residual[c];
            // pos_ffn（conv k=1，p1 输出 1024 维）
            LayerNorm(x.data(), lw[li].pln_w, lw[li].pln_b, 1, 128, lnx);
            std::vector<float> px(1024);
            for (int o = 0; o < 1024; ++o) {
                float acc = lw[li].p1_b[o];
                for (int k = 0; k < 128; ++k) acc += lnx[k] * lw[li].p1_w[o * 128 + k];
                px[o] = std::max(acc, 0.0f);
            }
            for (int o = 0; o < 128; ++o) {
                float acc = lw[li].p2_b[o];
                for (int k = 0; k < 1024; ++k) acc += px[k] * lw[li].p2_w[o * 1024 + k];
                x[o] = acc + x[o];
            }
        }
        LayerNorm(x.data(), ln_w, ln_b, 1, 128, x);
        Matmul(x.data(), out_w, out_b, 1, 128, 240, out);
    }
};


std::vector<float> Postnet(const std::vector<float>& dec, const Weights& w, int T) {
    std::vector<float> x = FsmnEncoder(dec, w, "post", T, 80, {17, 17, 17, 17});
    std::vector<float> h(128, 0), c(128, 0), out(T * 128);
    std::vector<float> xi(256);
    for (int t = 0; t < T; ++t) {
        std::copy(x.begin() + t * 256, x.begin() + (t + 1) * 256, xi.begin());
        LstmCell(xi, w.Get("post_lstm_w_ih"), w.Get("post_lstm_w_hh"), w.Get("post_lstm_b_ih"),
                 w.Get("post_lstm_b_hh"), h, c, 128);
        std::copy(h.begin(), h.end(), out.begin() + t * 128);
    }
    std::vector<float> res(T * 80);
    Matmul(out, w.Get("post_fc_w"), w.Get("post_fc_b"), T, 128, 80, res);
    for (int i = 0; i < T * 80; ++i) res[i] += dec[i];
    return res;
}

}  // namespace

KanttsPipeline::KanttsPipeline(const std::string& model_dir, const std::string& resource_dir,
                               const std::string& am_config)
    : enc_(new ModelSession(model_dir + "/am_enc.axmodel")),
      voc_(new ModelSession(model_dir + "/voc.axmodel")),
      pe_(new ModelSession(model_dir + "/pitch_energy.axmodel")),
      dur_(new ModelSession(model_dir + "/duration.axmodel")),
      post_(new ModelSession(model_dir + "/postnet.axmodel")),
      model_dir_(model_dir),
      frontend_(new Frontend(resource_dir, am_config)) {
    w_.Load(model_dir + "/host_weights");
    std::fprintf(stderr, "[stage] weights loaded\n");
}

KanttsPipeline::~KanttsPipeline() = default;

std::vector<float> KanttsPipeline::SynthesizeSymbols(
    const std::vector<std::string>& symbols) {
    std::vector<float> audio;
    for (const auto& sym : symbols) {
        auto in = frontend_->Encode(sym);
        auto t_stage = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[stage] encoded T=%d\n", in.T);
        int T = in.T;
        constexpr int MT = 128, D = 512, U = 32;
        const float* sy_w = w_.Get("sy_emb").data();      // (147,512)
        const float* tone_w = w_.Get("tone_emb").data();   // (10,512)
        const float* syll_w = w_.Get("syll_emb").data();   // (8,512)
        const float* ws_w = w_.Get("ws_emb").data();       // (8,512)
        const float* spk_w = w_.Get("spk_emb").data();     // (9,32)
        const float* emo_w = w_.Get("emo_emb").data();     // (36,32)
        const float* pos = w_.Get("pos_enc").data();       // (128,512)
        std::vector<float> x_emb(MT * D, 0.0f), attn_mask(MT, 0.0f), mask_f(MT, 0.0f);
        for (int t = 0; t < MT; ++t) {
            bool valid = t < T;
            mask_f[t] = valid ? 1.0f : 0.0f;
            attn_mask[t] = valid ? 0.0f : -3e4f;
            if (!valid) continue;
            const int* l = &in.ling[t * 4];
            for (int c = 0; c < D; ++c) {
                float v = sy_w[l[0] * D + c] + tone_w[l[1] * D + c]
                        + syll_w[l[2] * D + c] + ws_w[l[3] * D + c];
                x_emb[t * D + c] = v * std::sqrt(128.0f) + pos[t * D + c];
            }
        }
        std::vector<float> text_hid(MT * U), spk_hid(MT * U), emo_hid(MT * U);
        for (int t = 0; t < MT; ++t) {
            int s = in.spk[t], e = in.emo[t];
            for (int c = 0; c < U; ++c) {
                spk_hid[t * U + c] = spk_w[s * U + c];
                emo_hid[t * U + c] = emo_w[e * U + c];
            }
        }
        enc_->SetInput("x_emb", x_emb.data(), x_emb.size() * 4);
        enc_->SetInput("attn_mask", attn_mask.data(), attn_mask.size() * 4);
        enc_->SetInput("mask_f", mask_f.data(), mask_f.size() * 4);
        enc_->Run();
        std::vector<float> th_all(MT * U);
        enc_->GetOutput("output", th_all.data(), th_all.size() * 4);
        text_hid.assign(th_all.begin(), th_all.end());
        text_hid.resize(T * U);
        spk_hid.resize(T * U);
        emo_hid.resize(T * U);
        std::fprintf(stderr, "[timing] enc %.0fms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_stage).count());
        t_stage = std::chrono::steady_clock::now();
        if (std::getenv("KANTTS_DUMP_ENC")) {
            std::ofstream f("/tmp/kt/enc_text.bin", std::ios::binary);
            f.write((const char*)text_hid.data(), text_hid.size() * 4);
            std::ofstream f2("/tmp/kt/enc_spk.bin", std::ios::binary);
            f2.write((const char*)spk_hid.data(), spk_hid.size() * 4);
            std::ofstream f3("/tmp/kt/enc_emo.bin", std::ios::binary);
            f3.write((const char*)emo_hid.data(), emo_hid.size() * 4);
            std::ofstream f4("/tmp/kt/enc_ling.bin", std::ios::binary);
            f4.write((const char*)in.ling.data(), in.ling.size() * 4);
        }
        std::vector<float> memory;
        int lr_len = 0;
        std::vector<float> durations;
        const char* test_mem = std::getenv("KANTTS_TEST_MEM");
        if (test_mem) {
            std::ifstream mf(test_mem, std::ios::binary);
            std::vector<char> mb((std::istreambuf_iterator<char>(mf)), {});
            memory.resize(mb.size() / 4);
            std::memcpy(memory.data(), mb.data(), mb.size());
            lr_len = (int)memory.size() / 160 * 3;
            durations.assign(22, 5.0f);
            const char* test_dur = std::getenv("KANTTS_TEST_DUR");
            if (test_dur) {
                std::ifstream df(test_dur, std::ios::binary);
                std::vector<char> db((std::istreambuf_iterator<char>(df)), {});
                durations.resize(db.size() / 4);
                std::memcpy(durations.data(), db.data(), db.size());
            }
            std::fprintf(stderr, "[dbg] 使用参考 memory（%d 行）\n", (int)memory.size() / 160);
        } else {
            // pitch/energy/duration 默认 NPU（KANTTS_CPU_PRED 回退 CPU）
            std::vector<float> var_in(T * 96);
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < 32; ++c) {
                    var_in[t * 96 + c] = text_hid[t * 32 + c];
                    var_in[t * 96 + 32 + c] = spk_hid[t * 32 + c];
                    var_in[t * 96 + 64 + c] = emo_hid[t * 32 + c];
                }
            std::vector<float> pitch(T), energy(T);
            if (!std::getenv("KANTTS_CPU_PRED")) {
                std::vector<float> var_pad(128 * 96, 0.0f);
                std::copy(var_in.begin(), var_in.end(), var_pad.begin());
                pe_->SetInput("var_in", var_pad.data(), var_pad.size() * 4);
                pe_->Run();
                pe_->GetOutput("pitch", pitch.data(), pitch.size() * 4);
                pe_->GetOutput("energy", energy.data(), energy.size() * 4);
            } else {
                auto p = VarFsmnRnnPredictor(var_in, w_, "pitch", T, 96);
                auto e = VarFsmnRnnPredictor(var_in, w_, "energy", T, 96);
                pitch = std::move(p);
                energy = std::move(e);
            }
            std::vector<float> pe_c, ee_c;
            Conv1dSame(pitch, w_.Get("pitch_emb_w"), w_.Get("pitch_emb_b"), T, 1, 32, 9, pe_c);
            Conv1dSame(energy, w_.Get("energy_emb_w"), w_.Get("energy_emb_b"), T, 1, 32, 9, ee_c);
            std::vector<float> aug(T * 32);
            for (int i = 0; i < T * 32; ++i) aug[i] = text_hid[i] + pe_c[i] + ee_c[i];
            std::vector<float> cond(T * 96);
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < 32; ++c) {
                    cond[t * 96 + c] = aug[t * 32 + c];
                    cond[t * 96 + 32 + c] = spk_hid[t * 32 + c];
                    cond[t * 96 + 64 + c] = emo_hid[t * 32 + c];
                }
            std::vector<float> log_dur(T);
            if (!std::getenv("KANTTS_CPU_PRED")) {
                std::vector<float> cond_pad(22 * 96, 0.0f);
                std::copy(cond.begin(), cond.end(), cond_pad.begin());
                dur_->SetInput("cond", cond_pad.data(), cond_pad.size() * 4);
                dur_->Run();
                dur_->GetOutput("log_dur", log_dur.data(), log_dur.size() * 4);
            } else {
                log_dur = DurationAr(cond, w_, T, 96);
            }
            if (std::getenv("KANTTS_DUMP_ENC")) {
                std::fprintf(stderr, "[dbg-npu] pitch[0..3]=%.4f %.4f %.4f %.4f energy[0..3]=%.4f %.4f %.4f %.4f log_dur[0..3]=%.4f %.4f %.4f %.4f\n",
                             pitch[0], pitch[1], pitch[2], pitch[3],
                             energy[0], energy[1], energy[2], energy[3],
                             log_dur[0], log_dur[1], log_dur[2], log_dur[3]);
            }
            durations.resize(T);
            int sum = 0;
            std::vector<int> reps(T);
            for (int t = 0; t < T; ++t) {
                durations[t] = std::exp(log_dur[t]) - 1.0f;
                reps[t] = (int)(durations[t] + 0.5f);
                sum += reps[t];
            }
            if (std::getenv("KANTTS_DUMP_ENC")) {
                std::fprintf(stderr, "[dbg-npu] log_dur all:");
                for (int t = 0; t < T; ++t) std::fprintf(stderr, " %.3f", log_dur[t]);
                std::fprintf(stderr, " | reps sum=%d\n", sum);
            }
            int pad = 3 - sum % 3;
            if (pad == 3) pad = 0;
            int P = sum + pad;
            auto expand = [&](const std::vector<float>& src, std::vector<float>& dst) {
                dst.assign(P * 32, 0.0f);
                int pos = 0;
                for (int t = 0; t < T; ++t)
                    for (int r = 0; r < reps[t]; ++r) {
                        std::copy(src.begin() + t * 32, src.begin() + (t + 1) * 32,
                                  dst.begin() + (pos++) * 32);
                    }
            };
            std::vector<float> lr_text, lr_emo, lr_spk;
            expand(aug, lr_text);
            expand(emo_hid, lr_emo);
            expand(spk_hid, lr_spk);
            std::vector<float> rc(T + 1, 0);
            for (int t = 0; t < T; ++t) rc[t + 1] = rc[t] + reps[t];
            std::vector<float> lr_pos(P * 32, 0.0f);
            for (int p = 0; p < P; ++p) {
                int ph = 0;
                for (int t = 0; t < T; ++t)
                    if (rc[t] <= p && p < rc[t + 1]) { ph = p - rc[t] + 1; break; }
                for (int c = 0; c < 32; ++c) {
                    float inv = std::pow(10000.0f, 2.0f * (c / 2) / 32.0f);
                    float v = ph / inv;
                    lr_pos[p * 32 + c] = (c % 2 == 0) ? std::sin(v) : std::cos(v);
                }
            }
            for (int i = 0; i < P * 32; ++i) lr_text[i] += lr_pos[i];
            int MM = P / 3;
            memory.assign(MM * 160, 0.0f);
            for (int m = 0; m < MM; ++m) {
                for (int c = 0; c < 96; ++c) memory[m * 160 + c] = lr_text[m * 96 + c];
                for (int c = 0; c < 32; ++c) {
                    memory[m * 160 + 96 + c] = lr_spk[m * 96 + c];
                    memory[m * 160 + 128 + c] = lr_emo[m * 96 + c];
                }
            }
            lr_len = sum;
        }
        int M = (int)memory.size() / 160;
        std::fprintf(stderr, "[stage] memory M=%d lr_len=%d\n", M, lr_len);
        if (std::getenv("KANTTS_DUMP_ENC")) {
            std::ofstream f("/tmp/kt/full_mem.bin", std::ios::binary);
            f.write((const char*)memory.data(), memory.size() * 4);
            std::ofstream f2("/tmp/kt/full_dur.bin", std::ios::binary);
            f2.write((const char*)durations.data(), durations.size() * 4);
        }
        int x_band = (int)(*std::max_element(durations.begin(), durations.end()) / 3.0f + 0.5f);
        std::fprintf(stderr, "[stage] x_band=%d\n", x_band);
        std::vector<float> dec_all(M * 3 * 80);
        double dec_sum = 0;
        std::fprintf(stderr, "[timing] host(预测+memory) %.0fms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_stage).count());
        t_stage = std::chrono::steady_clock::now();
        // 交付配置：am_dec（PNCA 解码）默认 CPU，其余（enc/pe/dur/postnet/voc）全 NPU；
        // KANTTS_NPU_DEC=1 可切回 NPU 解码（QAT 实验用）。
        const bool dec_npu = std::getenv("KANTTS_NPU_DEC") != nullptr;
        if (dec_npu) {
            if (!dec_) dec_.reset(new ModelSession(model_dir_ + "/am_dec.axmodel"));
            static const char* k_names[12] = {
                "dequantize_per_tensor_101", "dequantize_per_tensor_166",
                "dequantize_per_tensor_231", "dequantize_per_tensor_296",
                "dequantize_per_tensor_361", "dequantize_per_tensor_426",
                "dequantize_per_tensor_491", "dequantize_per_tensor_556",
                "dequantize_per_tensor_621", "dequantize_per_tensor_686",
                "dequantize_per_tensor_751", "dequantize_per_tensor_816"};
            static const char* v_names[12] = {
                "dequantize_per_tensor_103", "dequantize_per_tensor_168",
                "dequantize_per_tensor_233", "dequantize_per_tensor_298",
                "dequantize_per_tensor_363", "dequantize_per_tensor_428",
                "dequantize_per_tensor_493", "dequantize_per_tensor_558",
                "dequantize_per_tensor_623", "dequantize_per_tensor_688",
                "dequantize_per_tensor_753", "dequantize_per_tensor_818"};
            std::vector<float> mem_pad(270 * 160, 0.0f);
            std::copy(memory.begin(), memory.end(), mem_pad.begin());
            std::vector<float> xk(12 * 8 * 270 * 16, 0.0f), xv(12 * 8 * 270 * 16, 0.0f);
            std::vector<float> frame(80, 0.0f), out(240), kbuf(8 * 16), vbuf(8 * 16);
            int32_t xb = x_band, ml = M;
            for (int s = 0; s < M; ++s) {
                dec_->SetInput("mel_frame", frame.data(), frame.size() * 4);
                dec_->SetInput("memory_step", memory.data() + s * 160, 160 * 4);
                dec_->SetInput("memory", mem_pad.data(), mem_pad.size() * 4);
                dec_->SetInput("x_k", xk.data(), xk.size() * 4);
                dec_->SetInput("x_v", xv.data(), xv.size() * 4);
                int32_t step_v = s;
                dec_->SetInput("step", &step_v, 4);
                dec_->SetInput("x_band", &xb, 4);
                dec_->SetInput("h_band", &xb, 4);
                dec_->SetInput("mem_len", &ml, 4);
                dec_->Run();
                dec_->GetOutput("output", out.data(), out.size() * 4);
                std::copy(out.begin(), out.begin() + 240, dec_all.begin() + s * 240);
                std::copy(out.begin() + 160, out.begin() + 240, frame.begin());
                for (int li = 0; li < 12; ++li) {
                    dec_->GetOutput(k_names[li], kbuf.data(), kbuf.size() * 4);
                    dec_->GetOutput(v_names[li], vbuf.data(), vbuf.size() * 4);
                    for (int h = 0; h < 8; ++h)
                        for (int d = 0; d < 16; ++d) {
                            xk[(li * 8 + h) * 270 * 16 + s * 16 + d] = kbuf[h * 16 + d];
                            xv[(li * 8 + h) * 270 * 16 + s * 16 + d] = vbuf[h * 16 + d];
                        }
                }
            }
            std::fprintf(stderr, "[dbg] dec NPU %d 步\n", M);
        } else {
            Decoder dec(w_);
            dec.Prepare(memory);
            std::vector<float> xk(12 * 8 * 270 * 16, 0.0f), xv(12 * 8 * 270 * 16, 0.0f);
            std::vector<float> frame(80, 0.0f), out;
            for (int s = 0; s < M; ++s) {
                std::vector<float> mem_step(memory.begin() + s * 160, memory.begin() + (s + 1) * 160);
                dec.Step(frame, mem_step, xk, xv, s, x_band, out);
                std::copy(out.begin(), out.begin() + 240, dec_all.begin() + s * 240);
                std::copy(out.begin() + 160, out.begin() + 240, frame.begin());
            }
        }
        for (size_t i = 0; i < dec_all.size(); ++i) dec_sum += dec_all[i] * dec_all[i];
        std::fprintf(stderr, "[dbg] dec rms=%.4f\n", std::sqrt(dec_sum / dec_all.size()));
        std::fprintf(stderr, "[timing] decode %d 步 %.0fms\n", M, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_stage).count());
        t_stage = std::chrono::steady_clock::now();
        std::vector<float> mel;
        {
            // postnet 模型固定 128 帧输入；M*3 可能超过 128，按 128 帧分块处理
            const int Tf = M * 3;
            for (int start = 0; start < Tf; start += 128) {
                int n = std::min(128, Tf - start);
                std::vector<float> dec_p(128 * 80, 0.0f);
                std::copy(dec_all.begin() + start * 80, dec_all.begin() + (start + n) * 80,
                          dec_p.begin());
                post_->SetInput("dec", dec_p.data(), dec_p.size() * 4);
                post_->Run();
                std::vector<float> mel_p(128 * 80);
                post_->GetOutput("output", mel_p.data(), mel_p.size() * 4);
                mel.insert(mel.end(), mel_p.begin(), mel_p.begin() + n * 80);
            }
        }
        std::fprintf(stderr, "[timing] postnet %.0fms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_stage).count());
        t_stage = std::chrono::steady_clock::now();
        double mel_sum = 0;
        for (size_t i = 0; i < mel.size(); ++i) mel_sum += mel[i] * mel[i];
        std::fprintf(stderr, "[dbg] mel rms=%.4f\n", std::sqrt(mel_sum / mel.size()));
        if (std::getenv("KANTTS_DUMP_ENC")) {
            std::ofstream f("/tmp/kt/full_mel.bin", std::ios::binary);
            f.write((const char*)mel.data(), mel.size() * 4);
        }
        // voc 帧数 = memory 完整帧数（sum 非 3 倍数时 lr_len < M*3，需包含 pad 帧）
        mel.resize(M * 3 * 80);
        // voc 分块
        const int C = 200, O = 40;
        int Tf = M * 3;
        for (int start = 0; start < Tf; start += C) {
            auto t_v = std::chrono::steady_clock::now();
            int end = std::min(start + C, Tf);
            int cs = std::max(0, start - O);
            std::vector<float> chunk(80 * C, 0.0f);
            for (int t = cs; t < end; ++t)
                for (int c = 0; c < 80; ++c) chunk[c * C + (t - cs)] = mel[t * 80 + c];
            voc_->SetInput("mel", chunk.data(), chunk.size() * 4);
            voc_->Run();
            std::fprintf(stderr, "[timing] voc chunk %.0fms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t_v).count());
            std::vector<float> wav(C * 200);
            voc_->GetOutput("wav", wav.data(), wav.size() * 4);
            int keep0 = (start - cs) * 200;
            int keepn = (end - start) * 200;
            audio.insert(audio.end(), wav.begin() + keep0, wav.begin() + keep0 + keepn);
        }
    }
    // 句末拼接静音，避免尾音被听不清（0.3s @ 16k）
    audio.insert(audio.end(), 4800, 0.0f);
    std::fprintf(stderr, "[dbg] audio samples=%zu\n", audio.size());
    return audio;
}

}  // namespace kantts
