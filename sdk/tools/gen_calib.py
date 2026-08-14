# -*- coding: utf-8 -*-
"""KAN-TTS 校准语料生成：中文文本 -> ttsfrd 风格符号 -> ling/emo/spk 输入 -> 校准 npy/tar.gz"""
import os
import re
import tarfile
from pathlib import Path

import jieba
import numpy as np
from pypinyin import Style, lazy_pinyin

OUT = Path("/data/yangrongzhao/kantts_recompile")

# ---------------- 前端资源 ----------------
_phone_xml = open("/tmp/PhoneSet.xml", encoding="utf-8").read()
phones = ["@" + n for n in re.findall(r"<name>([^<]+)</name>", _phone_xml)]
for i in range(1, 5):
    phones.append("@" + "#" + str(i))
tones = []
for raw in open("/tmp/tonelist.txt", encoding="utf-8"):
    line = raw.strip()
    tones.append("tone_none" if not line else "tone" + line)
SYLL = ["s_begin", "s_end", "s_none", "s_both", "s_middle"]
WS = ["word_begin", "word_end", "word_middle", "word_both", "word_none"]
EMO = ["emotion_none", "emotion_neutral", "emotion_angry", "emotion_disgust", "emotion_fear",
       "emotion_happy", "emotion_sad", "emotion_surprise", "emotion_calm", "emotion_gentle",
       "emotion_relax", "emotion_lyrical", "emotion_serious", "emotion_disgruntled",
       "emotion_satisfied", "emotion_disappointed", "emotion_excited", "emotion_anxiety",
       "emotion_jealousy", "emotion_hate", "emotion_pity", "emotion_pleasure", "emotion_arousal",
       "emotion_dominance", "emotion_placeholder1", "emotion_placeholder2", "emotion_placeholder3",
       "emotion_placeholder4", "emotion_placeholder5", "emotion_placeholder6", "emotion_placeholder7",
       "emotion_placeholder8", "emotion_placeholder9"]
SPK = ["F7", "F74", "FBYN", "FRXL", "M7", "xiaoyu"]

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]


def pinyin_syllable(ch):
    ps = lazy_pinyin(ch, style=Style.TONE3, errors="ignore")
    if not ps:
        return None
    p = ps[0]
    m = re.match(r"^([a-z]+)([0-9])?$", p)
    if not m:
        return None
    body, tone = m.group(1), int(m.group(2) or 5)  # 轻声默认 tone5
    for ini in INITIALS:
        if body.startswith(ini):
            fin = body[len(ini):]
            # 舌尖音 i -> ii（zi/ci/si），翘舌音 i -> ih（zhi/chi/shi/ri）
            if ini in ("z", "c", "s") and fin == "i":
                fin = "ii"
            if ini in ("zh", "ch", "sh", "r") and fin == "i":
                fin = "ih"
            # j/q/x/y 后 u 实为 ü -> v/ve/van/vn
            if ini in ("j", "q", "x", "y") and fin == "u":
                fin = "v"
            if ini in ("j", "q", "x", "y") and fin == "ue":
                fin = "ve"
            if ini in ("j", "q", "x", "y") and fin == "uan":
                fin = "van"
            if ini in ("j", "q", "x", "y") and fin == "un":
                fin = "vn"
            # 通用韵母规范：ui/iu/un 的标准写法
            if fin == "ui":
                fin = "uei"
            if fin == "iu":
                fin = "iou"
            if fin == "un":
                fin = "uen"
            return ini, fin, tone
    fin = body
    if fin == "u":
        fin = "v"
    return "", fin, tone


def word_symbols(word, sp="F7"):
    syms = []
    n = len(word)
    for idx, ch in enumerate(word):
        sy = pinyin_syllable(ch)
        if sy is None:
            continue
        ini, fin, tone = sy
        ini_ph = "xx" if ini == "x" else ini  # PhoneSet 中声母 x 写作 xx_c
        first = idx == 0
        last = idx == n - 1
        if ini and fin:
            f1 = "word_begin" if first else ("word_end" if last and n == 1 else "word_middle")
            f2 = "word_end" if last else "word_middle"
            syms.append("{%s_c$tone%d$s_begin$%s$emotion_neutral$%s}" % (ini_ph, tone, f1, sp))
            syms.append("{%s_c$tone%d$s_end$%s$emotion_neutral$%s}" % (fin, tone, f2, sp))
        elif fin:
            fl = "word_begin" if first else ("word_end" if last else "word_middle")
            syms.append("{%s_c$tone%d$s_both$%s$emotion_neutral$%s}" % (fin, tone, fl, sp))
        else:
            syms.append("{%s_c$tone%d$s_begin$word_begin$emotion_neutral$%s}" % (ini_ph, tone, sp))
    return syms


PUNCT = {"，": "#3", "、": "#1", "。": "#4", "！": "#4", "？": "#4", "；": "#3", "：": "#3"}


def text_to_symbols(text):
    parts = re.split(r"([，。！？；：、])", text)
    out = []
    ended = False
    for part in parts:
        if not part:
            continue
        if part in PUNCT:
            out.append("{%s$tone_none$s_none$word_none$emotion_none$F7}" % PUNCT[part])
            ended = True
            continue
        words = [w for w in jieba.cut(part) if w.strip()]
        for wi, w in enumerate(words):
            if wi > 0:
                out.append("{#1$tone_none$s_none$word_none$emotion_none$F7}")
            out.extend(word_symbols(w))
        ended = False
    if not ended:
        out.append("{#4$tone_none$s_none$word_none$emotion_none$F7}")
    return " ".join(out)


# ---------------- 编码（复刻 C++ Frontend） ----------------
def mkv(items):
    return list(items) + ["_", "~", "@[MASK]"]


def encode_sym(symbol_seq):
    sy_v = mkv(phones)
    tone_v = mkv(tones)
    syll_v = mkv(SYLL)
    ws_v = mkv(WS)
    emo_v = mkv(EMO)
    spk_v = mkv(SPK)
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


# ---------------- 校准语料 ----------------
CORPUS = [
    "八十万对六十万，优势在我！",
    "八十万人，六十万人，优势在我！",
    "北京今天天气怎么样，心情非常愉快。",
    "今天天气很好，我们一起去公园散步吧。",
    "他买了三千五百斤苹果，分给了邻居们。",
    "会议将于明天下午两点半准时开始，请大家提前十分钟入场。",
    "中国的总人口已经超过十四亿，这是世界上人口最多的国家。",
    "这次的工程预算是一百二十万元，比去年增加了百分之十五。",
    "怒发冲冠，凭栏处、潇潇雨歇。",
    "抬望眼，仰天长啸，壮怀激烈。",
    "三十功名尘与土，八千里路云和月。",
    "莫等闲、白了少年头，空悲切。",
    "靖康耻，犹未雪。臣子恨，何时灭。",
    "驾长车，踏破贺兰山缺。",
    "壮志饥餐胡虏肉，笑谈渴饮匈奴血。",
    "待从头、收拾旧山河，朝天阙。",
    "大家好，欢迎收听今天的新闻联播。",
    "全国各族人民正在为实现中华民族伟大复兴的中国梦而努力奋斗。",
    "今天沪深两市震荡上行，成交额突破一万亿元。",
    "天气预报说，明天华北地区有小到中雨，请市民注意出行安全。",
    "这款手机售价四千九百九十九元，支持五 g 网络。",
    "他一九九零年出生在上海，二零零八年考入清华大学。",
    "请把这份文件打印三份，分别送给各部门负责人。",
    "我们学校有三千多名学生，一百多位老师。",
    "飞机将于晚上七点四十五分起飞，请提前两个小时到达机场。",
    "这道菜需要放两勺盐、三勺糖和半杯醋。",
    "长江是中国最长的河流，全长约六千三百公里。",
    "他每天坚持跑步五公里，已经坚持了整整三年。",
    "明天的气温在零下五度到八度之间，出门记得多穿衣服。",
    "这台设备重达两吨半，需要用起重机才能移动。",
    "公司今年的营业额达到两千万元，净利润三百万元。",
    "请记住这个电话号码：零一零八八八八六六六六。",
    "历史不会忘记，八十年前的那场战争改变了整个世界的格局。",
    "优势在我，这句话至今仍被人们反复提起。",
    "无论面对多么强大的敌人，我们都要保持必胜的信心。",
    "科学技术的进步，正在深刻地改变着我们的生活方式。",
    "绿水青山就是金山银山，保护环境人人有责。",
    "他认真地阅读了每一份报告，并提出了许多宝贵的意见。",
    "孩子们在操场上快乐地奔跑着，欢声笑语充满了整个校园。",
    "这本书我已经读了三遍，每一次都有新的收获。",
    "火车即将进站，请乘客们站在黄色安全线以内。",
    "医生建议他多喝水，少吃油腻的食物，注意休息。",
    "今年的中秋节正好赶上国庆节，大家可以放八天假。",
    "她把房间收拾得干干净净，连窗户都擦得透亮。",
    "运动员们在赛场上奋力拼搏，为祖国赢得了荣誉。",
    "这条新闻在网络上迅速传播，引起了广泛的关注。",
    "我们坚信，只要团结一心，就没有克服不了的困难。",
    "人工智能技术正在快速发展，将给各行各业带来深刻变革。",
    "请大家爱护公共设施，不要在墙壁上乱涂乱画。",
    "春天的西湖格外美丽，柳树发芽，桃花盛开。",
    "他用了整整一个下午，终于修好了那台老旧的收音机。",
    "据统计，全国高速公路总里程已超过十六万公里。",
    "这部电影的票房突破了五十亿元，创造了新的纪录。",
    "她轻轻地说了一声谢谢，然后转身离开了房间。",
    "我们要尊重每一个人的劳动成果，珍惜每一粒粮食。",
    "清晨的公园里，老人们在打太极拳，孩子们在放风筝。",
    "请在收到邮件后及时回复，以便我们安排后续工作。",
    "他的话音刚落，全场立刻响起了热烈的掌声。",
    "面对突如其来的暴雨，志愿者们在雨中坚持疏导交通。",
    "这份合同需要双方签字盖章后才能生效。",
    "世界上没有两片完全相同的树叶，每个人都是独一无二的。",
    "他讲了一个有趣的笑话，把大家都逗乐了。",
    "考试的时候要沉着冷静，认真审题，先易后难。",
    "这家餐馆的招牌菜是红烧肉，味道非常地道。",
    "我们工厂每月生产十万件产品，出口到三十多个国家。",
    "夜深了，城市依然灯火通明，车流如织。",
    "请同学们在图书馆内保持安静，不要影响他人学习。",
    "他终于实现了自己的梦想，成为一名优秀的外科医生。",
    "窗外下着濛濛细雨，她望着远方陷入了沉思。",
    "这段旅程虽然辛苦，但沿途的风景让人终生难忘。",
    "我国自主研发的卫星成功发射，标志着航天技术又迈上了新台阶。",
    "请大家排好队，依次上车，不要拥挤。",
    "这份报告的结论是：项目可行，建议尽快立项实施。",
    "他喜欢在周末去郊外钓鱼，享受片刻的宁静。",
    "母亲做的饭菜总是那么可口，让人回味无穷。",
    "新年的钟声敲响了，人们互相祝福，憧憬着美好的未来。",
    "这项技术已申请国家专利，具有自主知识产权。",
    "请勿在电梯内吸烟，谢谢合作。",
    "她穿着红色的连衣裙，站在人群中格外显眼。",
    "经过三年的努力，这条隧道终于全线贯通。",
    "我们的目标是：到二零三五年，基本实现社会主义现代化。",
    "他把零钱放进储蓄罐里，攒了很久终于买了一辆自行车。",
    "这场比赛十分精彩，双方战成二比二平。",
    "火灾发生时，请保持冷静，用湿毛巾捂住口鼻，弯腰前进。",
    "幼儿园的小朋友们围坐在一起，认真地听老师讲故事。",
    "这只小猫只有三个月大，却已经学会了捉老鼠。",
    "他反复检查了三遍，确认没有遗漏任何细节。",
    "假期里，我们一家四口去了海边，度过了一段愉快的时光。",
    "请保持手机静音，会议马上就要开始了。",
    "这里的房价每平方米超过五万元，一般人根本买不起。",
    "他一边听着音乐，一边悠闲地喝着咖啡。",
    "城市轨道交通建设正在加快推进，预计明年年底通车。",
    "请大家把垃圾扔进分类垃圾桶，共同保护环境。",
    "她花了两个小时精心准备晚餐，等待丈夫回家。",
    "本届运动会共有来自全国各地的五千多名运动员参加。",
    "望着满天繁星，他想起了童年时在乡下的美好时光。",
    "这份文件的内容涉及商业机密，请勿外传。",
    "为了赶工期，工人们日夜奋战在建设第一线。",
    "他说话总是慢条斯理，给人一种稳重可靠的感觉。",
    "春天来了，漫山遍野的油菜花竞相开放。",
    "请根据实际情况填写表格，并在规定时间内提交。",
    "这家公司去年纳税超过一亿元，是当地的纳税大户。",
    "我们应当传承和发扬优秀的传统文化，增强文化自信。",
    "窗台上摆放着一盆兰花，散发着淡淡的清香。",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sym_paths = []
    with open(OUT / "corpus_symbols.txt", "w", encoding="utf-8") as f:
        for idx, text in enumerate(CORPUS):
            sym = text_to_symbols(text)
            f.write("%s\t%s\n" % (text, sym))
            sym_paths.append((text, sym))

    enc_samples = {"inputs_ling": [], "inputs_emo": [], "inputs_spk": [], "inputs_len": []}
    for text, sym in sym_paths:
        ling, emo, spk, T = encode_sym(sym)
        enc_samples["inputs_ling"].append(ling[None])
        enc_samples["inputs_emo"].append(emo[None])
        enc_samples["inputs_spk"].append(spk[None])
        enc_samples["inputs_len"].append(np.array([T], np.int32))

    calib_dir = OUT / "calib_data"
    calib_dir.mkdir(exist_ok=True)
    for name, samples in enc_samples.items():
        d = calib_dir / ("enc_" + name)
        d.mkdir(exist_ok=True)
        for i, s in enumerate(samples):
            np.save(d / ("%04d.npy" % i), np.ascontiguousarray(s))
        with tarfile.open(calib_dir / ("enc_%s.tar.gz" % name), "w:gz") as tar:
            for f in sorted(d.glob("*.npy")):
                tar.add(f, arcname=f.name)
    print("语料句数:", len(sym_paths))
    print("T 分布:", sorted(set(len(s.split()) for _, s in sym_paths))[:10])
    for n in ["inputs_ling", "inputs_emo", "inputs_spk", "inputs_len"]:
        print(n, len(enc_samples[n]), "samples")


if __name__ == "__main__":
    main()
