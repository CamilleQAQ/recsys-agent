def rank(groups, context):
    # RRF 融合三通道 + 双通道一致性加成；two_tower 降权
    W = {"two_tower": 0.5, "itemcf": 1.0, "popularity": 1.0}
    K = 60.0
    out = {}
    for g in groups:
        cands = g["candidates"]
        scores = {}
        src = {}
        for c in cands:
            item = c["item"]
            ranks = c.get("ranks", {}) or {}
            s = 0.0
            present = 0
            for name, w in W.items():
                r = ranks.get(name)
                if r is not None:
                    s += w / (K + r)
                    present += 1
            # 一致性加成：itemcf 与 popularity 同时出现
            if "itemcf" in ranks and "popularity" in ranks:
                s += 0.15 / (K + min(ranks["itemcf"], ranks["popularity"]))
            # 三通道全支持额外小加成
            if present >= 3:
                s += 0.05 / (K + min(ranks.values()))
            scores[item] = s
            src[item] = c
        ordered = sorted(scores, key=lambda it: (-scores[it], it))
        out[str(g["id"])] = ordered
    return out
