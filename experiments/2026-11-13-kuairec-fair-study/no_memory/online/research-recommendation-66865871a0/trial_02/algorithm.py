def rank(groups, context):
    # 第2轮：进一步降 two_tower 权重，强化 itemcf/popularity 一致性，引入 train_count 可靠性先验
    W = {"two_tower": 0.25, "itemcf": 1.0, "popularity": 1.0}
    K = 40.0
    out = {}
    for g in groups:
        cands = g["candidates"]
        scores = {}
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
                s += 0.30 / (K + min(ranks["itemcf"], ranks["popularity"]))
            # 三通道全支持额外加成
            if present >= 3:
                s += 0.10 / (K + min(ranks.values()))
            # train_count 可靠性先验：对数缩放，弱化极端热门
            tc = c.get("train_count", 0) or 0
            if tc > 0:
                import math
                s += 0.05 * math.log1p(tc) / (K + 1.0)
            scores[item] = s
        ordered = sorted(scores, key=lambda it: (-scores[it], it))
        out[str(g["id"])] = ordered
    return out
