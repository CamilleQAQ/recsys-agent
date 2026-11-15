"""Phase 3：KuaiSAR 广告系统（数据、六个模型、CTR/多任务预测、排序、竞价与预算）。

模块边界：

* ``data``：按曝光请求构造广告候选（``is_ad=1``）与点击标签，落独立 npy 工件；
* ``simulated``：真实转化标签缺失时的独立模拟转化样本（显式标注 synthetic）；
* ``features``：复用 ``recsys.ranking.features`` 的批次特征视图；
* ``models``：DeepFM / DIN / DIEN / DCNv2 / MMoE / PLE 的统一广告适配器；
* ``registry``：六模型的注册、权重解析与预测器；
* ``ranking`` / ``auction`` / ``pipeline``：广告排序、竞价与预算模拟；
* ``tools`` / ``agent``：Ads Agent（工具、业务分离，LangGraph + 真实 LLM / 离线计划）。
"""

from __future__ import annotations

AD_MODEL_NAMES = ("deepfm", "din", "dien", "dcnv2", "mmoe", "ple")

__all__ = ["AD_MODEL_NAMES"]
