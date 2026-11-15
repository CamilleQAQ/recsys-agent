# 许可证汇总

以下信息来自各仓库根的 `LICENSE` 文件（前 11 个于 2026-11-04 克隆，FuxiCTR / genrec 于 2026-11-05 克隆），**不构成法律意见**。
再分发、商用、修改后发布前，请按实际用途复核。

| 仓库 | 许可证 | 许可证文件 | 再分发注意 |
| --- | --- | --- | --- |
| RecBole | MIT | `RecBole/LICENSE` | 保留版权声明即可 |
| TorchEasyRec | Apache-2.0 | `TorchEasyRec/LICENSE` | 保留 LICENSE 与 NOTICE，声明修改 |
| DeepCTR-Torch | Apache-2.0 | `DeepCTR-Torch/LICENSE` | 同上 |
| FuxiCTR | Apache-2.0 | `FuxiCTR/LICENSE` | 同上 |
| LightGBM | MIT | `LightGBM/LICENSE`、`LightGBM/R-package/LICENSE` | 保留版权声明 |
| faiss | MIT | `faiss/LICENSE`、`faiss/THIRD_PARTY_NOTICES` | **必须同时保留 THIRD_PARTY_NOTICES** |
| genrec | MIT | `genrec/LICENSE` | 保留版权声明 |
| MMRec | **GPL-3.0** | `MMRec/LICENSE` | 传染性许可：分发衍生作品需以 GPL-3.0 开源 |
| SASRec.pytorch | Apache-2.0 | `SASRec.pytorch/LICENSE` | 保留声明 |
| BERT4Rec | Apache-2.0 | `BERT4Rec/LICENSE` | 保留声明 |
| generative-recommenders | Apache-2.0 | 根 `LICENSE` + `generative_recommenders/research/cppilot/LICENSE` | 子目录另有许可，需一并保留 |
| OpenOneRec | **根 LICENSE 缺失**（README 声明 Apache-2.0） | `benchmarks/LICENSE`、`verl_distillation/LICENSE`、`verl_rl/LICENSE` | 许可以子目录文件为准；根仓库再分发前须向作者确认 |
| recommenders | Apache-2.0 | `recommenders/LICENSE` | 保留声明 |
| torch-rechub | MIT | `torch-rechub/LICENSE` | 保留版权及许可声明 |
| skim-search-agent | Apache-2.0 | `skim-search-agent/LICENSE` | 保留许可证和上游声明 |
| YieldAgent | MIT | `YieldAgent/LICENSE` | 保留许可证和版权声明 |

## 需要特别留意

1. **MMRec 是 GPL-3.0**：如果把它接入产品代码，整个分发物的许可义务会变化。仅本地实验不受影响。
   本项目只做源码归档和独立环境实验，不与主栈代码链接。
2. **OpenOneRec 根目录没有 LICENSE**：README 里的 Apache-2.0 声明不能替代许可证文件，
   子目录（benchmarks / verl_rl / verl_distillation）各有自己的 LICENSE。要复用它，先确认授权范围。
3. **faiss 的 THIRD_PARTY_NOTICES**：包含第三方代码声明，不能只留 LICENSE。
4. **数据集许可是另一回事**：
   - KuaiRec：研究用途，使用前看数据集页面条款；
   - KuaiSAR：CC BY-NC-SA 4.0（非商业 + 相同方式共享），数据不入库、不随代码分发，见
     [../docs/kuasar-roadmap.md](../docs/kuasar-roadmap.md)；
   - MovieLens：GroupLens 研究许可。
5. **模型权重**：`third_party/` 里没有任何预训练权重，上游 README 提到的权重（Qwen3/OneRec 等）
   另有许可，不能默认按代码许可使用。

## 本项目自身代码

`src/` 下的代码来自本项目，与上表无关。对外发布前请另行补充项目级 LICENSE。

## 自动研究参考快照

`autoresearch/README.md` 声明 MIT；选取的上游快照没有独立 LICENSE 文件，来源与文件哈希见 `autoresearch/UPSTREAM.json`。使用时应保留来源声明并核对上游授权范围。
