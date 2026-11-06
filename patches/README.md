# patches：第三方代码补丁

`third_party/` 保持与上游一致，**禁止直接修改**。确有必要时：

```powershell
# 1. 在 third_party/<repo> 里改好并验证
# 2. 导出 diff（相对上游 commit）
git diff > patches/<repo>-<一句话说明>.patch   # 需要临时 .git，或直接 diff -u 原始文件
# 3. 在 patches/README.md 登记：仓库、上游 commit、补丁作用、是否可上游化
# 4. 撤销本地修改，把补丁留在 patches/ 里作为唯一改动记录
```

## 已登记的补丁

| 仓库 | 上游 commit | 补丁 | 说明 |
| --- | --- | --- | --- |
| BERT4Rec | `615eaf2` | 无（仅删除数据 + 新增 `data/README.md`） | 删除示例语料，说明数据来源 |
| SASRec.pytorch | `26a8fa8` | 无（仅删除数据 + 新增 `python/data/README.md`） | 同上 |

新增补丁必须同时更新上表，否则下一个人无法判断 `third_party/` 里哪部分不是上游原样代码。
