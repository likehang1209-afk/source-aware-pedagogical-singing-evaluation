# GitHub 与 Zenodo 公开步骤

## 发布前

1. 先确认 SVQTD 提供者是否允许再次公开逐段标签和派生预测。
2. 运行 `python scripts/check_release.py`。
3. 搜索并确认不存在音频、URL、邮箱、SSH 密码、原始来源 ID 和绝对路径。
4. 填写最终论文题目、作者顺序、ORCID 和文章 DOI。

## GitHub

建议仓库名：

`source-aware-pedagogical-singing-evaluation`

在 GitHub 创建一个空的 Public repository，不要再次自动生成 README、
License 或 `.gitignore`，因为本地包已经包含这些文件。然后在本目录运行：

```bash
git init
git branch -M main
git add .
git commit -m "Release reproducibility package v1.0.0"
git remote add origin https://github.com/YOUR_ACCOUNT/source-aware-pedagogical-singing-evaluation.git
git push -u origin main
```

也可以把本目录加入 GitHub Desktop，点击 **Publish repository**，取消
`Keep this code private` 后发布。

## Zenodo DOI

1. 登录 Zenodo，并在账户设置中连接 GitHub。
2. 在 Zenodo 的 GitHub 页面点击 `Sync now`。
3. 找到该仓库并打开连接开关。
4. 回到 GitHub，创建 `v1.0.0` Release。
5. 等待 Zenodo 自动归档，然后打开记录并复制版本 DOI。
6. 将 DOI 写回论文 Code Availability、GitHub README 和最终引文。

仓库已提供 `CITATION.cff`。Zenodo 官方说明该文件可以描述软件版本；
若同时加入 `.zenodo.json`，Zenodo 会优先使用后者，因此当前不重复提供。

官方说明：

- GitHub：<https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>
- Zenodo 连接仓库：<https://help.zenodo.org/docs/github/enable-repository/>
- Zenodo 归档 Release：<https://help.zenodo.org/docs/github/archive-software/github-upload/>
- Zenodo 软件元数据：<https://help.zenodo.org/docs/github/describe-software/>

## 论文中的 Code Availability

正式公开后替换方括号：

> The reproducibility package, including duplicate-audit scripts,
> pseudonymized split assignments, model configurations, out-of-fold
> predictions, class-support diagnostics, source-cluster bootstrap analyses,
> and source-constrained permutation tests, is available at [GitHub URL] and
> archived at [Zenodo DOI]. The original SVQTD audio is not redistributed.
