# FoldFlow 项目代码目录结构说明

经过对比和验证，本项目 `D:\Paper\Code` 目录下的核心代码库区分如下：

## 1. `FoldFlow` (整合修改版 - 🚨 上传服务器专用)
**状态：已整合我们所有的创新点和修改（Modified/Integrated Version）**
- **这是我们正在开发的主力工作目录。**
- 内部已经包含了我们针对“免训练能量引导流匹配”所做的全部核心代码植入，具体包括：
  - `FoldFlow/guidance/energy_guidance.py`: 新增的模块，包含针对引力和斥力能量场的 `BinderEnergyGuidance` 类。
  - `FoldFlow/models/se3_fm.py`: 已修改，注入了针对 SE(3) 流形的切空间投影，以及最新的防御性创新机制——**自适应余弦相似度节流阀 (Adaptive Cosine-Similarity Throttling)**。
  - `runner/inference.py` 及 `test_energy_guidance.py`: 已修改并包含了外部引导的开关逻辑和数学验证脚本。
- **后续操作**：请直接将此 `FoldFlow` 文件夹打包并上传至 A40 Linux 服务器进行实验。

## 2. `FoldFlow_Extracted\FoldFlow-main` (原始未修改版 - 📦 备份参考)
**状态：官方原始版本（Original/Pristine Version）**
- **这是解压后的纯净版 FoldFlow 基座代码。**
- 内部**没有**我们添加的 `guidance` 文件夹，也**没有**针对能量引导或节流阀的任何修改。
- 它的作用是作为 Baseline 参考、对照组以及防止代码改崩时的原始备份。

## 3. 其他压缩包
- `Code.zip` & `FoldFlow.zip`：当前代码状态或原始状态的打包存档。

---
**总结**：如果要上服务器跑带引导的生成实验，请认准并打包上传 **`FoldFlow`** 文件夹！
