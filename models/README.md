# Models

本仓库 **不包含任何模型权重文件**。  
模型体积较大，统一由使用者在本地下载并管理。

---

## 已验证模型（M1 / M1.5）

当前阶段仅测试并验证以下模型配置，用于本地 CPU 推理与推理稳定性控制（防止输出跑成“教学废话”）：

- **Qwen2.5-3B-Instruct (GGUF, Q5_K_M)**  
  体积适中（≈2.4GB），在 CPU 环境下可稳定运行，适合工程调试与 RAG 实验。

示例文件名：

- `qwen2.5-3b-instruct-q5_k_m.gguf`

---

## 下载方式（Windows / PowerShell）

在 **仓库根目录** 执行以下命令：

```powershell
cd .\models
$env:HF_HUB_DISABLE_XET="1"
& "..\.venv\Scripts\hf.exe" download Qwen/Qwen2.5-3B-Instruct-GGUF `
  qwen2.5-3b-instruct-q5_k_m.gguf `
  --local-dir .
