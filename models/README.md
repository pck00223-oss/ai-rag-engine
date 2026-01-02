# Models

This repo does **not** include any model weights.

## Recommended (small, local)
- Qwen2.5-3B-Instruct GGUF (Q5_K_M)

Example file:
- `qwen2.5-3b-instruct-q5_k_m.gguf`

## Download (Windows PowerShell)

From repo root:

```powershell
cd .\models
$env:HF_HUB_DISABLE_XET="1"
& "..\.venv\Scripts\hf.exe" download Qwen/Qwen2.5-3B-Instruct-GGUF `
  qwen2.5-3b-instruct-q5_k_m.gguf `
  --local-dir .
