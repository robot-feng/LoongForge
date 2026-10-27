# LoongForge 本机 uv 环境

此环境面向 `/data/tzq/LoongForge` 的 Linux x86-64、Python 3.12 和 A100。项目依赖由 `pyproject.toml` 与 `uv.lock` 管理；`local-gpu` 是默认依赖组，包含项目 GPU 依赖、LeRobot 和本机编译的算子。

在仓库根目录同步环境：

```bash
cd /data/tzq/LoongForge
uv sync --locked
```

推荐先激活本机环境，再使用 `uv run`：

```bash
source .venv/bin/activate
uv run --locked python -c 'import torch, flash_attn; print(torch.__version__, flash_attn.__version__)'
uv run --locked LoongForge --help
```

本机激活脚本会设置 CUDA 和 FFmpeg 路径。无需激活时，要显式把 FFmpeg 目录加到 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/ffmpeg${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --locked --env-file .uv.env python -c 'from torchcodec.decoders import VideoDecoder'
```

本机原有 `LD_LIBRARY_PATH` 已设置；`uv run --env-file` 不会覆盖已有环境变量，因此不能仅靠 `.uv.env` 来设置 FFmpeg 搜索路径。`uv run` 默认会同步项目环境，需要保留锁文件及其引用的本机 wheel。

## 本机依赖与迁移

- PyTorch 2.11.0+cu128，CUDA Toolkit 12.9；Flash Attention 使用 Astral 提供的 `2.8.3.post1`、CUDA 12.8、PyTorch 2.11、CPython 3.12 预编译 wheel，不从源码编译。文件位于 `.wheels/`，SHA256 为 `a50d3d6d300bee531f9068c42a83aedfeb1bc8dac50a813ab4a8abe8037df955`；来源：<https://wheels.astral.sh/simple/cu128/flash-attn/>。
- `.wheels/` 还保存 PyTorch、应用 LoongForge 补丁的 TransformerEngine 2.9、GR00T N1.7、Wall-OSS 和 DMuon wheel；`uv.lock` 引用了这些本机路径。迁移时需一并复制这些文件，重新构建时注意对应 Python、PyTorch、CUDA 和 GPU 架构。DMuon wheel 使用 Python fallback，未构建 fast-clip 扩展。
- `third_party/Loong-Megatron` 是可编辑的本地依赖。`.venv/lib/ffmpeg` 保存 TorchCodec 使用的 FFmpeg 运行库；`.uv.env` 中的绝对路径须在迁移后更新。单独复制 `uv.lock` 不能复现这些系统和本地文件。
- LoongForge GPU 依赖与 LeRobot 0.5.0 对 PyTorch、torchvision、torchcodec、diffusers 及 NumPy 的版本声明不完全一致，本机通过 uv override 统一版本。锁定成功不等于所有组合都通过运行验证；`uv pip check` 可能继续报告上游元数据不一致。
- `decord2` 替代标签错误的 `decord` 0.6；两者占用相同模块路径。实测在同一进程中先导入 `decord` 再导入 `torchcodec`，解码 LIBERO AV1 视频可能失败；先导入 `torchcodec` 再导入 `decord` 可通过当前样本。建议同一任务只使用一种视频后端。
- TransformerEngine 2.9 的 Flash Attention 版本检测可能对 2.8.3.post1 给出超出支持范围的提示，不能据此假定 TransformerEngine 的 Flash Attention 后端已启用。

## 数据与模型

本机已有的 LIBERO 数据位于 `/data/tzq/datasets/starVLA/Datasets/libero/libero_10_no_noops_1.0.0_lerobot`，Qwen3.5-0.8B 权重位于 `/data/tzq/datasets/starVLA/playground/Pretrained_models/Qwen3.5-0.8B`。该数据为 v2.1，通用具身训练参数应指定 `--lerobotdataset-version v2.1`；不能直接交给只支持 v3 的 LeRobot 0.5 默认加载器。LoongForge 内置 v2 加载器已读取实际 episode 152，包括两路图像、8 维状态、8 步 7 维动作和任务文本。Qwen 基础模型测试不等同于 `examples/embodied` 中各策略的端到端训练或推理；各策略需要其配置对应的检查点、数据和必要算子。

## 当前验证

- `uv sync --locked` 和激活环境后的 `uv run --locked` 已通过。
- Flash Attention 预编译 wheel 在 A100 上通过 BF16 前向和反向检查；TorchCodec 和 Decord 分别通过真实 LIBERO 视频读取。
- 本机 Qwen3.5-0.8B 使用一帧 LIBERO 图像完成生成及一次梯度更新；这不是具身策略训练。
- `examples/embodied` 中各策略的配置和模型模块以及 GR00T N1.7、Wall-OSS 自定义算子均通过导入；训练调度和异步检查点的 8 项单元测试通过。尚未对每种策略完成端到端训练或推理。
