# OSDTrack 原生 Windows 迁移与验证

本文面向 **Windows 10/11 + NVIDIA GPU + Miniconda + PowerShell**。全程为原生 Windows，不使用 WSL、Docker；仓库脚本和每用户 Conda 环境本身不需要管理员权限。建议把仓库放在较短的纯英文路径（例如 `C:\src\osdtrack`），避免旧版第三方代码遇到长路径或非 ASCII 路径问题。

## 1. 先确认迁移对象

论文终版 OSDTrack 是 **UAV 微调 B + 静态 D0**。它不是当前 `b-plus-d` Release 所保存的 GRU-D 时序版本，也不是历史 `main` 上的 V4。

静态 D0 和 GRU-D 的文件大小恰好相同，但内容、SHA-256 和运行语义不同。Windows 终版必须使用：

```text
D0_static_final.safetensors
SHA-256: 0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd
```

`GRU_D_final.safetensors` **不能**改名后代替 D0。

当前迁移层位于 `osdtrack/`，负责消除机器绝对路径并提供统一配置。归档中的 Linux 训练/评测入口仍用于溯源，其中部分包含 `fcntl`、Bash 路径和原机器断言；不要在 Windows 上直接运行 `research/*/pipeline.py`、`tracking/run_tracker_v4.py` 或 `tools/restore_sources.py --apply`。

迁移层不直接改写冻结研究源码：`osdtrack.compat` 只在启动时替换 LoRAT 权重路径和 YOLO 子进程两个构造缝。因此 `python tools/verify_snapshot.py` 仍应对 915 个冻结文件和 34 个辅助文件通过逐字节校验。

## 2. 前置条件

在目标机准备：

- 64 位 Windows 10 或 Windows 11；
- 可正常执行 `nvidia-smi` 的 NVIDIA Windows 驱动；
- 每用户安装的 Miniconda（不需要装到系统目录）；
- PowerShell 5.1 或更高版本，推荐 PowerShell 7；
- Microsoft Visual C++ 2015–2022 Redistributable（若尚未安装，安装程序可能要求管理员权限）；
- 足够的磁盘空间：五个权重约 1.54 GB，两个 Conda 环境还需要数 GB。

PyTorch CUDA wheel 自带所需 CUDA 运行库，通常不要求单独安装完整 CUDA Toolkit。驱动仍须兼容 CUDA 12.8。源实验使用 RTX 5090 D；其他 NVIDIA 型号必须在实机完成本文的 CUDA 和短序列验证，不能仅凭“能导入”判定等价。

先在 Miniconda Prompt 或已初始化 Conda 的 PowerShell 中检查：

```powershell
conda --version
nvidia-smi
```

如果 PowerShell 禁止本次运行脚本，只对当前窗口放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

这不会修改机器或用户的永久执行策略。

## 3. 准备五个权重

默认目录为仓库内的 `weights`，也可以用任意绝对路径并通过脚本的 `-WeightsDir` 传入。

| Windows 文件名 | 字节数 | SHA-256 | 来源 |
|---|---:|---|---|
| `dinov2_vitl14_pretrain.pth` | 1,217,586,395 | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` | 已有的官方 DINOv2-L 本地副本；当前 Release 未重复上传 |
| `LoRAT_large-378.bin` | 130,101,948 | `601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0` | GitHub Release `bd-test420-20261004` |
| `LoRAT_B_UAV_final.safetensors` | 130,101,948 | `e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc` | GitHub Release `bd-test420-20261004` |
| `D0_static_final.safetensors` | 1,002,996 | `0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd` | **当前不在 GitHub Release，必须从 Ubuntu 人工复制** |
| `yolo26_uav_best.pt` | 60,285,155 | `efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433` | GitHub Release `bd-test420-20261004` |

私有 Release：<https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/bd-test420-20261004>

Ubuntu 上的 D0 原文件为：

```text
/home/xu/zyx/send/LORAT_B_DE_R1_20261004/D0/final.safetensors
```

通过移动硬盘、内网文件服务或其他已批准方式复制到 Windows 后，将它命名为 `D0_static_final.safetensors`。复制前后都应核对 SHA，不要仅看文件名：

```bash
# Ubuntu
sha256sum /home/xu/zyx/send/LORAT_B_DE_R1_20261004/D0/final.safetensors
```

```powershell
# Windows
Get-FileHash .\weights\D0_static_final.safetensors -Algorithm SHA256
```

安装脚本不会下载、替换或覆盖任何权重。

## 4. 双 Python 环境

原实验刻意让 LoRAT/静态 D0 与 YOLO 检测器运行在两个独立进程和两个 Python 环境中，Windows 迁移继续保留这个边界。

| 进程 | Ubuntu 源环境锚点 | Windows 环境文件 |
|---|---|---|
| LoRAT + B + 静态 D0 | Python 3.10.21、Torch 2.7.1+cu128、Torchvision 0.22.1+cu128 | `environment\lorat-windows.yml` |
| YOLO | Python 3.10.20、Torch 2.11.0+cu128、Torchvision 0.26.0+cu128、Ultralytics 8.4.100 | `environment\yolo-windows.yml` |

Linux 和 Windows 的 Conda 构建号不同，因此 YAML 固定 Python 3.10 系列，并固定关键 pip 包版本。它们是以源环境为锚的 **Windows 重建候选**，不是已经在每种 Windows/GPU 上证明可安装和数值等价的锁文件。若某个固定 CUDA wheel 在目标机索引中不存在，安装应明确失败；不要为了“装上”而静默降级 Torch、改成 CPU wheel或合并两个环境。先记录错误并确认新的版本组合。

在仓库根目录运行：

```powershell
.\scripts\setup_windows.ps1 `
  -RepoRoot $PWD `
  -WeightsDir "D:\OSDTrack\weights"
```

如果权重就在默认 `weights`：

```powershell
.\scripts\setup_windows.ps1 -RepoRoot $PWD
```

脚本可重复执行，行为如下：

1. 用 `conda env update --prune` 创建或收敛更新 `osdtrack-lorat` 与 `osdtrack-yolo`；
2. 找到 YOLO 环境的真实 `python.exe`；
3. 在两个项目环境中保存七个机器本地变量；
4. 默认调用完整验证脚本。

它不修改系统 PATH，不写系统级环境变量，也不要求管理员权限。可选参数：

- `-LoRATEnvName NAME`、`-YoloEnvName NAME`：使用自定义环境名；
- `-CondaExecutable C:\...\conda.exe`：Conda 不在当前 PATH 时显式指定；
- `-SkipEnvironmentUpdate`：环境已经创建时，只刷新路径配置并验证；
- `-SkipVerify`：权重尚未复制齐时仅建环境；
- `-SkipGpuCheck`：只做 CPU/导入检查，**不算完整验收**。

保存的统一变量为：

| 变量 | 含义 |
|---|---|
| `OSDTRACK_WEIGHTS_DIR` | 当前机器的权重根目录 |
| `OSDTRACK_DINO_WEIGHT` | DINOv2-L 骨干 |
| `OSDTRACK_LORAT_WEIGHT` | LoRAT 构造权重 |
| `OSDTRACK_B_WEIGHT` | UAV 微调 B 权重 |
| `OSDTRACK_D0_WEIGHT` | 论文终版静态 D0 权重 |
| `OSDTRACK_YOLO_WEIGHT` | YOLO 检测器权重 |
| `OSDTRACK_YOLO_PYTHON` | 独立 YOLO 环境的 `python.exe` |

路径保存在项目的 Conda 环境配置中，不依赖 Ubuntu 目录，也不要求 Windows 盘符固定。移动仓库、权重目录或重建 YOLO 环境后，应重新运行安装脚本。

## 5. 独立验证

需要复验时执行：

```powershell
.\scripts\verify_windows.ps1 `
  -RepoRoot $PWD `
  -WeightsDir "D:\OSDTrack\weights"
```

验证脚本依次完成：

1. 对五个权重逐一核对字节数和 SHA-256；
2. 验证 915 个 Ubuntu 冻结文件和 34 个辅助文件未被迁移层改写；
3. 在 LoRAT 环境导入 OpenCV、NumPy、Safetensors、Torch 与 `osdtrack` 公共 API；
4. 运行 `python -m osdtrack.runtime show-config`；
5. 运行 `python -m osdtrack.runtime preflight`，再次从 Python 侧严格校验五个权重；
6. 强制隐藏 CUDA，在 CPU 上严格加载静态 D0、执行合成张量前向，并核对 B 的 301 个键；
7. 用 Pytest 跑通无 GPU 的仿真线协议和会话状态机 smoke；
8. 在 YOLO 环境中以 CPU 加载本地 `.pt` 检测器，不执行下载或真实推理；
9. 查询 NVIDIA 驱动，并在两个环境中分别执行一个 CUDA 张量操作。

最后出现以下内容，才代表安装级检查通过：

```text
OSDTrack native Windows verification: PASS
```

`-SkipGpuCheck` 只适合定位包或权重问题。用它得到的 PASS 不能证明目标 Windows GPU 环境可运行。

也可以手动查看统一配置：

```powershell
conda run -n osdtrack-lorat python -m osdtrack.runtime show-config
conda run -n osdtrack-lorat python -m osdtrack.runtime preflight
```

## 6. 对接仿真环境

本仓库的仿真边界使用通用当前帧协议，不绑定 AirSim、UE、PX4 或其他具体平台。适配器只应把仿真器当前时刻的 RGB 帧和控制信息送入服务，不得预读未来帧、GT 或未来可见性。

协议生命周期为：

```text
reset(session_id)
  -> init(frame_id, timestamp, bbox_xywh, RGB uint8 HxWx3)
  -> track(frame_id, timestamp, RGB uint8 HxWx3) ...
  -> close
```

Python 侧协议入口为：

```python
from osdtrack.protocol import Request, encode_request, decode_response
from osdtrack.service import TrackingService, serve
```

`TrackingService(backend)` 封装跟踪后端，`serve(service, input_stream, output_stream)` 处理线协议。仿真端需要保证：

- `frame_id` 单调递增，同一会话只有一次 `init`；
- 图像是连续的 `uint8` RGB HWC，不是 BGR；
- 初始框为首帧内有效的 `[x, y, width, height]`；
- 每次 `track` 只对应一张当前帧；
- 超时、进程退出和协议错误显式上报，不复用上次结果冒充当前帧；
- 新实验使用新的输出目录，不覆盖 Ubuntu 正式结果。

具体仿真平台适配完成后，至少跑一次 `reset -> init -> track -> close` 协议 smoke，再跑一段固定短序列。安装级 PASS 只证明依赖、权重和基础调用成立，不能替代真实 LoRAT + YOLO + D0 推理。

### 6.1 启动本机 TCP 服务

在仓库根目录运行：

```powershell
conda run --no-capture-output -n osdtrack-lorat python -m osdtrack.server `
  --transport tcp `
  --host 127.0.0.1 `
  --port 8765 `
  --io-timeout 30 `
  --output-dir .\runs\simulator
```

服务在模型和 YOLO 子进程就绪后，会向 stderr 输出一行 `{"event":"ready",...}`。一次 TCP 连接完成 `close` 后进程退出，由仿真管理器按实验重启可以避免跨实验污染状态。

默认只允许 loopback 绑定。`--io-timeout` 限制每次客户端 socket 读写的最长秒数，避免不完整帧永久占用服务；它不限制模型推理时间。`--allow-remote` 只是显式放开非本机监听，当前协议没有 TLS 或身份验证，不应暴露到不可信网络。如果仿真器已经能管理子进程，也可用 `--transport stdio`；此时 stdout 完全为二进制协议，日志只写 stderr。

16 字节网络字节序前缀后是有界 JSON 头和原始 RGB bytes，图像不做 base64。每帧包含形状和 RGB SHA-256。`track` 成功响应另有：

- `tracker_status="tracking"`：已确认框；
- `tracker_status="tentative"`：有当帧暂定框，但权威锚点未必写回；
- `tracker_status="lost"`：本帧无框，`bbox=null`，这是正常跟踪状态，不是协议错误。

若 LoRAT、YOLO 或控制器在处理一帧期间抛出异常，服务进入 `reset_required` 状态；仿真端必须重新发送 `reset -> init`，不能重放同一帧假定模型内部状态未改变。YOLO 子进程单次响应默认超时 60 秒，可用 `OSDTRACK_YOLO_TIMEOUT_SECONDS` 显式调整为 `(0, 3600]` 秒。

### 6.2 固定 8 帧数值对照

仓库保留了 `UAV-Anti-UAV_Test_000001` 前 8 帧的 RGB SHA 清单与 Ubuntu D0 预测，但不重复分发数据集图像。将这 8 张原图按 `00000001.jpg` 至 `00000008.jpg` 复制到一个目录，然后运行：

```powershell
conda run --no-capture-output -n osdtrack-lorat python .\tools\smoke_d0.py `
  --frames-dir "D:\OSDTrack\smoke\img" `
  --initial-bbox 272 375 13 8 `
  --max-frames 8 `
  --frame-manifest .\tests\fixtures\d0_smoke_manifest.json `
  --expected .\tests\fixtures\d0_smoke_expected.txt `
  --output-dir .\runs\d0-smoke
```

脚本先校验 8 帧解码后 RGB 像素 SHA，再进行推理，并检查首帧模板未更改。Ubuntu 原环境的迁移层已得到与归档完全相同的预测 SHA-256 `96c4a59c6d0ab09e9376787c786b19adc217ae9f37d7eeecc4c4215b5c79ba00`，逐坐标最大绝对误差约 `5e-10`。Windows 与不同 GPU 可能有浮点末位差异，默认对照容差是 `1e-4`；超出时不应直接放宽，先检查环境版本、状态转移和 YOLO 候选顺序。

## 7. 迁移验收建议

建议按下面顺序留存记录：

1. `verify_windows.ps1` 的完整终端输出；
2. `nvidia-smi`、两个环境的 Python/Torch/CUDA/GPU 信息；
3. 同一组规范图像、同一首帧框下的 Windows 短序列输出；
4. 与 Ubuntu 静态 D0 运行逐帧比较框、状态和候选选择；
5. 最后再接入实时仿真并测端到端延迟、掉帧和进程恢复。

不同 GPU 和 Windows CUDA 内核可能造成浮点末位差异，因此验收应同时看逐帧轨迹、状态转移和汇总指标，不能只比较一个 FPS 或第一帧结果。未完成短序列对照前，应把状态描述为“Windows 环境搭建通过”，而不是“论文结果已在 Windows 复现”。

## 8. 常见问题

### Conda 找不到

使用 “Miniconda Prompt” 打开 PowerShell，或传入：

```powershell
.\scripts\setup_windows.ps1 -CondaExecutable "C:\Users\USER\miniconda3\Scripts\conda.exe"
```

### D0 缺失或 SHA 与 GRU-D 相同

当前 Release 没有静态 D0。重新从 Ubuntu 的 `LORAT_B_DE_R1_20261004/D0/final.safetensors` 人工复制。若 SHA 为 `998a0b111232fae8ad8b6b2cdfa5ed906101c25969abf670850b7f0f01ccdd51`，拿到的是 GRU-D，不能用于论文终版。

### `torch.cuda.is_available()` 为 false

先确认 `nvidia-smi` 正常，再确认环境中的 Torch 版本带 `+cu128`。CPU-only Torch、过旧驱动以及远程会话的 GPU 权限都可能导致失败。不要通过跳过 GPU 检查把它当作完成。

### DLL 加载失败

安装或修复 Microsoft Visual C++ 2015–2022 Redistributable，检查杀毒软件是否隔离 DLL，并尽量使用较短的纯英文仓库路径。不要从不可信网站单独下载缺失 DLL。

### 出现 `fcntl` 或 `/home/xu/...` 错误

这通常表示误用了保留的 Linux 归档入口。改用 `osdtrack` 的 portable runtime，并确认六个 `OSDTRACK_*` 变量来自安装脚本。不要为了绕过错误删除冻结断言或批量替换归档元数据。

### YOLO 环境重建后无法启动子进程

重新运行 `setup_windows.ps1 -SkipEnvironmentUpdate`。它会重新发现 YOLO 的 `python.exe` 并更新 `OSDTRACK_YOLO_PYTHON`。
