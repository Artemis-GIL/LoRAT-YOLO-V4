# OSDTrack Windows 部署操作手册

本文用于在 **Windows 10/11 + NVIDIA GPU** 上部署论文终版 OSDTrack。当前固定版本为 **UAV 微调 B + 静态 D0**，不是旧 V4，也不是 GRU-D。

## 1. 本次应使用的版本

- GitHub 仓库：<https://github.com/Artemis-GIL/LoRAT-YOLO-V4>
- Windows 分支：`codex/windows-native`
- 版本标签：`osdtrack-d0-windows-rc1-20261112`
- 下载页面：<https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/osdtrack-d0-windows-rc1-20261112>

> **重要：** 仓库默认分支 `main` 是历史 V4。不要从仓库首页直接点击 `Code -> Download ZIP`，否则会下载错误版本。必须从上面的 Release 页面下载 `Source code (zip)`，或者克隆后切换到 `codex/windows-native`。

RC1 表示代码、权重、Ubuntu 对照和基础测试已经整理完成，但仍需要在目标 Windows/NVIDIA 环境上完成实机验收。

## 2. 安装前准备

在 Windows 中准备：

- 64 位 Windows 10 或 Windows 11；
- NVIDIA Windows 显卡驱动；
- Miniconda，建议按当前用户安装；
- PowerShell 5.1 或更高版本，推荐 PowerShell 7；
- Microsoft Visual C++ 2015–2022 Redistributable x64；
- 稳定网络和建议至少 30 GB 可用磁盘空间，供权重、两个 CUDA Conda 环境及下载缓存使用。

本方案不使用 WSL、Docker 或 Ubuntu。PyTorch CUDA wheel 自带运行库，通常不需要另外安装完整 CUDA Toolkit，但 NVIDIA 驱动必须正常。

建议使用短的纯英文目录，例如：

```text
D:\OSDTrack
```

避免把仓库放在过深、含空格或中文的路径中。

## 3. 下载正确源码

### 方法 A：从 Release 下载，推荐

1. 登录 GitHub。
2. 打开 Windows RC1 Release：
   <https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/osdtrack-d0-windows-rc1-20261112>
3. 必要时展开页面底部的 `Assets`，下载 `Source code (zip)`。
4. 解压后，将目录整理为 `D:\OSDTrack`。

确认以下路径直接存在：

```text
D:\OSDTrack\scripts
D:\OSDTrack\osdtrack
D:\OSDTrack\environment
D:\OSDTrack\provenance
```

如果看到 `D:\OSDTrack\某个压缩包目录\scripts`，说明多套了一层目录。应进入内层目录执行后续命令，或者将内层内容移动到 `D:\OSDTrack`。

### 方法 B：使用 Git

已经配置 GitHub 私有仓库登录时，可以执行：

```powershell
cd D:\
git clone --branch codex/windows-native --single-branch `
  https://github.com/Artemis-GIL/LoRAT-YOLO-V4.git OSDTrack
cd D:\OSDTrack
```

执行以下命令确认分支：

```powershell
git branch --show-current
```

正确输出应为：

```text
codex/windows-native
```

## 4. 下载五个必需权重

先建立权重目录：

```powershell
cd D:\OSDTrack
New-Item -ItemType Directory -Force .\weights | Out-Null
```

### 4.1 从 GitHub Release 下载四个权重

从 Windows RC1 Release 下载以下文件，并放进 `D:\OSDTrack\weights`：

```text
LoRAT_large-378.bin
LoRAT_B_UAV_final.safetensors
D0_static_final.safetensors
yolo26_uav_best.pt
SHA256SUMS.txt
```

### 4.2 下载官方 DINOv2-L/14

在 PowerShell 中执行：

```powershell
cd D:\OSDTrack

Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" `
  -OutFile ".\weights\dinov2_vitl14_pretrain.pth"
```

如果浏览器或 PowerShell 下载中断，不要使用不明来源的同名文件；重新从上述 Meta 官方地址下载。

### 4.3 五个权重的固定身份

| 文件名 | 字节数 | SHA-256 |
|---|---:|---|
| `dinov2_vitl14_pretrain.pth` | 1,217,586,395 | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| `LoRAT_large-378.bin` | 130,101,948 | `601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0` |
| `LoRAT_B_UAV_final.safetensors` | 130,101,948 | `e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc` |
| `D0_static_final.safetensors` | 1,002,996 | `0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd` |
| `yolo26_uav_best.pt` | 60,285,155 | `efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433` |

特别注意：旧文件 `GRU_D_final.safetensors` 即使大小也是 `1,002,996` 字节，也不能代替静态 D0。它的 SHA-256 为 `998a0b111232fae8ad8b6b2cdfa5ed906101c25969abf670850b7f0f01ccdd51`，属于错误版本。

确认目录内容：

```powershell
Get-ChildItem .\weights
```

正确结构应为：

```text
D:\OSDTrack\weights\dinov2_vitl14_pretrain.pth
D:\OSDTrack\weights\LoRAT_large-378.bin
D:\OSDTrack\weights\LoRAT_B_UAV_final.safetensors
D:\OSDTrack\weights\D0_static_final.safetensors
D:\OSDTrack\weights\yolo26_uav_best.pt
D:\OSDTrack\weights\SHA256SUMS.txt
```

安装脚本会再次严格检查五个权重的字节数和 SHA-256。

## 5. 打开正确的终端

从开始菜单打开 **Miniconda PowerShell Prompt**。不要使用尚未初始化 Conda 的普通命令提示符。

执行：

```powershell
conda --version
nvidia-smi
```

继续安装前必须满足：

- `conda --version` 能输出 Conda 版本；
- `nvidia-smi` 能显示 NVIDIA GPU、驱动版本和显存信息。

如果其中任意一条失败，先解决 Miniconda 或显卡驱动问题，不要继续执行安装脚本。

## 6. 一键创建环境并验证

在 Miniconda PowerShell Prompt 中执行：

```powershell
cd D:\OSDTrack

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\setup_windows.ps1 -RepoRoot $PWD
```

`Set-ExecutionPolicy -Scope Process` 只对当前窗口生效，不会永久修改系统执行策略。

安装脚本会：

1. 创建或更新 `osdtrack-lorat` Conda 环境；
2. 创建或更新 `osdtrack-yolo` Conda 环境；
3. 安装项目固定版本的 Python、PyTorch 和其他依赖；
4. 保存当前机器的仓库、权重和 YOLO Python 路径；
5. 校验五个权重；
6. 检查静态 D0，拒绝 GRU-D；
7. 运行冻结文件、Python 导入、CPU smoke、协议和 CUDA 检查。

首次安装需要下载较多依赖，耗时取决于网络和磁盘速度。安装过程中不要关闭终端，也不要为了消除错误而自行更换 Python、Torch、Torchvision 或 CUDA wheel 版本。

如果权重不在默认目录，可以显式指定：

```powershell
.\scripts\setup_windows.ps1 `
  -RepoRoot $PWD `
  -WeightsDir "D:\OSDTrack\weights"
```

## 7. 验收标准

安装末尾必须出现：

```text
OSDTrack native Windows verification: PASS
```

需要重新验证时执行：

```powershell
cd D:\OSDTrack
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\verify_windows.ps1 -RepoRoot $PWD
```

只有看到完整 PASS，才能认为 Windows 安装级检查完成。使用 `-SkipGpuCheck` 得到的结果不能证明 GPU 环境可运行。

建议保存以下信息：

```powershell
nvidia-smi
conda run -n osdtrack-lorat python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
conda run -n osdtrack-yolo python -c "import sys, torch, ultralytics; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(ultralytics.__version__); print(torch.cuda.get_device_name(0))"
```

出现问题时，把以下内容一起保存并发回：

- 从第一条错误开始的完整终端输出；
- `nvidia-smi` 输出；
- 上述两个环境信息命令的输出；
- `Get-ChildItem D:\OSDTrack\weights` 输出；
- 不要只发送最后一行报错截图。

## 8. 安装通过后做什么

仿真软件尚未确定时，看到 PASS 后即可停止。此时不需要启动服务器，也不需要自己编写仿真接口。

确定 AirSim、Unreal Engine、PX4 或其他平台后，再实现对应适配器。现有统一协议为：

```text
reset(session_id)
  -> init(frame_id, timestamp, initial_bbox, current_rgb_frame)
  -> track(frame_id, timestamp, current_rgb_frame)
  -> track(...)
  -> close
```

适配器只允许发送当前 RGB 帧、时间戳和首帧初始化框，不能向算法提供未来帧、GT、未来可见性或其他前视信息。

后续本机 TCP 服务启动形式为：

```powershell
conda run --no-capture-output -n osdtrack-lorat python -m osdtrack.server `
  --transport tcp `
  --host 127.0.0.1 `
  --port 8765 `
  --io-timeout 30 `
  --output-dir .\runs\simulator
```

在仿真软件确定以前不要执行这一步；届时还需要完成平台适配和固定短序列验证。

## 9. 常见问题

### 下载后 README 仍写 V4 或 GRU-D

拿错了源码。重新从 `osdtrack-d0-windows-rc1-20261112` Release 下载 `Source code (zip)`，或者切换到 `codex/windows-native`。

### 找不到 `conda`

关闭当前窗口，从开始菜单打开 Miniconda PowerShell Prompt。仍然失败时检查 Miniconda 是否正确安装到当前用户。

### PowerShell 禁止运行脚本

只对当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### D0 哈希错误

正确静态 D0 的 SHA-256 必须以 `0a273528...` 开头并完整等于表中数值。如果得到 `998a0b11...`，下载的是旧 GRU-D，必须删除并从 Windows RC1 Release 重新下载 `D0_static_final.safetensors`。

### `torch.cuda.is_available()` 为 `False`

先检查 `nvidia-smi`。确认安装的是 NVIDIA Windows 驱动，并确认脚本创建的环境使用带 `+cu128` 的固定 PyTorch。不要通过改成 CPU 版本或跳过 GPU 检查来宣布完成。

### 出现 DLL 加载失败

安装或修复 Microsoft Visual C++ 2015–2022 Redistributable x64，重新打开终端后再验证。不要从第三方 DLL 下载站单独下载缺失文件。

### 路径中出现 `/home/xu/...` 或提示缺少 `fcntl`

运行了历史 Linux 入口。应使用 `scripts\setup_windows.ps1`、`scripts\verify_windows.ps1` 和 `osdtrack` portable runtime，不要直接运行 `research/*/pipeline.py` 或历史 `tracking/run_tracker_v4.py`。

## 10. 完成检查表

- [ ] 下载的是 `osdtrack-d0-windows-rc1-20261112` 源码；
- [ ] 仓库路径为短英文路径；
- [ ] 五个权重文件名、大小和 SHA 正确；
- [ ] `conda --version` 正常；
- [ ] `nvidia-smi` 正常；
- [ ] `setup_windows.ps1` 执行完成；
- [ ] `verify_windows.ps1` 显示完整 PASS；
- [ ] 保存了驱动、Python、Torch、CUDA 和 GPU 信息；
- [ ] 仿真平台确定前没有擅自修改算法或依赖版本。

完成以上检查后，Windows 基础部署阶段结束。
