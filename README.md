# OSDTrack：B + 静态 D0 原生 Windows 迁移版

本分支保存论文终版 **OSDTrack（UAV 微调 B + 静态 D0）** 的原生 Windows 迁移层、冻结研究源码和严格权重校验。D0 使用 `ActionValue(temporal=False)`；它不是历史 `b-plus-d` 分支发布的 GRU-D，也不是 `main` 上的 V4。

- 分支：`codex/windows-native`
- 版本标签：`osdtrack-d0-windows-rc1-20261112`
- Release：[OSDTrack B+D0 native Windows migration RC1](https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/osdtrack-d0-windows-rc1-20261112)
- Windows 10/11 + NVIDIA GPU + Miniconda + PowerShell；不需要 WSL 或 Docker。
- Ubuntu 上已经通过冻结文件校验、D0 权重加载、CPU smoke、协议测试和固定 8 帧数值对照；目标 Windows/GPU 仍需按文档完成实机验收。

## 先下载什么

新 Release 自包含四个允许分发的运行权重和校验表：

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `LoRAT_large-378.bin` | LoRAT-L/378 构造器权重 | `601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0` |
| `LoRAT_B_UAV_final.safetensors` | UAV 微调 B 权重 | `e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc` |
| `D0_static_final.safetensors` | 论文终版静态 D0 | `0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd` |
| `yolo26_uav_best.pt` | 同步当前帧 YOLO | `efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433` |
| `SHA256SUMS.txt` | 上述四个附件的校验表 | 见 Release |

第五个权重是 Meta 官方发布的 DINOv2-L/14 骨干，本仓库不重复分发。在 PowerShell 中下载并校验：

```powershell
New-Item -ItemType Directory -Force .\weights | Out-Null
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" `
  -OutFile ".\weights\dinov2_vitl14_pretrain.pth"
Get-FileHash .\weights\dinov2_vitl14_pretrain.pth -Algorithm SHA256
```

正确文件大小必须为 `1,217,586,395` 字节，SHA-256 必须为：

```text
d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
```

官方下载链接来自 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2#pretrained-models)。安装和验证脚本不会静默下载、替换或覆盖任何权重。

## Windows 快速入口

仓库默认分支 `main` 必须继续保留历史 V4，因此不要从仓库首页直接点 `Code -> Download ZIP`。请在上述新 Release 页面下载自动生成的 `Source code (zip)`，或者克隆后切换到 `codex/windows-native`：

```powershell
git clone https://github.com/Artemis-GIL/LoRAT-YOLO-V4.git
cd LoRAT-YOLO-V4
git switch codex/windows-native
```

将五个权重放入仓库根目录的 `weights` 后，在已初始化 Conda 的 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1 -RepoRoot $PWD
```

重新验证：

```powershell
.\scripts\verify_windows.ps1 -RepoRoot $PWD
```

出现 `OSDTrack native Windows verification: PASS` 代表安装级检查通过。完整前置条件、双 Conda 环境、五权重校验、CUDA 检查、仿真协议和固定短序列对照见 [原生 Windows 部署说明](docs/WINDOWS_NATIVE.md)。

## 当前分支包含什么

| 路径 | 内容 |
|---|---|
| `osdtrack/` | 无 Ubuntu 绝对路径的 D0 portable runtime、配置、协议、服务和 YOLO 子进程 |
| `scripts/` | Windows 环境搭建与严格验收脚本 |
| `environment/` | LoRAT/D0 与 YOLO 两个 Windows Conda 环境 |
| `tests/`、`tools/smoke_d0.py` | 协议测试和固定 8 帧数值对照 |
| `provenance/OSDTRACK_D0.json` | 五权重身份、大小、哈希及算法不变量 |
| `research/`、`dependencies/` | 从历史 B+GRU-D 归档继承的冻结研究源码和依赖快照 |

论文终版 D0 的身份与指标口径见 [OSDTrack D0 provenance](docs/OSDTRACK_D0_PROVENANCE.md)。仿真软件尚未绑定；当前使用通用的 `reset -> init -> track -> close` 当前帧协议，后续只需实现平台适配层。

## 历史版本边界

旧版本保持原样，没有被本分支或新 Release 覆盖：

- `main` / `v4-test420-20261003`：历史 V4；
- `b-plus-d` / `bd-test420-20261004`：历史 B+GRU-D。

本分支是从 `b-plus-d` 继承后增加 D0/Windows 层，因此 `results/test420/`、`provenance/WEIGHTS.json`、`provenance/RELEASE_ASSETS.json`、`tools/check_cpu.py` 和 [历史恢复说明](docs/REPRODUCE.md)仍忠实记录旧 GRU-D 实验，不能当作 D0 发布清单或 D0 Windows 入口。当前发布应以本 README、`provenance/OSDTRACK_D0.json` 和 `provenance/OSDTRACK_D0_RELEASE_ASSETS.json` 为准。

第三方技术与代码署名、许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和各上游许可证；本仓库不重新许可第三方权重或数据。
