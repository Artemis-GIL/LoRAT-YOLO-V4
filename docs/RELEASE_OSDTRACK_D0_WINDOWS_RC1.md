# OSDTrack B + static D0 — native Windows migration RC1

This release publishes the paper-final **B + static D0** model identity together with the native Windows portability layer. D0 is frozen as `ActionValue(temporal=False)` and is not interchangeable with the historical GRU-D asset.

## Download the correct source

The repository default branch remains the historical V4 `main` branch. For this version, download this release's automatically generated **Source code (zip)** or clone the repository and run:

```powershell
git switch codex/windows-native
```

Do not use the repository landing page's generic `Code -> Download ZIP`, because that follows `main`.

## Included assets

- `LoRAT_large-378.bin`
- `LoRAT_B_UAV_final.safetensors`
- `D0_static_final.safetensors`
- `yolo26_uav_best.pt`
- `SHA256SUMS.txt`

All four weights are checked by exact byte count and SHA-256. In particular, `D0_static_final.safetensors` must have SHA-256 `0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd`. The old `GRU_D_final.safetensors` is not included and must not be substituted.

## External required backbone

The Meta DINOv2-L/14 backbone is required but is not redistributed here:

- Official URL: <https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth>
- File name: `dinov2_vitl14_pretrain.pth`
- Bytes: `1,217,586,395`
- SHA-256: `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`

## Validation status

- Frozen archive verification: 915 source snapshots + 34 auxiliary files passed.
- Portable protocol tests: 21 passed.
- Static D0 checkpoint: size and SHA-256 verified; CPU load/forward previously passed in the source environment.
- Fixed 8-frame Ubuntu equivalence output SHA-256: `96c4a59c6d0ab09e9376787c786b19adc217ae9f37d7eeecc4c4215b5c79ba00`.
- Actual Windows/NVIDIA installation and numerical comparison are still required; therefore this release is marked RC1.

Start with [`docs/WINDOWS_NATIVE.md`](https://github.com/Artemis-GIL/LoRAT-YOLO-V4/blob/codex/windows-native/docs/WINDOWS_NATIVE.md). The D0 Test420 summaries are included in the source tree under `results/osdtrack_d0/test420/`.

Historical `main`, `b-plus-d`, `v4-test420-20261003`, `bd-test420-20261004`, and their releases are unchanged.
