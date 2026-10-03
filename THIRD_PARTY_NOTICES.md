# Third-party provenance

The V4 controller, wrappers and the archival documentation do not replace upstream ownership or licenses.

- LoRAT: https://github.com/LitingLin/LoRAT ; repository commit recorded at upload: `5260744ff0a65207a73289bbda1788c377d23bcc`. The exact inference snapshot is identified by per-file SHA-256, not by an assumption that the local working tree equals the upstream commit. License included under `dependencies/LOCAL_TRACKER_SELECTION_R1_20261002/external/LoRAT/LICENSE`.
- MMOT evaluation: https://github.com/983632847/Awesome-Multimodal-Object-Tracking ; recorded commit `bdd579fd3363e2bea51c89dd3b337f0427d291e4`. Upstream repository license included under `third_party/MMOT_Evaluation_Toolkit/LICENSE`.
- DINOv2: external pretrained weights, not uploaded. The LoRAT dependency includes its vendored implementation and existing notices. Weight provenance is retained in `provenance/WEIGHTS.json`.
- Ultralytics/YOLO: the adapter depends on the separately installed package. Package source is not copied. The custom UAV checkpoint and third-party LoRA checkpoint are archived as private release assets; their originating terms remain applicable.

No new blanket open-source license was selected for the user's research code during this private backup. Public distribution requires a separate review of the actual code, model and dataset terms.
