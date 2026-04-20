# 示例工作流

## remove_mosaic_basic.json（推荐，可直接拖进 ComfyUI 网页）

完整流程：

```
VHS_LoadVideo ──IMAGE──┐
                       │
LadaLoadDetectionModel ─► detection_model ─┐
                                            ├─► LadaRemoveMosaic ─IMAGE─┬─► VHS_VideoCombine（出 mp4）
LadaLoadRestorationModel ─► restoration_model ┘                          └─► PreviewImage（预览首帧/批次）
                                            ▲
VHS_LoadVideo ──AUDIO──────────────────────────────────► VHS_VideoCombine（保留原音轨）
```

依赖的额外节点：[`ComfyUI-VideoHelperSuite`](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)（`VHS_LoadVideo` / `VHS_VideoCombine`）。
没装的话直接在 ComfyUI Manager 里搜 `VideoHelperSuite` 安装即可。

### 用法

1. 把视频丢到 `ComfyUI/input/`，例如 `input.mp4`。
2. 在 ComfyUI 网页里直接把 `remove_mosaic_basic.json` 拖进画布。
3. 检查 3 个红框：
   - `VHS_LoadVideo` 选你刚放进去的视频。
   - `LadaLoadDetectionModel` 选检测模型（默认 `lada_mosaic_detection_model_v3.1_fast.pt`）。
   - `LadaLoadRestorationModel` 选修复模型（默认 `lada_mosaic_restoration_model_generic_v1.2.pth`）。
4. 点 **Queue Prompt**，输出会保存到 `ComfyUI/output/RemoveMosaic/`。

### 参数小抄

`LadaRemoveMosaic` 节点：

- **fps**：仅用于内部临时视频的时间轴，建议跟原视频帧率保持一致（25 / 30 / 60 都行）。
- **max_clip_length**：lada 内部一次喂给修复模型的最大帧数，越大越吃显存。8 GB 显存建议 60~120，24 GB 可以拉到 200+。
- **device**：`auto` 即可；多卡机器可手动选 `cuda:0` / `cuda:1`。
- **fp16**：显存吃紧时设 `enable`，质量损失基本看不出。

## remove_mosaic_api.json（API 调用专用）

给后端/脚本调用 `/prompt` 接口用的紧凑格式。改完 `video` / `model_name` 字段直接 POST 即可：

```bash
curl -X POST http://127.0.0.1:8188/prompt \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": $(cat workflows/remove_mosaic_api.json)}"
```
