# ComfyUI_RemoveMosaic

> ComfyUI 的去马赛克插件，把 [Lada](https://github.com/ladaapp/lada) 的检测 +
> 修复管线封装成可以在 ComfyUI 工作流里直接使用的节点。

**所有 lada 推理代码已经 vendor 到本插件的 `lada/` 子目录**（AGPL-3.0），
用户不需要 `pip install lada`，只需要让 ComfyUI 的 Python 环境里有底层
依赖（torch / torchvision / mmengine / ultralytics / av / opencv 等）。

## 节点

插件提供 3 个节点，分类都在 `RemoveMosaic` 下：

| 节点 | 输入 | 输出 |
| --- | --- | --- |
| **Load Mosaic Detection Model (Lada)** | `model_name` 下拉（`models/lada/` 里的 `.pt` 文件）<br>`device` / `fp16` | `LADA_DETECTION_MODEL` |
| **Load Mosaic Restoration Model (Lada)** | `model_name` 下拉（`models/lada/` 里的 `.pth` 文件）<br>`device` / `fp16` | `LADA_RESTORATION_MODEL` |
| **Remove Mosaic (Lada)** | `images`（IMAGE 批次）<br>`detection_model`<br>`restoration_model`<br>`fps`、`max_clip_length`（可选） | `images`（IMAGE 批次） |

模型支持的版本在下拉菜单里会自动列出，比如：

- 检测：`v2`, `v3`, `v3.1-fast`, `v3.1-accurate`, `v4-fast`, `v4-accurate`
- 修复：`basicvsrpp-v1.0`, `basicvsrpp-v1.1`, `basicvsrpp-v1.2`（推荐），
  `deepmosaics`（旧的 DeepMosaics 模型）

只要你把对应的权重文件放到 `ComfyUI/models/lada/`，下拉菜单就会出现。

## 安装

### 1. 拉取插件

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-fork>/ComfyUI_RemoveMosaic.git
```

### 2. 安装依赖

在 ComfyUI 用的同一个 Python 环境里执行：

```bash
pip install -r ComfyUI_RemoveMosaic/requirements.txt
```

### 3. 装好系统的 ffmpeg / ffprobe

vendor 的 lada 代码会调用 `ffprobe` 读取视频元数据（处理 IMAGE 批次时也会用，
因为内部要先编码成临时视频）。请确认它们在 PATH 里：

```bash
ffmpeg -version
ffprobe -version
```

- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: 在 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并加进 PATH

### 4. 下载模型权重

模型权重托管在 Hugging Face：[`ladaapp/lada`](https://huggingface.co/ladaapp/lada/tree/main)。
全部放到 `ComfyUI/models/lada/`（首次加载插件时会自动创建该目录），节点会自动扫描出现在的文件。

常用文件：

| 类型 | 文件名 | 说明 |
| --- | --- | --- |
| 检测 | `lada_mosaic_detection_model_v3.1_fast.pt` | 6 MB，速度优先 |
| 检测 | `lada_mosaic_detection_model_v3.1_accurate.pt` | 20 MB，质量优先 |
| 检测 | `lada_mosaic_detection_model_v2.pt` | 45 MB，老版本 |
| 修复 | `lada_mosaic_restoration_model_generic_v1.2.pth` | 78 MB，推荐 |
| 修复 | `lada_mosaic_restoration_model_generic_v1.2_full.pth` | 287 MB，含判别器 |
| 修复 | `lada_mosaic_restoration_model_generic_v1.1.pth` | 78 MB，旧版 |

> 注：目前 HF 仓库里挂的是 v3.1 系列检测模型，README 里的 v4 是 GitHub Release 上的旧文档命名，按下面的命令下载即可。

#### 一键下载（推荐，二选一）

**方式 A：用 `huggingface-cli`**

```bash
pip install -U "huggingface_hub[cli]"
cd /path/to/ComfyUI/models/lada
huggingface-cli download ladaapp/lada \
    lada_mosaic_detection_model_v3.1_fast.pt \
    lada_mosaic_restoration_model_generic_v1.2.pth \
    --local-dir . --local-dir-use-symlinks False
```

**方式 B：用 `wget` / `curl`**

```bash
cd /path/to/ComfyUI/models/lada

# 检测模型（任选一个）
wget https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_fast.pt
wget https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_accurate.pt

# 修复模型（推荐这个）
wget https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth
```

国内/AutoDL 拉不动 HF 时换镜像：把上面所有 `https://huggingface.co` 替换成 `https://hf-mirror.com` 即可，例如：

```bash
wget https://hf-mirror.com/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth
```

或全局设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

放好之后重启 ComfyUI，节点下拉里就能看到这些模型。也可以放任意其他 `.pt` / `.pth`（包括旧的 DeepMosaics `clean_youknow_video.pth`），节点会自动把它们都列出来。

## 工作流示例

```
LoadVideo ─►─┐
             ├─► Remove Mosaic (Lada) ─► VideoCombine
LoadDet  ─►─┤
             │
LoadRest ─►─┘
```

- `LoadVideo` 输出 IMAGE 批次连到 `Remove Mosaic` 的 `images` 输入。
- 两个 Load 节点输出的模型分别接 `detection_model` / `restoration_model`。
- `fps` 建议填实际帧率（影响时序窗口），`max_clip_length` 控制每次喂给
  BasicVSR++ 的最大帧数（显存不够调小，时序闪烁严重就调大）。

## 关于内部实现

- vendor 的 `lada/` 是 lada 项目的精简子集，只保留推理路径：
  `restorationpipeline/`、`models/{basicvsrpp, deepmosaics, yolo}/`、
  必要的 `utils/`，以及一份精简过的 `mmagic` 子集（去掉了仅训练 / 评估
  用到的 GAN loss、感知 loss、可视化 hook、评估器等）。
- `lada/__init__.py` 不再触发 gettext / Cocoa 等初始化，仅提供 `ModelFiles`
  注册表与常量。
- `Remove Mosaic` 节点会把 IMAGE 批次先编码为一个 **无损 ffv1 MKV** 临时文件
  再喂给 lada 的 `FrameRestorer`，处理完自动清理。这样既保留了 lada 多线程
  管线的全部行为，又避免了反复重写。

## 许可证

本插件以 **AGPL-3.0** 协议发布，与上游 lada 保持一致。原 lada 项目版权归
[Lada Authors](https://github.com/ladaapp/lada/graphs/contributors) 所有。
