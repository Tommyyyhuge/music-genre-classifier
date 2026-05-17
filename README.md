# 音乐流派分类器 (Music Genre Classifier)

基于 **AST (Audio Spectrogram Transformer)** 的 16 类细粒度音乐流派分类模型。

## 模型信息

- **预训练模型**: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
  - 在 AudioSet 上预训练的 Audio Spectrogram Transformer
  - 1350 万参数，12 层 Transformer，768 维隐藏层
- **微调数据集**: [CCMUSIC Music Genre](https://huggingface.co/datasets/ccmusic-database/music_genre)（36,375 条 mel 频谱图）
- **分类头**: 16 类细粒度音乐流派（替换原 AudioSet 的 527 类分类头）
- **输入**: mel 频谱图，128 频率 bin × 1024 时间帧

### 16 个流派类别

| 编号 | 流派 | 中类 |
|------|------|------|
| 0 | Symphony（交响乐） | 古典 |
| 1 | Opera（歌剧） | 古典 |
| 2 | Solo（独奏） | 古典 |
| 3 | Chamber（室内乐） | 古典 |
| 4 | Soul / R&B | Soul/R&B |
| 5 | Pop Vocal Ballad（流行抒情） | 流行 |
| 6 | Adult Contemporary（成人当代） | 流行 |
| 7 | Teen Pop（青少年流行） | 流行 |
| 8 | Contemporary Dance Pop（当代舞曲） | 舞曲 |
| 9 | Dance Pop（流行舞曲） | 舞曲 |
| 10 | Classic Indie Pop（经典独立流行） | 独立 |
| 11 | Chamber Cabaret / Art Pop（艺术流行） | 独立 |
| 12 | Adult Alternative Rock（成人另类摇滚） | 摇滚 |
| 13 | Uplifting Anthemic Rock（振奋摇滚） | 摇滚 |
| 14 | Soft Rock（软摇滚） | 摇滚 |
| 15 | Acoustic Pop（原声流行） | 摇滚 |

### 性能（测试集 3,638 条）

| 模型 | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| **AST** | **96.84%** | **96.45%** | **96.84%** |
| ResNet50 基线 | 约 94% | 约 93% | 约 94% |

---

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

依赖列表：`torch`, `torchvision`, `transformers`, `datasets`, `modelscope`, `librosa`, `scikit-learn`, `tensorboard`, `matplotlib`, `tqdm` 等。

### 硬件要求

| 操作 | 最低显存 | 建议 |
|------|----------|------|
| 推理（单条） | 2 GB | CPU 也可以 |
| AST 训练 | 12 GB（batch=8） | 24 GB（batch=12） |
| ResNet50 训练 | 4 GB | 8 GB |

---

## 快速使用（推理）

### 1. 下载权重

从 GitHub Releases 或训练输出中获取 `checkpoints/ast/best_model.pt`（约 345 MB）。

### 2. 命令行推理

```python
from src.inference.predictor import predict

# GPU 推理
results = predict("周杰伦-夜曲.mp3", device="cuda")
for genre, confidence in results:
    print(f"{genre}: {confidence:.2%}")

# CPU 推理
results = predict("song.mp3", device="cpu")
```

输出示例：

```
Dance_pop: 89.58%
Adult_contemporary: 3.21%
Teen_pop: 2.15%
```

### 3. 作为模块使用

```python
from src.inference.predictor import GenrePredictor

predictor = GenrePredictor(
    checkpoint_path="./checkpoints/ast/best_model.pt",
    device="cuda"
)

# 预测整首歌（滑动窗口取平均）
results = predictor.predict("song.mp3", top_k=3)
print(results)
```

**工作原理**：输入 mp3 → 提取 mel 频谱图（128×1024） → AST 模型 → softmax → Top-K 流派。

---

## 训练

### 数据准备

数据集通过 ModelScope 自动下载（国内镜像，速度快），无需手动处理。首次运行会自动缓存到 `~/.cache/modelscope/`。

### 训练 AST 模型

```bash
python scripts/train_ast.py
```

关键配置（`src/training/config.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_classes` | 16 | 流派数 |
| `image_size` | (128, 1024) | mel 频谱图尺寸 |
| `batch_size` | 12 | 24GB 显存建议值 |
| `epochs` | 30 | 含 early stop |
| `lr_backbone` | 5e-5 | 预训练层学习率 |
| `lr_head` | 5e-4 | 分类头学习率 |
| `use_mixup` | True | 数据增强 |
| `use_specaugment` | True | 频谱增强 |
| `early_stop_patience` | 5 | 5 epoch 不降则停 |

### 训练 ResNet50 基线

```bash
python scripts/train_baseline.py
```

### 训练产出

训练完成后自动保存到：

```
checkpoints/
├── ast/best_model.pt       # AST 最佳权重 (~345 MB)
└── resnet/best_model.pt    # ResNet50 最佳权重 (~94 MB)
```

---

## 评估

```bash
python scripts/evaluate.py
```

输出内容：
- 测试集整体 Accuracy / Macro F1 / Weighted F1
- 每个类别的 Precision / Recall / F1
- Mid-level（6 大类）准确率
- AST vs ResNet50 对比表
- 混淆矩阵图（保存到 `logs/`）

---

## 项目结构

```
music-genre-classifier/
├── scripts/
│   ├── train_ast.py         # AST 训练入口
│   ├── train_baseline.py    # ResNet50 训练入口
│   └── evaluate.py          # 双模型评估 + 对比
├── src/
│   ├── data/
│   │   ├── dataset.py       # 数据加载（ModelScope / HuggingFace）
│   │   └── transforms.py    # 预处理、SpecAugment、Mixup
│   ├── models/
│   │   ├── ast_model.py     # AST 模型构建 + 参数分组
│   │   └── resnet_baseline.py
│   ├── training/
│   │   ├── config.py        # 训练配置（超参数、路径）
│   │   └── trainer.py       # 训练器（含 early stop、TensorBoard）
│   ├── evaluation/
│   │   ├── metrics.py       # 评估指标、混淆矩阵、Mid-level 准确率
│   │   └── compare.py
│   └── inference/
│       └── predictor.py     # 本地推理（mp3 → 流派）
├── checkpoints/             # 模型权重
├── logs/                    # TensorBoard 日志 + 混淆矩阵图
├── requirements.txt
└── README.md
```

---

## 部署建议

### 方案 A：本地 Python 环境
直接 clone 仓库 → `pip install -r requirements.txt` → 放入权重文件 → 调用 `predict()`。

### 方案 B：Gradio Web 界面（推荐）
可以加一个简单的 Gradio 界面：

```python
import gradio as gr
from src.inference.predictor import predict

def classify(file):
    results = predict(file.name, device="cuda")
    return {g: c for g, c in results}

gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", label="上传音频"),
    outputs=gr.Label(num_top_classes=3, label="预测流派"),
    title="音乐流派分类器"
).launch()
```

### 方案 C：Docker 部署
直接使用 AutoDL 等云 GPU 服务，拉取仓库后安装依赖即可运行。

---

## License

本项目基于 MIT License。预训练模型遵循 HuggingFace 模型使用协议，数据集 [CCMUSIC](https://huggingface.co/datasets/ccmusic-database/music_genre) 为 MIT License。
