# Song Cover Pipeline — 歌曲翻唱管线

基于 MSST 和 DDSP-SVC 的完整歌曲翻唱管线。输入一首带伴奏的歌曲，输出用新音色翻唱的版本。

## 管线流程

```
原始歌曲 (人声+BGM)
    │
    ▼
┌──────────────────────────┐
│ Stage 1: 和声分离         │  Mel-Band RoFormer Karaoke (becruily)
│  分离主唱人声与伴奏        │  → Vocals + Instrumental
└──────────────────────────┘
    │
    ▼ Vocals
┌──────────────────────────┐
│ Stage 2: 混响分离         │  BS-RoFormer Dereverb (anvuew)
│  分离干声与混响尾          │  → noreverb + reverb (残差)
└──────────────────────────┘
    │
    ▼ noreverb (干声)
┌──────────────────────────┐
│ Stage 3: 音色替换         │  DDSP-SVC 6.3 (RectifiedFlow)
│  将干声替换为目标音色      │  model_20000.pt, infer_step=100, t_start=0.4
└──────────────────────────┘
    │
    ▼ 新音色人声
┌──────────────────────────┐
│ Stage 4: 混音叠加         │  音频叠加 + 增益 + 归一化
│  新音色 + BGM + 混响     │  → 最终翻唱歌曲
└──────────────────────────┘
```

## 项目结构

```
song-cover-pipeline/
├── run_pipeline.py          # 统一入口脚本
├── requirements.txt         # Python 依赖
├── configs/                 # 配置文件（每个任务一个 YAML）
│   └── ria_cover.yaml       # 默认配置：ria音色翻唱
├── src/                     # 源代码模块
│   ├── msst_separator.py    # MSST 音频分离（和声/混响）
│   ├── ddsp_converter.py    # DDSP-SVC 音色转换
│   ├── audio_mixer.py       # 多轨混音/叠加
│   └── pipeline.py          # 管线编排器
├── input/                   # 输入歌曲
└── output/                  # 输出目录
    └── {任务名}/
        ├── 01_harmony_separation/
        ├── 02_reverb_separation/
        ├── 03_timbre_conversion/
        └── 04_final_mix/
```

## 环境要求

- **OS**: Linux
- **GPU**: NVIDIA GPU with 8GB+ VRAM (测试于 RTX 4090 48GB)
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA

### 外部依赖

- **MSST** (Music-Source-Separation-Training-GUI): 音频分离代码
- **DDSP-SVC 6.3**: 音色转换推理代码

### Python 包

```bash
pip install torch numpy scipy soundfile librosa pyyaml tqdm
pip install pyworld parselmouth resampy transformers torchcrepe torchfcpe
pip install wandb einops omegaconf ml_collections loralib gin-config
```

## 快速开始

### 1. 模型准备

**Stage 1 — 和声分离模型**:
```bash
# 从 HuggingFace 下载
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('becruily/mel-band-roformer-karaoke', 'mel_band_roformer_karaoke_becruily.ckpt', local_dir='./pretrain')
"
```

**Stage 2 — 混响分离模型**:
```bash
# 已经存在于 MSST pretrain/ 目录
# dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt
```

**Stage 3 — 音色转换模型**:
DDSP-SVC checkpoint. 可通过训练获得，或使用预训练模型。
默认配置指向: `exp/ria/model_20000.pt`

### 2. 配置

编辑 `configs/ria_cover.yaml`，设置正确的路径：

```yaml
task:
  input_song: "input/你的歌曲.wav"
  output_dir: "output/你的歌曲"

harmony_separation:
  msst_code_dir: "/path/to/Music-Source-Separation-Training-GUI"
  checkpoint_path: "/path/to/mel_band_roformer_karaoke_becruily.ckpt"

timbre_conversion:
  ddsp_project_dir: "/path/to/DDSP-barbara-6.3/DDSP-barbara-6.3"
  model_ckpt: "exp/ria/model_20000.pt"
  infer_step: 100       # 推理轮数
  t_start: 0.4          # 浮动参数
```

### 3. 运行

```bash
# 验证配置
python run_pipeline.py --validate-only

# 运行完整管线
python run_pipeline.py

# 指定配置文件和强制重跑
python run_pipeline.py --config configs/my_task.yaml --force
```

### 4. 输出

输出目录结构：
```
output/{任务名}/
├── 01_harmony_separation/
│   ├── 歌曲名_Vocals.wav           # 主唱人声
│   └── 歌曲名_Instrumental.wav      # 伴奏+和声
├── 02_reverb_separation/
│   ├── 歌曲名_Vocals_noreverb.wav   # 干声（无混响）
│   └── 歌曲名_Vocals_reverb.wav     # 混响尾
├── 03_timbre_conversion/
│   └── 歌曲名_converted.wav         # 新音色人声
└── 04_final_mix/
    └── 歌曲名_cover.wav             # 最终翻唱歌曲 ★
```

## 配置文件说明

```yaml
harmony_separation:
  model_type: "mel_band_roformer"    # 模型架构
  target_stem: "Vocals"              # 主目标 stem
  other_stems: ["Instrumental"]      # 额外保留的 stem
  chunk_batch: 16                    # 推理批次大小
  device: "cuda:0"

reverb_separation:
  model_type: "bs_roformer"
  target_stem: "noreverb"            # 干声
  other_stems: ["reverb"]            # 混响（如果模型不输出则自动计算残差）
  chunk_batch: 8

timbre_conversion:
  infer_step: 100                    # 推理轮数 (rounds)
  t_start: 0.4                       # 浮动 (float): 0=纯扩散, 1=纯DDSP
  method: "euler"                    # ODE求解器
  pitch_extractor: "rmvpe"           # F0提取方法
  key: 0                             # 音高移调 (半音)
  vocal_register_shift: 0            # 音区偏移
  threshold: -60                     # 响度阈值 (dB)

mixing:
  vocal_gain: 0.0                    # 人声增益 (dB)
  instrumental_gain: 0.0             # 伴奏增益 (dB)
  reverb_gain: -3.0                  # 混响增益 (dB)
  normalize_output: true             # 峰值归一化
```

## 管线特性

- **断点续跑**: 每阶段完成后自动保存 checkpoint，中断后可继续
- **配置驱动**: 不同任务只需创建不同配置文件
- **GPU 显存管理**: 每阶段后自动释放 GPU 显存
- **残差计算**: 在内存中直接相减生成互补 stem，避免冗余 STFT/iSTFT
- **声道自适应**: 自动处理单声道/立体声转换
- **长度对齐**: 混音时自动对齐所有轨道长度

### 不降低质量的提速原则

- 保持模型、`infer_step`、F0 提取器和数值精度不变，优先提高 MSST
  `chunk_batch`，以显存换吞吐量。
- `linked_separation` 会在内存中传递中间人声，减少磁盘读写和重复变换。
- `torch.compile` 适合重复或批量任务；单首短音频需先计入首次编译开销后再比较。
- `use_amp`、更换 F0 提取器、`segment_batch_size > 1` 和降低
  `infer_step` 都可能改变结果，不属于严格的等价优化，应先做 A/B 听测和指标验证。

## 参数说明

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| infer_step | 推理轮数 | 100 | ODE 采样步数，越大质量越好但越慢 |
| t_start | 浮动 | 0.4 | 0=纯扩散(生硬), 1=纯DDSP(自然但变化小) |
| method | 采样器 | euler | euler(快) 或 rk4(更精确) |
| key | 音高移调 | 0 | 半音数，正数升调，负数降调 |
| reverb_gain | 混响增益 | -3.0dB | 控制混响尾音量，负值减弱 |

## 注意事项

1. DDSP-SVC 推理时需要切换到其项目目录执行（内部使用相对路径加载预训练模型）
2. 所有模型 checkpoint 路径在配置文件中指定，支持绝对路径和相对路径
3. 大文件（>10分钟）建议适当降低 infer_step 以控制推理时间
4. 首次运行会缓存 F0 曲线到 DDSP 项目的 cache/ 目录
