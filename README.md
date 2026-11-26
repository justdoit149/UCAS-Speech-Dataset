# 直播语音数据集流水线（voicedata）

从 `data/SIP/*.mp3` 生成“主播本人说话”的数据集：切分、声纹筛选、ASR 转写、噪音标注，最终按录音导出 zip。

默认配置是 `configs/server.yaml`（GPU + 1.7B ASR）；本地 0.6B 验证需显式指定 `configs/local.yaml`。首次 `build` 会自动检查并下载缺失模型。

当前本地已有一个完整样例：`out/sandpaper-20241223/`。

## 处理流程

1. **发现与解码**：扫描 `data/SIP/`，用 ffmpeg 解码为 16k 单声道 WAV。
2. **VAD + AED**：FireRedVAD 切出语音段；FireRedVAD AED 按帧区分 `speech / singing / music`，只保留 speech 帧，剔除 BGM 歌词和纯音乐，并合并相邻短句。
3. **声纹筛选**：CAM++ 提取 embedding，与 `说话声纹.json` 中的主播模板计算最大余弦相似度。
4. **ASR**：Qwen3-ASR 转写，同时记录 token 平均 logprob 作为置信度。
5. **噪音分类**：NoisySpeechDetection 输出 `noisy` 0/1 和 `noise_prob`。
6. **导出**：按规则过滤空文本、填充词、低置信片段，每个 mp3 生成一个 zip，并重建全局索引。

`sv >= 0.30` 的片段才进入 ASR/噪音；`sv >= 0.40` 且通过文本过滤的片段才会导出。

## 技术栈

| 用途 | 模型 / 工具 |
| --- | --- |
| VAD | FireRedVAD（`fireredvad==0.0.2`） |
| 歌声/音乐检测 | FireRedVAD AED |
| 声纹 | `iic/speech_campplus_sv_zh-cn_16k-common`（ModelScope） |
| ASR（默认，服务器） | Qwen3-ASR-1.7B |
| ASR（本地验证） | Qwen3-ASR-0.6B（`qwen-asr==0.0.6`） |
| 噪音分类 | `Etherll/NoisySpeechDetection-v0.2` |
| 音频读写 | `soundfile` / `librosa`，外部需要 ffmpeg/ffprobe |
| 其他 | Python 3.11、PyYAML、NumPy、SciPy、psutil、pytest |

完整依赖见 `requirements.txt`。

## 目录

```text
homework/
├── configs/
│   ├── local.yaml              # 本地 CPU 验证：0.6B ASR（需显式指定）
│   └── server.yaml             # 默认：服务器 GPU + 1.7B ASR
├── data/SIP/                   # 输入 mp3（本地 13 个；全量在服务器）
├── models/                     # 模型权重，缺失时 build 自动下载
├── out/<job>/                  # 最终产物：zip + 索引 + calibration.html
├── work/<job>/                 # 运行状态：state.db、pipeline.log（导出后音频缓存自动清理）
├── scripts/download_models.py  # 手动预下载模型（可选）
├── tests/                      # 使用假模型的 pytest 测试
├── voicedata/                  # 流水线源码
│   ├── cli.py                  # build / status / inspect / validate
│   ├── pipeline.py             # 阶段编排、锁、日志
│   ├── ingest.py               # 输入发现与登记
│   ├── decode.py               # 解码
│   ├── vad.py                  # VAD + AED + 切片
│   ├── sv.py                   # 声纹
│   ├── asr.py                  # ASR
│   ├── noise.py                # 噪音分类
│   ├── export.py               # zip / JSON 导出与索引
│   ├── workcache.py            # work 音频清理与按需恢复
│   ├── validate.py             # 产物校验
│   ├── inspector.py            # 人工抽听 HTML
│   ├── db.py                   # SQLite 状态库
│   ├── config.py               # 配置与参数哈希
│   └── adapters/               # 各模型适配器
├── README.md
├── requirements.txt
├── 数据集任务-需求确认.md
├── 说话声纹.json               # 主播声纹模板，26×192，必需
├── 小声说话声纹.json           # 另一个人，未使用
├── 大作业要求.pdf
└── short-c67_c_x6/             # 别人的工作，与本流水线无关
```

## 输入

- 音频：`data/SIP/` 下的 `.mp3`，可用 `--file` 按文件名子串筛选。
- 声纹：`说话声纹.json`，形状 `(26, 192)`。`小声说话声纹.json` 不是同一个人，不要替换使用。
- 模型：`configs/*.yaml` 中的 `paths.model_dir` 指向 `models/`。

## 输出

每个 mp3 在 `out/<job>/` 下生成一个同名 zip：

```text
out/<job>/
├── <stem>.zip
├── index.jsonl
├── index.meta.json
└── calibration.html          # inspect 生成，不属于正式数据
```

zip 内部：

```text
<stem>.zip
├── manifest.json             # 原文件、时长、阈值、模型、参数哈希
├── segments.jsonl            # 一行一个片段，不含音频
└── audio/
    └── <segment_id>.flac     # 16k / 单声道 / 16bit
```

`segments.jsonl` 示例：

```json
{"segment_id":"f28f82b73a7643d0","start_s":613.2,"end_s":614.827,
 "duration_s":1.627,"asr_text":"我不是全职主播。","noisy":1,
 "audio":"audio/f28f82b73a7643d0.flac","sv_score":0.4847,
 "noise_prob":0.9976,"asr_conf":-6.993}
```

`index.jsonl` 在片段字段前加 `audio_file` 和 `archive`：

```json
{"audio_file":"20231206-125154.mp3","archive":"20231206-125154.zip",
 "segment_id":"f28f82b73a7643d0","start_s":613.2,"end_s":614.827,
 "duration_s":1.627,"asr_text":"我不是全职主播。","noisy":1,
 "audio":"audio/f28f82b73a7643d0.flac","sv_score":0.4847,
 "noise_prob":0.9976,"asr_conf":-6.993}
```

字段：

- `start_s/end_s`：相对原 mp3 的绝对时间；`duration_s = end_s - start_s`。
- `audio`：zip 内相对路径。
- `noisy`：1 表示有背景杂音，BGM 也算；`noise_prob` 为对应概率。
- `asr_conf`：token 平均 logprob，越低越不可信；缺失表示旧数据或模型未返回。
- `sv_score`、`noise_prob` 仅在 `export.include_scores: true` 时输出。
- `segments.jsonl` 与 `index.jsonl` 中，同一录音的片段按 `start_s` 升序，同一录音内不重叠；不同录音按 zip 文件名分组。

旧版单 JSON（`audio_base64` 内嵌 WAV）仍可通过 `export.format: json` 生成；该模式不生成索引，`validate`/`inspect` 仍能读取旧 `.json`。

## work 目录与磁盘占用

`work/<job>/` 是运行缓存，不是交付物。默认 `run.cleanup_audio_after_export: true`：每个录音导出成功后，删掉该文件的整段解码 WAV 和所有候选片段 WAV，只保留 `state.db` 和 `pipeline.log`，因此不会和 `out/` 里的成品各存一份大音频。

- 正常全量跑完后，`work/<job>/` 只保留 `state.db` 和日志；样例约 260KB，状态库大小随片段数增长。
- 之后只改导出阈值/填充词/置信度，或重跑指定文件的声纹/ASR/噪音，缺失的片段 WAV 会按数据库里的绝对起止时间从源 mp3 临时重新切出，用完再次清理。
- 所有阶段都已完成时，再跑同一条命令只检查状态并重建索引，不会重新解码或加载模型。
- 需要保留全部中间音频（例如频繁反复调试、不想依赖源文件）时，设为 `false`。
- 源 mp3 被移走或删除后，已清理的片段无法恢复；`out/` 里的成品不受影响。

## 安装

需要 Python 3.11，且 `ffmpeg`、`ffprobe` 在 `PATH` 中。

```bash
python -m pip install -r requirements.txt
# qwen-asr 在 Windows 上建议避免安装 sox/gradio 等可选依赖：
python -m pip install qwen-asr==0.0.6 --no-deps
```

## 模型准备

不需要先手动下载。`build` 开始时会按本次要跑的阶段检查模型，缺什么就下载什么，下载完再登记输入文件；下载失败会打印“模型准备失败”并退出（退出码 2），不会留下半登记的任务。

- 默认 `server.yaml` 需要：FireRedVAD（VAD + AED）、CAM++、Qwen3-ASR-1.7B、NoisySpeechDetection。
- 本地 `local.yaml` 需要：上述模型中的 0.6B ASR 版本。
- 只跑 `--stages export` 或 `decode` 时不检查模型；`run.fake_models: true` 也不下载。

需要提前下载或补下模型时：

```bash
python scripts/download_models.py                  # 默认服务器模型（含 1.7B）
python scripts/download_models.py --only asr06b    # 本地 0.6B
python scripts/download_models.py --only vad sv noise
python scripts/download_models.py --list
```

模型目录结构：

```text
models/
├── FireRedVAD/{VAD,AED}/
├── campplus/
├── Qwen3-ASR-0.6B/
├── Qwen3-ASR-1.7B/
└── NoisySpeechDetection/
```

## 运行

不写 `--config` 时默认使用 `configs/server.yaml`。

```bash
# 服务器全量（默认配置）
python -m voicedata build --job full

# 中断后重跑同一条命令即可续跑；--retry-failed 重试失败项
python -m voicedata build --job full --retry-failed

# 只跑一个片段验证流程（--file 必须恰好匹配一个文件）
python -m voicedata build --job demo --file 20231206 --start 600 --duration 120

# 本地 CPU / 0.6B 必须显式指定 local.yaml
python -m voicedata build --config configs/local.yaml \
    --job sandpaper-20241223 --file 20241223

# 只重新导出，不加载模型也不检查模型
python -m voicedata build --config configs/local.yaml \
    --job sandpaper-20241223 --file 20241223 --stages export --threshold 0.45

# 强制重跑指定阶段
python -m voicedata build --config configs/local.yaml \
    --job sandpaper-20241223 --file 20241223 --stages vad,sv,asr,noise,export --force
```

常用参数：

- `--config`：配置文件，默认 `configs/server.yaml`；本地 0.6B 用 `configs/local.yaml`。
- `--job`：任务名，决定 `work/<job>/` 和 `out/<job>/`。
- `--file`：文件名子串过滤。
- `--start` / `--duration`：只处理一个时间片段，单位秒。
- `--stages`：`decode,vad,sv,asr,noise,export` 的子集。
- `--force`：重跑所选阶段。
- `--retry-failed`：把 failed/partial 的阶段和片段重新入队。
- `--limit N`：最多处理前 N 个文件。
- `--threshold`：只影响导出阈值。
- `--device` / `--dtype`：覆盖配置中的设备和精度。

## 查看与校验

```bash
# 查看每个文件的阶段状态
python -m voicedata status --job sandpaper-20241223
python -m voicedata status --job sandpaper-20241223 --json

# 校验 zip、FLAC、时间、索引和数据库记录
python -m voicedata validate --job sandpaper-20241223

# 生成人工抽听页（默认 80 段，sample=0 表示全部）
python -m voicedata inspect --job sandpaper-20241223 --sample 80
```

`calibration.html` 支持播放、按 SV/置信度查看片段，并可导出 CSV。页面按原音频时间顺序展示；`--sample N` 仍优先抽取与阈值接近的片段。

## 配置要点

`server.yaml` 是默认配置；`local.yaml` 只用于本地 CPU + 0.6B 验证。两者的过滤参数一致，主要区别是设备、精度、ASR 模型规模和 batch size。模型是否自动下载由所选阶段和 `asr.model_subdir` 决定。

| 配置 | 作用 |
| --- | --- |
| `vad.*` | FireRedVAD 分段 |
| `aed.*` | 歌声/音乐过滤与短句合并 |
| `sv.threshold` | 导出所需的最低声纹分，默认 0.40 |
| `sv.asr_min_score` | 进入 ASR/噪音的最低声纹分，默认 0.30 |
| `asr.*` | ASR 模型、语言、最大生成长度 |
| `noise.threshold` | `noisy` 判定阈值，默认 0.50 |
| `export.drop_fillers` / `filler_max_s` / `filler_texts` | 填充词过滤 |
| `export.min_asr_conf` | ASR 置信度下限，默认 -7.0；`null` 关闭 |
| `export.format` | `zip`（默认）或旧的 `json` |
| `export.include_scores` | 是否输出 `sv_score` / `noise_prob` / `asr_conf` |
| `run.cleanup_audio_after_export` | 导出后清理该文件的 work 音频缓存，默认 `true` |

参数修改后由阶段哈希自动决定哪些阶段需要重跑，不需要手工清理 `work/`。

## 当前数据

- 已清掉早期的 `smoke120`、`smoke300` 测试 job。
- 当前保留的完整样例：`out/sandpaper-20241223/`（用 `configs/local.yaml` + 0.6B 在本地跑的）
  - 93 段，242.4s 语音
  - 23 段 `noisy=1`，70 段 `noisy=0`
  - zip 约 4.4MB
  - `calibration.html` 默认抽取 80 段
- 对应状态：`work/sandpaper-20241223/state.db`；音频缓存已清理，work 目录约 250KB，可继续 inspect、重新导出或校验。

## 已知限制

- AED 负责过滤歌声/音乐；若歌词被 AED 误判为 speech，仍可能进入 ASR。
- 声纹是段级打分；主播和其他人/歌词重叠时，混合段可能通过阈值。
- `noisy` 只是标签，不会因为 BGM 而删除片段。
- `inspect` 只能看到已导出的片段；要看阈值以下的候选，先用较低 `--threshold` 重新导出。
