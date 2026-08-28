MedVLM-Finetune: 医疗多模态大模型极限微调与数据工程
本项目致力于在消费级受限算力（8GB VRAM）下，完成前沿多模态大模型（Qwen2.5-VL-3B）在放射学影像诊断任务上的端到端指令微调。项目不仅打通了模型训练的完整闭环，更在底层算力适配与高阶数据工程上进行了深度优化。
只有本地的RTX5060显卡可用，所以选的模型比较小

核心技术亮点
显存优化与硬件适配 

在单卡 RTX 5060 (8GB VRAM) 的严苛限制下，通过 NF4 4-bit 量化、Paged-AdamW 优化器内存卸载与 Unsloth 算子融合，成功将训练显存峰值控制在 6GB 内。

解决了 PyTorch 官方对 NVIDIA 最新 RTX 50 系（sm_120架构）CUDA Kernel 缺失的底层兼容性难题。

数据工程流水线 

LLM 知识蒸馏 (Knowledge Distillation)：调用 Qwen-Turbo API 提取原始诊断报告中的“疾病-影像特征”三元组，作为先验知识注入多模态 Prompt。

困难负样本挖掘 (Hard Negative Mining)：跨域引入 hf-vision/chest-xray-pneumonia 官方数据集中的完全健康胸片，强制模型学习“无异常”表述，显著降低医疗事实幻觉与假阳性率。该数据集是开源的。

医学视觉特征工程 (Visual Feature Engineering)：运用 CLAHE (限制对比度自适应直方图均衡化) 算法对原始 X 光片进行局部对比度增强，凸显骨骼与肺部结节边缘。

严谨的自动化量化评估 (Quantitative Evaluation)

构建了防数据泄露的 Train/Test 严格切分流水线。

引入自然语言生成领域的标准指标 BLEU-4（专注医学术语的精确命中）与 ROUGE-L（评估诊断句式的逻辑连贯性），用量化数据验证微调效果。

📂 项目结构
Plaintext


```
```
MedVLM-Finetune
 ├ configs
 │ └ train_config.yaml # 模型、数据及超参数配置文件
 ├ scripts
 │ ├ data_engineering.py # 核心数据增强、知识蒸馏与负样本融合脚本
 │ ├ train.py # 极限微调训练脚本 (LoRA)
 │ └ evaluate_quantitative.py # 自动化评估与指标计算脚本 (BLEU/ROUGE)
 ├ src
 │ ├ data_handler.py # 数据加载与多态指令格式化适配器
 │ ├ model_builder.py # 视觉模型组装与显存优化层
 │ └ inference.py # 单图推理脚本
 └ download_model.py # 断点续传与防断连下载器
```
```

快速开始

1. 环境准备
   本项目基于最新的 PyTorch Nightly 构建以支持最新显卡架构。

Bash

# 建议使用 Python 3.10

pip install openai opencv-python datasets rouge-score nltk jieba

执行数据工程 (Data Engineering)
该步骤将自动进行 CLAHE 图像增强、困难负样本融合与大模型知识抽取。

Bash
python scripts/data_engineering.py
注：需在脚本中配置自己的 DashScope API Key。生成的数据将存储在 data/engineered_dataset。

启动模型微调 (Fine-tuning)
调用 src/model_builder.py 动态加载 4-bit 模型并注入 LoRA 适配器。

Bash
python scripts/train.py
微调完成后的外挂权重将安全保存至 lora_model 目录。

量化评估 (Evaluation)
对预留的独立测试集进行推理，并计算自然语言生成指标。

Bash
python scripts/evaluate_quantitative.py
评估报告及明细将导出至 outputs/eval_report.json。

# 这里是需要的库及其版本

```
```
Package            Version
------------------ ---------
absl-py            2.5.0
accelerate         1.14.0
aiohappyeyeballs   2.7.1
aiohttp            3.14.3
aiosignal          1.4.0
annotated-doc      0.0.5
annotated-types    0.8.0
anyio              4.14.2
async-timeout      5.0.1
attrs              26.1.0
bitsandbytes       0.50.1
certifi            2026.7.22
charset-normalizer 3.5.1
click              8.4.2
colorama           0.4.6
cut-cross-entropy  25.1.1
datasets           4.3.0
defusedxml         0.7.1
diffusers          0.40.0
dill               0.4.0
docstring_parser   0.18.0
exceptiongroup     1.3.1
filelock           3.32.4
frozenlist         1.8.0
fsspec             2025.9.0
h11                0.16.0
hf_transfer        0.1.9
hf-xet             1.6.0
httpcore           1.0.9
httpcore2          2.12.0
httpx              0.28.1
httpx2             2.12.0
huggingface_hub    1.28.0
idna               3.19
importlib_metadata 9.0.0
jieba              0.42.1
Jinja2             3.1.6
jiter              0.16.0
joblib             1.5.3
markdown-it-py     4.2.0
MarkupSafe         3.0.3
mdurl              0.1.2
mpmath             1.3.0
msgspec            0.21.1
multidict          6.7.1
multiprocess       0.70.16
nest-asyncio       1.6.0
networkx           3.4.2
nltk               3.10.3
numpy              2.2.6
openai             3.5.0
opencv-python      5.0.0.93
opentelemetry-api  1.44.0
packaging          26.3
pandas             2.3.3
peft               0.20.0
pillow             12.3.0
pip                26.1.2
platformdirs       4.11.4
propcache          0.5.2
protobuf           7.36.0
psutil             7.2.2
pyarrow            25.0.1
pydantic           2.13.4
pydantic_core      2.46.4
Pygments           2.21.0
python-dateutil    2.9.0.post0
pytz               2026.3.post1
PyYAML             6.0.3
regex              2026.7.19
requests           2.34.2
rich               15.0.0
rouge_score        0.1.2
safetensors        0.8.0
sentencepiece      0.2.2
sentry-sdk         2.68.1
setuptools         81.0.0
shellingham        1.5.4
six                1.17.0
sniffio            1.3.1
structlog          26.1.0
sympy              1.14.0
tokenizers         0.22.2
torch              2.12.0.dev20260408+cu128
torchvision        0.27.0.dev20260407+cu128
tqdm               4.70.0
transformers       5.5.0
triton-windows     3.7.1.post27
trl                0.24.0
truststore         0.10.4
typeguard          4.6.0
typer              0.27.1
typing_extensions  4.16.0
typing-inspection  0.4.4
tyro               1.0.16
tzdata             2026.3
unsloth            2026.8.19
unsloth_zoo        2026.8.13
urllib3            2.7.0
wandb              0.28.2
wheel              0.47.0
xformers           0.0.35
xxhash             4.0.1
yarl               1.24.5
zipp               4.1.0
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
```
```