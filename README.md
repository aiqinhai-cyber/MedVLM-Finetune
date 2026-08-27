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
📦 MedVLM-Finetune
 ┣ 📂 configs
 ┃ ┗ 📜 train_config.yaml           # 模型、数据及超参数配置文件
 ┣ 📂 scripts
 ┃ ┣ 📜 data_engineering.py         # 核心数据增强、知识蒸馏与负样本融合脚本
 ┃ ┣ 📜 train.py                    # 极限微调训练脚本 (LoRA)
 ┃ ┗ 📜 evaluate_quantitative.py    # 自动化评估与指标计算脚本 (BLEU/ROUGE)
 ┣ 📂 src
 ┃ ┣ 📜 data_handler.py             # 数据加载与多态指令格式化适配器
 ┃ ┣ 📜 model_builder.py            # 视觉模型组装与显存优化层
 ┃ ┗ 📜 inference.py                # 单图推理脚本
 ┗ 📜 download_model.py             # 断点续传与防断连下载器

快速开始
1. 环境准备
本项目基于最新的 PyTorch Nightly 构建以支持最新显卡架构。

Bash
# 建议使用 Python 3.10
pip install openai opencv-python datasets rouge-score nltk jieba
2. 执行数据工程 (Data Engineering)
该步骤将自动进行 CLAHE 图像增强、困难负样本融合与大模型知识抽取。

Bash
python scripts/data_engineering.py
注：需在脚本中配置自己的 DashScope API Key。生成的数据将存储在 data/engineered_dataset。

3. 启动模型微调 (Fine-tuning)
调用 src/model_builder.py 动态加载 4-bit 模型并注入 LoRA 适配器。

Bash
python scripts/train.py
微调完成后的外挂权重将安全保存至 lora_model 目录。

4. 量化评估 (Evaluation)
对预留的独立测试集进行推理，并计算自然语言生成指标。

Bash
python scripts/evaluate_quantitative.py
评估报告及明细将导出至 outputs/eval_report.json。