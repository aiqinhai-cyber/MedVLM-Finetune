import os
import sys
import json
import torch
import jieba
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from unsloth import FastVisionModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def compute_metrics(ground_truth, prediction):
    """计算大模型常用的自然语言生成指标 (中文适用)"""
    # 使用 jieba 进行中文分词
    gt_tokens = list(jieba.cut(ground_truth))
    pred_tokens = list(jieba.cut(prediction))
    
    # 1. 计算 BLEU-4 (关注生成的精准度，即模型生成的话在标准答案里命中了多少)
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
    
    # 2. 计算 ROUGE-L (关注召回率，即最长公共子序列，评估句子结构的完整度)
    # ROUGE 库默认按空格分词，我们把切好的中文词用空格连起来骗过它
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_res = scorer.score(" ".join(gt_tokens), " ".join(pred_tokens))
    
    return bleu_score, rouge_res['rougeL'].fmeasure

def main():
    print("=== 1. 加载本地测试集 ===")
    try:
        test_dataset = load_from_disk("data/test_dataset")
    except:
        print("未找到测试集！请先运行最新的 scripts/train.py 生成切分数据。")
        return

    print("=== 2. 组装 AI 放射科医生 (Base + LoRA) ===")
    model, tokenizer = FastVisionModel.from_pretrained(
        "lora_model",
        load_in_4bit=True,
        device_map="auto"
    )
    # 开启 Unsloth 2倍速推理优化
    FastVisionModel.for_inference(model)
    
    results = []
    bleu_list = []
    rouge_list = []

    print(f"\n=== 3. 开始自动化量化评估 (共 {len(test_dataset)} 条 X光片) ===")
    # 为了测试，，这里只跑前 20 条
    for idx, example in enumerate(tqdm(test_dataset)):
        # 解析数据结构提取内容
        user_content = example["messages"][0]["content"]
        assistant_content = example["messages"][1]["content"]
        
        # 获取图像、指令和真实医生的标准诊断
        image = user_content[0]["image"]
        instruction = user_content[1]["text"]
        ground_truth = assistant_content[0]["text"]

        # 构建对话 Prompt
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # 将输入送入模型 (由于是纯推理，关闭梯度计算省显存)
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(images=image, text=input_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        # 解码模型的预测结果
        input_len = inputs.input_ids.shape[1]
        prediction = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # 计算单条数据的指标
        bleu, rouge_l = compute_metrics(ground_truth, prediction)
        bleu_list.append(bleu)
        rouge_list.append(rouge_l)

        # 保存对齐日志
        results.append({
            "id": idx,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge_l, 4)
        })

    # === 4. 输出最终成绩单 ===
    avg_bleu = np.mean(bleu_list)
    avg_rouge = np.mean(rouge_list)
    
    print("\n" + "="*40)
    print("医疗多模态微调最终量化报告 ")
    print("="*40)
    print(f"评估样本数: {len(results)} 张医疗图像")
    print(f"平均 BLEU-4 得分:  {avg_bleu:.4f} (衡量医学术语精准度)")
    print(f"平均 ROUGE-L 得分: {avg_rouge:.4f} (衡量句式逻辑连贯性)")
    print("="*40)

    # 导出报表
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"BLEU": avg_bleu, "ROUGE_L": avg_rouge},
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print("💾 详细测试报告已导出至 outputs/eval_report.json")

if __name__ == "__main__":
    main()