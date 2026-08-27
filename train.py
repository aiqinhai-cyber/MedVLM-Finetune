import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_handler import MedicalDataBuilder
from src.model_builder import VisionModelBuilder
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict

def main():
    print("=== 第一步：初始化模型与配置 ===")
    model_builder = VisionModelBuilder(config_path="configs/train_config.yaml")
    model, tokenizer = model_builder.load_and_prepare_model()
    config = model_builder.config

    print("\n=== 第二步：准备与切分数据集 ===")
    data_builder = MedicalDataBuilder(
        dataset_name=config['data']['dataset_name'],
        instruction=config['data']['instruction']
    )
    full_dataset = data_builder.get_formatted_dataset()

    
    # 切分数据集，严防数据泄露，这里我预留 50 条数据作为测试集，剩下的用于训练
    print("正在进行 Train/Test 切分...")
    splits = full_dataset.train_test_split(test_size=50, seed=42)
    train_dataset = splits["train"]
    test_dataset = splits["test"]
    
    # 将测试集保存到本地，供评估脚本读取
    os.makedirs("data", exist_ok=True)
    test_dataset.save_to_disk("data/test_dataset")
    print(f"数据切分完成！训练集: {len(train_dataset)}条 | 测试集: {len(test_dataset)}条 (已保存至 data/test_dataset)")

    print("\n=== 第三步：配置训练器 ===")
    data_collator = UnslothVisionDataCollator(model, tokenizer) 

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset, # 现在只用切分后的训练集训练
        args=SFTConfig(
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            max_steps=config['training']['max_steps'],
            learning_rate=config['training']['learning_rate'],
            optim=config['training']['optim'],
            output_dir=config['training']['output_dir'],
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            report_to="none"
        )
    )

    print("\n=== 第四步：开始极限微调训练 ===")
    trainer_stats = trainer.train()
    print(f"训练完成！统计信息: {trainer_stats}")
    
    print("\n=== 第五步：保存 LoRA 适配器 ===")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("模型外挂已成功且安全地保存至 lora_model 目录！")

if __name__ == "__main__":
    main()