import yaml
import torch
from unsloth import FastVisionModel

class VisionModelBuilder:
    def __init__(self, config_path="configs/train_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def load_and_prepare_model(self):
        print(f"正在加载基础模型: {self.config['model']['name_or_path']} ...")
        
       
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=self.config['model']['name_or_path'],
            load_in_4bit=self.config['model']['load_in_4bit'],
            use_gradient_checkpointing="unsloth",
            
            # 强行指定模型全部加载到第一张 GPU (cuda:0) 上，禁止系统自作主张卸载到 CPU
            device_map={"": 0}, 
            
            # 限制大模型的最大理解长度。7B 模型的 KV Cache 极其吃显存，
            # 限制在 2048 个 token 完全足够处理医学图像和简短报告！
            max_seq_length=4096, 
        )
        
        print("基础模型加载完成。正在注入 LoRA 适配器...")
        model = FastVisionModel.get_peft_model(
            model,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            bias=self.config['lora']['bias'],
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True
        )
        
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"当前 GPU: {gpu_stats.name} | 显存容量: {round(gpu_stats.total_memory/1024**3, 3)} GB")
        print(f"模型加载后已占用显存: {start_gpu_memory} GB")
        
        return model, tokenizer