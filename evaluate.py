import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.data_handler import MedicalDataBuilder
from src.inference import VLMInferencer

def main():
    with open("configs/train_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=== 获取测试图像 ===")
    data_builder = MedicalDataBuilder(config['data']['dataset_name'], config['data']['instruction'])
    dataset = data_builder.get_formatted_dataset()
    test_image = dataset[0]['messages'][0]['content'][1]['image']

    print("=== 加载微调模型 ===")
    # 推理时指向刚才保存的 lora_model，Unsloth 会自动把外挂套在基础模型上
    inferencer = VLMInferencer(model_path="lora_model")
    
    print("=== 开始推理 ===")
    inferencer.predict(image=test_image, instruction=config['data']['instruction'])

if __name__ == "__main__":
    main()