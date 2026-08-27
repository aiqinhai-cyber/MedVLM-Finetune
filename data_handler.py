import os
from datasets import load_dataset, load_from_disk

class MedicalDataBuilder:
    def __init__(self, dataset_name, instruction=None):
        self.dataset_name = dataset_name
        self.default_instruction = instruction
        
        # 智能检测：如果传入的是本地路径，就直接加载；如果是线上名字，就联网下载
        if os.path.exists(self.dataset_name):
            print(f"正在加载本地工程化增强数据集: {self.dataset_name}")
            self.dataset = load_from_disk(self.dataset_name)
        else:
            print(f"正在联网拉取开源数据集: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name, split="train")

    def _format_function(self, example):
        # 核心逻辑：优先使用我们在数据工程中生成的“多样化指令”，如果没有才用默认的
        current_instruction = example.get("instruction", self.default_instruction)
        
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": current_instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["text"]}
                    ]
                }
            ]
        }

    def get_formatted_dataset(self):
        # 映射成大模型认识的对话格式，并移除多余列节约内存
        return self.dataset.map(self._format_function, remove_columns=self.dataset.column_names)