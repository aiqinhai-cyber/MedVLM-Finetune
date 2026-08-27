from unsloth import FastVisionModel
from transformers import TextStreamer

class VLMInferencer:
    def __init__(self, model_path="merged_model"):
        print(f"正在加载微调后的模型: {model_path} ...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=True
        )
        FastVisionModel.for_inference(self.model)

    def predict(self, image, instruction):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors='pt'
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        print("\n[AI 放射科医生诊断报告]:")
        _ = self.model.generate(
            **inputs, 
            streamer=text_streamer, 
            max_new_tokens=128, 
            temperature=1.5, 
            min_p=0.1
        )