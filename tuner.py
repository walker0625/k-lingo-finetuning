import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

import subprocess
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    print("📦 python-dotenv가 없습니다. 설치를 시작합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# 1. 환경 변수 및 경로 설정
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_level1.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "result", "adaptor")

# 2. 모델 로드 (RTX 4090 최적화)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 3. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. 포맷팅 함수
def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples["input"])):
        text = (
            f"<|im_start|>system\n{examples['instruction'][i]}<|im_end|>\n"
            f"<|im_start|>user\n{examples['input'][i]}<|im_end|>\n"
            f"<|im_start|>thought\n{examples['thought'][i]}<|im_end|>\n"
            f"<|im_start|>call\n{examples['tool_call'][i]}<|im_end|>\n"
            f"<|im_start|>observation\n{examples['observation'][i]}<|im_end|>\n"
            f"<|im_start|>assistant\n{examples['output'][i]}<|im_end|>\n"
        )
        texts.append(text)
    return { "text" : texts }

# 5. 데이터 준비
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. 트레이너 설정 (RTX 4090용 bf16 활성화)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
)

# 7. 실행 및 저장
trainer.train()
model.save_pretrained(OUTPUT_DIR)

# 8. 허깅페이스 업로드
if HF_TOKEN and HF_REPO_ID:
    print(f"☁️ Uploading to Hugging Face: {HF_REPO_ID}")
    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)