from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_ID = "kakaocorp/kanana-1.5-2.1b-instruct-2505"

# 모델 초기화 (서버 시작 시 1회 로드)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    temperature=0.7,
    repetition_penalty=1.1,
)
