import torch
from datasets import load_dataset
from rich import print_json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "codingmonster1234/chess-sft-modelv2"
subfolder = "checkpoint-168"

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=subfolder,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # Keeps it fast without the version mismatch error
)

test_set = load_dataset("codingmonster1234/chess-reasoning-sft", split="test")
example = test_set[0]
print_json(data=example)

messages = example["prompt"]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

# 3. Generate the Reasoning + Move
output = model.generate(
    **inputs,
    max_new_tokens=2048,  # Enough space for the <think> block
    do_sample=False,  # Set to False for "Greedy" (more deterministic) moves
    temperature=None,  # Not used when do_sample=False
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
