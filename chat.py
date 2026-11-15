#Uninstall any outdated versons of torchao that will cause the program to crash.
!pip uninstall -y torchao
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"
ADAPTER_PATH = "/kaggle/input/datasets/x3n0shad3/jerryai-database"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
model.eval()

def chat():
    history = [
        {"role": "system", "content": "You are Jerry, a helpful AI assistant created by X3N0SHAD3."}
    ]
    print("Chatting with Jerry. Type 'exit', 'quit' or 'q' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        if not user_input:
            continue
            
        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        history.append({"role": "assistant", "content": response_text})
        print(f"Jerry: {response_text}\n")

if __name__ == "__main__":
    chat()