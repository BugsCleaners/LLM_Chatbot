# Lab Description



```bash
apt update && apt install python3-pip -y
```

```bash
mkdir llm-chatbot
cd llm-chatbot
```

```bash
cat >requirements.txt <<EOF
transformers>=4.37.0
torch>=2.1.0
accelerate>=0.25.0
einops>=0.7.0
jinja2>=3.1.0
EOF
```

```bash
pip install -r requirements.txt
```

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )
    
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

from transformers import pipeline

generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
        )
    
print("What do you want?")
user_input = input()

messages = [{"role":"user", "content": user_input}]
response = generator(messages)
print(response[0]["generated_text"])
```

With loop

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )
    
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

from transformers import pipeline

generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
        )

while True:
    print("-" *50) # Horizontal line
    print("What do you want?")
    user_input = input("\033[92mType something, or X to exit: \033[0m") # Ask the user for input in green color
    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        messages = [{"role":"user", "content": user_input}]
        response = generator(messages)
        print(response[0]["generated_text"])
```

```bash
python3 llm-chatbot.py
```

```bash
microsoft/Phi-3-mini-4k-instruct 
TinyLlama/TinyLlama-1.1B-Chat-v1.0.
```

```bash
cat>llm-chatbot.py<<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        )

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

from transformers import pipeline

generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
        )

while True:
    print("-" *50) # Horizontal line
    print("What do you want?")
    user_input = input("\033[92mType something, or X to exit: \033[0m") # Ask the user for input in green color
    if user_input in ['X', 'x']: # If user types X or x, exit the program
        print("Exiting.")
        break
    else:
        messages = [{"role":"user", "content": user_input}]
        response = generator(messages)
        print(response[0]["generated_text"])
EOF
```
