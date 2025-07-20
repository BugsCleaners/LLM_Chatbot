# Description

```bash
apt update && apt install python3-pip -y
mkdir llm-chatbot
cd llm-chatbot
cat >requirements.txt <<EOF
transformers>=4.37.0
torch>=2.1.0
accelerate>=0.25.0
einops>=0.7.0
jinja2>=3.1.0
EOF
pip install -r requirements.txt
```

```bash
cat > tok.py <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    )
    
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

user_input = "How much is a gazillion?"
user_input_as_tokens = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)

model_output = model.generate(input_ids=user_input_as_tokens, max_new_tokens=50)
print(tokenizer.decode(model_output[0]))

EOF
```

```bash
python3 tok.py
```

```bash
cat > tok.py <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    )
    
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

user_input = "How much is a gazillion?"
user_input_as_tokens = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)

model_output = model.generate(input_ids=user_input_as_tokens, max_new_tokens=50)
print(tokenizer.decode(model_output[0]))

user_input = "How much is a gazillion?<|assistant|>";
user_input_as_tokens = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device);
model_output = model.generate(input_ids=user_input_as_tokens, max_new_tokens=50);

print(tokenizer.decode(model_output[0]));

print(user_input_as_tokens);

for id in user_input_as_tokens[0]:
    print(tokenizer.decode(id));

for id in model_output[0]:
    print(tokenizer.decode(id));

print(model_output[0]);

print(tokenizer.decode(29900));
print(tokenizer.decode(29892));

EOF
```

```bash
python3 tok.py
```

```bash
cat > tok.py <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    )
    
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

user_input = "What is a gazebo? <|assistant|>";
user_input_as_tokens = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device);
print(user_input_as_tokens);

user_input = "Which country is indigenous to gazelles? <|assistant|>";
user_input_as_tokens = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device);
print(user_input_as_tokens);

print(tokenizer.decode(12642));

EOF
```

```bash
python3 tok.py
```

```bash
cat > tok.py <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    );
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct");

print(tokenizer.decode(12642));

EOF
```

```bash
python3 tok.py
```
