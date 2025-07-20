```bash
apt update && apt install python3-pip -y
```

```bash
mkdir llm-chatbot-rag
cd llm-chatbot-rag
```

```bash
cat >requirements.txt <<EOF
transformers>=4.37.0
torch>=2.1.0
langchain>=0.1.0
langchain-community>=0.0.10
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
accelerate>=0.25.0
einops>=0.7.0
jinja2>=3.1.0
tensorflow==2.16.1
tf-keras==2.16.0
EOF
```

```bash
pip install -r requirements.txt
```

```bash
mkdir documents
cd documents
wget -O - https://gitlab.practical-devsecops.training/-/snippets/67/raw/main/TechCorpXYZFiles.sh | bash
cd ..
```

```bash
wget -O llm-chatbot-rag.py https://gitlab.practical-devsecops.training/-/snippets/68/raw/main/llm-chatbot-rag.py
python3 llm-chatbot-rag.py
```

full code 

```bash
cat > llm-chatbot-rag.py <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

## Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

## Initiate tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

## Load documents from a given directory
directory_path = "documents"

# Check if given path is a directory, and not a file
if not os.path.isdir(directory_path):
    print(f"Error: '{directory_path}' is not a valid directory")

# List all files
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Load the contents of a file into an object named documents
documents = []
for filename in files:
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'r') as file:
        content = file.read()
        documents.append({"content": content})

## Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

texts = [doc["content"] for doc in documents]
#print("---------------------------------Texts: ---------------------------------\n")
#print(texts)
split_texts = text_splitter.create_documents(texts)
#print("---------------------------------Split_texts: ---------------------------------\n")
#print(split_texts)

# Create vector store
vectorstore = FAISS.from_documents(split_texts, embeddings)

## Create text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

## While loop for continous chat
while True:
    print("+" *50)
    user_input = input("\033[92mType your message. Type 'X' or 'x' to exit.\033[0m")
    if user_input in ['X', 'x']:
        print("Exiting.")
        break
    else:
        query = user_input
        ## Use the user input and retrieve relevant documents
        #relevant_docs = vectorstore.similarity_search(query, k=3)
        relevant_docs = vectorstore.similarity_search(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context (context contains relevant text from documents)
        prompt = f"""Context: {context}
        
        Question: {query}
        
        Answer based on the context provided:"""
        
        print("\033[95mPrompt with context: \033[0m\n" + prompt)
        print("---------------------------------Calling LLM---------------------------------")
        messages = [{"role": "user", "content": prompt}]
        output = generator(messages)
        print("+" *50)
        print("\033[94mAI message: \033[0m" + output[0]["generated_text"])
        print("+" *50)
EOF
```

```bash
python3 llm-chatbot-rag.py
```

```bash
Revenue 2023
```
