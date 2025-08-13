# LLM
Langchain and RAG Technique Project 

# 📚 PDF Question Answering with LangChain & ALLaM

## 🚀 Overview
This project demonstrates how to build a **PDF Question Answering** system using [LangChain](https://www.langchain.com/), semantic search, and a **fully open-source large language model (LLM)**.

The basic idea:
1. Read a PDF document.
2. Split the text into smaller chunks.
3. Convert the text into embeddings (vector representations).
4. Use semantic search to find the most relevant chunks.
5. Generate answers in **Arabic** or **English**.

---

## 🛠 Initial Approach (OpenAI API)
The first version used **OpenAI** for both embeddings and the LLM:

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
```
## ❌ Problem
Using OpenAI requires:
- A valid API key (`OPENAI_API_KEY`).
- Internet access to call their servers.
- Pay-per-request charges.

If the API key is missing, expired, or incorrect, the application will fail.  
Additionally, OpenAI models are **closed-source**, meaning no local hosting or customization.

---

## ✅ Open-Source Solution
To avoid API key issues and gain full control, the project was updated to use **completely open-source** models.

### 🔹 Embeddings
**[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)**
- Supports multiple languages including Arabic.
- Runs locally on CPU or GPU.
- No API key required.

### 🔹 Language Model
**[`ALLaM-AI/ALLaM-7B-Instruct-preview`](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview)**
- Arabic-English capable.
- Can run fully offline.
- Fine-tuning is possible for domain-specific use cases.

---

## 💡 Why Open-Source is Better
- **No API Key** – No external authentication needed.
- **Offline Ready** – Works without internet access.
- **Data Privacy** – Your documents never leave your machine.
- **Customizable** – Can be fine-tuned for higher accuracy.
- **Cost-Free** – No usage charges per request.

---

## ⚙️ Workflow
1. **PDF Reading** – Extract text using `PyPDF2` or LangChain loaders.
2. **Text Splitting** – Use `CharacterTextSplitter` to break text into chunks.
3. **Embeddings** – Generate vector representations with `multilingual-e5-base`.
4. **Vector Store** – Store embeddings in FAISS for fast similarity search.
5. **Retrieval** – Find the most relevant chunks for a query.
6. **Answer Generation** – Feed retrieved text to `ALLaM-7B-Instruct` for the final answer.

## 📦 Installation
```python
pip install langchain
pip install pypdf
pip install faiss-cpu
pip install transformers
pip install sentence-transformers
```
## ▶️ Usage
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 1. Load and split PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = splitter.split_documents(documents)

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = FAISS.from_documents(texts, embeddings)

# 3. Search for relevant chunks
query = "What are the admission requirements?"
docs = vectorstore.similarity_search(query)

# 4. Load ALLaM model
qa_pipeline = pipeline(
    "text-generation",
    model="ALLaM-AI/ALLaM-7B-Instruct-preview",
    device_map="auto"
)

# 5. Generate answer
context = " ".join([d.page_content for d in docs])
answer = qa_pipeline(f"Answer the question based on context:\n{context}\nQuestion: {query}")
print(answer[0]['generated_text'])
```
---

### 📌 Summary of Steps
1. **Document Source** – Load PDF or other files.  
2. **Text Splitting** – Break into chunks for efficient search.  
3. **Embedding Model** – Convert chunks to vector embeddings.  
4. **Vector Store** – Save embeddings in FAISS.  
5. **User Query** – Input question.  
6. **Query Embedding** – Convert query to vector.  
7. **Similarity Search** – Find top matches from FAISS.  
8. **Context Injection** – Combine matches + query.  
9. **LLM Generation** – ALLaM model produces grounded answer.  
 
---

## 📚 References
- [LangChain Documentation](https://python.langchain.com/)  
- [Hugging Face Models](https://huggingface.co/models)  
- [ALLaM-AI/ALLaM-7B-Instruct-preview](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview)  
- [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)  


