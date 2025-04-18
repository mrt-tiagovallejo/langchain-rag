# Langchain RAG example using Ollama for embeddings and inference

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using [Langchain](https://www.langchain.com/) and [Ollama](https://ollama.com/) for local text embeddings.

Originally from [pixegami/langchain-rag-tutorial](https://github.com/pixegami/langchain-rag-tutorial), I’ve adapted it to use Ollama with the [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) model for local embeddings, so you can experiment with it entirely offline—no OpenAI API keys needed.

It loads Markdown documents, splits them into chunks, embeds them with `nomic-embed-text`, and stores them in a Chroma vector database for efficient querying. When a user sends a query, it retrieves relevant chunks and uses `llama3` via Ollama to generate a contextual answer locally.

For more details please watch: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).

## ⚙️ Install dependencies

1. 💡 Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.

2. 🐍 Create a virtual environment

```
py -3.12 -m venv venv
venv\Scripts\activate
python --version
```
Should show 3.12.x

3. 📦 Install project dependencies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

4. 📄 Install markdown dependencies with: 

```python
pip install "unstructured[md]"
```

5. 🦙 Install and configure **Ollama** (for local embeddings and inference):

    - Download and install Ollama from https://ollama.com/download
    - Pull the embedding and inferece models:
    ```bash
    ollama pull nomic-embed-text
    ollama pull llama3:8b
    ```

## 🛠️ Create the vector database

Build the local Chroma database using Ollama for embeddings.

```python
python create_database.py
```

## 🔍 Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```