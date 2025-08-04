
# 📄 DocuMind

**DocuMind** is an intelligent document processing and question-answering API built using FastAPI. It allows you to upload PDF files, process and chunk their contents, store embeddings in a vector database, and interact with the documents using summarization and question-answering capabilities.

---

## 🚀 Features

- Upload and process PDF files
- Chunk and vectorize documents
- Summarize content using an LLM
- Ask contextual questions (RAG)
- Perform semantic similarity searches
- Generate visual analysis from document queries

---

## 🛠️ Setup & Installation

### 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn app:app --reload
```

### 3. Access the API Documentation

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Endpoints

### 1. **Upload PDF**

- **POST** `/upload`
- **Description**: Uploads a PDF file to the server.
- **Request**: `multipart/form-data`
- **Response**:
  ```json
  {
    "filename": "example.pdf"
  }
  ```

---

### 2. **Process PDF**

- **POST** `/process/{filename}`
- **Description**: Extracts text, chunks it, and stores in the vector database.
- **Response**:
  ```json
  [
    "uuid1",
    "uuid2",
    ...
  ]
  ```

---

### 3. **Summarize Document**

- **POST** `/summarize`
- **Body**:
  ```json
  {
    "document_name": "example"
  }
  ```
- **Response**:
  ```json
  {
    "summary": "This is the summary...",
    "document_name": "example"
  }
  ```

---

### 4. **Ask a Question (RAG)**

- **POST** `/ask`
- **Body**:
  ```json
  {
    "document_name": "example",
    "query": "What is the key finding?"
  }
  ```
- **Response**:
  ```json
  {
    "summary": "The document states...",
    "document_name": "example"
  }
  ```

---

### 5. **Ask with Chart Context**

- **POST** `/chartwith`
- **Same body and response format as `/ask`**

---

### 6. **Query Document (Vector Search)**

- **POST** `/query`
- **Same body and response format as `/ask`**


## ⚙️ Configuration

All configuration settings can be managed through `app/configuration/config.py`.

