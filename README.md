# RAG Chatbot Backend with Flask
This Flask-based backend application allows you to process PDF documents, generate embeddings, and engage in interactive chat sessions using Open Source LLMs. The service provides endpoints to start chat sessions, send queries, and retrieve chat history.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [API Documentation](#api-documentation)
- [Potential Improvements](#potential-improvements)

## Setup Instructions

### 1. Prerequisites

Ensure you have the following installed:
- **Python 3.8+**: Download from [Python.org](https://www.python.org/downloads/)
- **Pip**: Python package installer (comes with Python 3.4+)

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/GodReaper/rag-chatbot-backend-with-flask.git
cd pdf-chatbot-backend
```

 ### 3.Install Required Packages
 Install the necessary Python packages using Pip:
 ```bash
 pip install -r requirements.txt
 ```
 ### 4.Run the application
 Start the Flask server:
 ```bash
 python app.py
 ```

The server will be accessible at http://0.0.0.0:8080.


## API Reference

#### Process PDF Document

```http
  POST /api/documents/process
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `file` | **Required**. The PDF document to be processed. Upload this file in the form-data section of the request body. |

#### Get item

```http
  POST /api/chat/start
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `assest_id`      | `string` | **Required**. The unique identifier of the asset related to the chat session. This ID is generated after uploading the document. |


```http
  POST /api/chat/message
```
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `chat_thread_id`      | `string` | **Required**.The ID of the chat thread where the message is sent. This ID is obtained from the start chat route.
| `query`      | `string` | **Required** The user's message or query to be processed.

```http
  GET /api/chat/history
```
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `chat_thread_id`      | `string` | **Required**. The ID of the chat thread for which the history is requested. This ID is obtained from the start chat route.

## Features

- Process PDF Document
- Automatic Text Chunking
- Embedding Generation
- Query Processing
- Contextual Responses
- Source Tracking
- Historical Data Retrieval
- Conversation Tracking


## Potential Improvements

* **User Authentication:** Implement user authentication and authorization to secure API endpoints.
* **Chat Thread Management:** Add features to manage chat threads, such as deletion or archiving.
* **Scalability:** Explore options to scale the backend using cloud services or containerization (e.g., Docker).
* **Caching:** Implement caching for frequently accessed documents to improve performance.
* **Frontend Integration:** Develop a frontend application to provide a complete user experience.
* **Advanced Query Capabilities:** Enhance query support to handle more complex queries or multi-document retrieval.
