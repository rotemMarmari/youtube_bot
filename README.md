# YouTube Video Summarizer & Q&A Bot

This project is a Python-based tool that **summarizes YouTube videos and allows users to ask questions about their content**, using the video transcript, semantic search with FAISS, and a locally running Large Language Model (LLM) via Ollama.

The application includes a simple **Gradio web interface** for interactive usage.

---

## Features

* Fetches YouTube video transcripts (manual transcripts are prioritized over auto-generated ones)
* Generates concise video summaries using an LLM
* Enables question answering based on video content
* Uses semantic search (FAISS + embeddings) to retrieve relevant transcript chunks
* Runs fully locally using Ollama (no external LLM APIs)
* Simple and clean Gradio UI

---

## Tech Stack

* **Python**
* **youtube-transcript-api** – transcript extraction
* **LangChain** – prompt management and LLM chains
* **FAISS** – vector similarity search
* **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
* **Ollama** – local LLM runtime
* **Gradio** – web interface

---

## How It Works

1. The YouTube transcript is fetched and preprocessed.
2. The transcript is split into overlapping chunks.
3. Chunks are embedded using a HuggingFace embedding model.
4. A FAISS index is built for semantic similarity search.
5. * **Summarization**: The full transcript is passed to the LLM with a structured prompt.
   * **Q&A**: Relevant chunks are retrieved from FAISS and passed to the LLM as context.
6. Results are displayed through a Gradio interface.

---

## Prerequisites

* Python 3.9+
* Ollama installed and running locally
* LLaMA model pulled via Ollama

```bash
ollama pull llama3.2:3b
```

Note: If your system has sufficient resources, you can use a larger LLaMA model to improve summary and Q&A quality.

---

## Installation

```bash
git clone https://github.com/rotemMarmari/youtube_bot.git
cd youtube_bot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the App

```bash
python gradio_UI.py
```

The app will be available at:

```
http://localhost:7860
```

---

## Usage

1. Enter a YouTube video URL.
2. Click **"Summarize Video"** to generate a concise summary.
3. Ask any question related to the video and click **"Ask a Question"**.

<img width="1846" height="807" alt="צילום מסך 2025-12-17 121641" src="https://github.com/user-attachments/assets/c1eb7dc8-6b6f-4010-896f-e4a0aac3f5ec" />

---

## Limitations

* Only works with videos that have available transcripts
* Currently supports English transcripts only
* Designed for local usage, not production deployment

---

## Future Improvements

* Support additional languages
* Persist FAISS indexes for reuse
* Add timestamp-aware answers
* Improve error handling and transcript availability feedback
* Dockerize the application

---

## Disclaimer

This project is for educational and experimental purposes.
It relies on YouTube transcripts and local LLM inference accuracy.
