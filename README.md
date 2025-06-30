#  HR Resource Query Chatbot

A smart chatbot for HR teams to query employee profiles using semantic search and modern AI embeddings.

##  Overview

This chatbot helps HR managers quickly find the best-matched employees based on:
- Skills
- Minimum experience
- Availability
- Free-text queries (e.g. “healthcare projects”)

Built with:
- Sentence Transformers for embeddings
- Semantic similarity search
- Gradio for web interface

##  Features

✅ Free-text search  
✅ Skill filter  
✅ Minimum experience filter  
✅ Availability filter (available / busy)  
✅ Natural language output describing matching employees  
✅ Deployable to Hugging Face Spaces

##  Architecture

- **Frontend/UI**: Gradio web app
- **Embeddings**: `all-MiniLM-L6-v2` model via `sentence-transformers`
- **Similarity search**: Cosine similarity with sklearn
- **Data storage**: In-memory pandas DataFrame

##  Setup & Installation

Clone the repo:

```bash
git clone https://github.com/your-username/humanresource-chatbot.git
cd humanresource-chatbot

Install dependencies:
pip install -r requirements.txt

Run locally:
python app.py

🔗 Deployment on Hugging Face Spaces
This app is ready for deployment on Hugging Face Spaces:

Create a new Space (Gradio type)

Upload:

app.py
requirements.txt
Spaces automatically builds and launches your app!

## API Documentation
There’s no REST API yet. All usage happens via the Gradio web UI.

##  AI Development Process
AI tools used: ChatGPT for code assistance and optimization.

Phases assisted by AI:
Code conversion (ipywidgets → Gradio)

Estimated AI-assisted code %: ~30%

Interesting AI solutions:
Rapid generation of a deployable Gradio interface
Clean handling of semantic search logic

Where AI couldn’t help:
Local environment debugging (manual setup needed)

## Technical Decisions
Why Sentence Transformers? Small but powerful models for fast local inference.
OpenAI vs open-source? Chose open-source (sentence-transformers) for zero cost and privacy.
Local vs cloud LLMs? Local embeddings reduce API costs. All runs locally without cloud APIs.
Performance vs cost vs privacy: The chosen model balances speed and privacy while avoiding cloud costs.

## Future Improvements
Integrate live HR database instead of hard-coded data
Add REST API for programmatic queries
Add user authentication
Deploy as a docker container for wider compatibility
