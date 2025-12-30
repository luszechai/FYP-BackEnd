# SFU Admission Chatbot - Backend

A RAG (Retrieval-Augmented Generation) based chatbot for Saint Francis University admission inquiries.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_actual_api_key_here
```

**Important:** Never commit your `.env` file to Git! It's already in `.gitignore`.

### 3. Prepare Data File

Place your `merged_rag_data.json` file in the project root directory.

The JSON file should have the following structure:

```json
{
  "documents": [
    {
      "content": "Your document text content here...",
      "section": "Admission Requirements",
      "metadata": {
        "source": "admission_guide.pdf",
        "structured": false
      }
    }
  ]
}
```

### 4. Run the Application

```bash
python main.py
```

## Features

- **RAG-based Architecture**: Retrieves relevant documents before generating responses
- **Adaptive Configuration**: Automatically adjusts parameters based on query complexity
- **Query Enhancement**: Expands queries for better retrieval (person names, program codes, etc.)
- **Conversation Memory**: Maintains context across multiple exchanges
- **Performance Tracking**: Tracks response times and retrieval metrics
- **Date/Time Awareness**: Provides current date/time context for deadline queries

## Project Structure

```
FYP-BackEnd/
├── main.py                 # Entry point
├── config.py               # Configuration (reads from .env)
├── .env                    # Environment variables (NOT in git)
├── .env.example            # Template for .env
├── requirements.txt        # Python dependencies
├── merged_rag_data.json    # Your data file
└── src/
    ├── chatbot.py          # Main chatbot class
    ├── query_enhancer.py   # Query enhancement logic
    ├── retrieval.py        # Hybrid retrieval strategies
    ├── prompts.py          # Prompt templates
    ├── adaptive_config.py  # Adaptive parameter adjustment
    ├── utils.py            # Utility functions
    ├── memory.py           # Conversation memory
    ├── llm_provider.py     # LLM API interactions
    ├── vector_db.py        # ChromaDB operations
    └── evaluation.py       # Evaluation dashboard
```

## Configuration

Most settings can be adjusted in `config.py`. Key settings:

- `USE_ADAPTIVE_CONFIG`: Enable/disable automatic parameter adjustment (default: True)
- `RETRIEVAL_K`: Base number of documents to retrieve (default: 5)
- `LLM_MAX_TOKENS`: Maximum response length (default: 1000)
- `CHUNK_SIZE`: Document chunk size for vector DB (default: 1600)

## Security Notes

- **Never commit API keys or `.env` files to Git**
- The `.env` file is already in `.gitignore`
- Use environment variables for all sensitive data
- Rotate API keys if accidentally exposed

## Getting Your DeepSeek API Key

1. Visit https://platform.deepseek.com/
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Copy it to your `.env` file

## Troubleshooting

- **"DEEPSEEK_API_KEY is not set"**: Make sure you've created a `.env` file with your API key
- **"merged_rag_data.json not found"**: Place your data file in the project root
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

