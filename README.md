# RAG Chatbot with PDF Processing

A Streamlit application that allows users to chat with their PDF documents using AI models (Gemini/Groq).

## Features
- PDF document processing and text extraction
- Chat interface for querying PDF content
- Support for multiple AI models (Gemini/Groq)
- Real-time responses

## Deployment Instructions

### Local Development
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   GORQ_API_KEY=your_groq_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the following in the deployment settings:
   - Main file path: `app.py`
   - Python version: 3.9 or higher
   - Add your environment variables (GOOGLE_API_KEY and GORQ_API_KEY)

## Requirements
- Python 3.9+
- Streamlit
- Unstructured
- LlamaIndex
- LangChain
- Google Generative AI
- Other dependencies listed in requirements.txt

## System Dependencies
- Tesseract OCR
- Poppler Utils

## License
MIT 