# YouTube to LLM Training Data Converter

A Streamlit application that converts YouTube videos into high-quality training data formats for Language Learning Models (LLMs). The tool transcribes YouTube videos and processes them into various structured formats suitable for LLM training.

## Features

- Automatic YouTube video download and audio extraction
- Speech-to-text transcription using Whisper
- LLM enhancement using Groq's LLaMA 3.1 70B model
- Multiple output formats:
  - Enhanced JSON with structured content analysis
  - Question-Answer pairs for training
  - Instruction-following format
  - CSV training data with segmented content

## Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- Groq API key
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd youtube-llm-converter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

3. Install the required packages:
```bash
pip install streamlit yt-dlp whisper langchain-groq pandas python-dotenv
```

4. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Enter a YouTube URL and select your desired output format:
   - Enhanced JSON: Provides structured content with topics and insights
   - QA Pairs: Generates question-answer pairs suitable for training
   - Instruction Format: Creates instruction-following training examples
   - CSV Training Data: Produces segmented data in industry-standard format

4. Click "Convert" and wait for the processing to complete

5. Download the generated training data in your chosen format

## Output Formats

### Enhanced JSON
```json
{
  "cleaned_transcription": "...",
  "topics": [...],
  "key_insights": [...],
  "technical_terms": [...]
}
```

### QA Pairs
```json
[
  {
    "question": "...",
    "answer": "...",
    "context": "...",
    "type": "..."
  }
]
```

### Instruction Format
```json
[
  {
    "instruction": "...",
    "input": "...",
    "output": "...",
    "type": "..."
  }
]
```

### CSV Training Data
Includes columns for:
- segment_id
- timestamp
- content
- topic
- tags
- category
- difficulty
- speaker
- keywords
- summary
- source_title
- data_type

## Technical Details

The application uses several key components:

- **yt-dlp**: For downloading YouTube videos and extracting audio
- **Whisper**: OpenAI's speech recognition model for transcription
- **LangChain + Groq**: For enhanced content processing using LLaMA 3.1
- **Streamlit**: For the web interface
- **Pandas**: For data manipulation and CSV generation

## Error Handling

The application includes comprehensive error handling for:
- Invalid YouTube URLs
- Failed video downloads
- Transcription errors
- LLM processing failures
- File system operations

## Limitations

- Processing time depends on video length and chosen output format
- Requires stable internet connection
- API rate limits may apply (Groq)
- Audio file quality affects transcription accuracy

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[Add your chosen license here]

## Acknowledgments

- OpenAI's Whisper for transcription capabilities
- Groq for LLM processing
- Streamlit for the web interface framework
