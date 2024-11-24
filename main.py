import streamlit as st
import yt_dlp
import whisper
import os
import json
import csv
import pandas as pd
from typing import List, Dict

class TranscriptionDataConverter:
    def __init__(self, transcription: str, video_title: str):
        """
        Initialize the converter with transcription text
        """
        self.transcription = transcription
        self.video_title = video_title

    def generate_qa_pairs(self, num_questions: int = 5) -> List[Dict]:
        """
        Generate question-answer pairs from the transcription
        """
        import spacy
        import random

        try:
            # Load English tokenizer, tagger, parser, NER and word vectors
            nlp = spacy.load("en_core_web_sm")
        except:
            st.warning("SpaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        # Process the transcription
        doc = nlp(self.transcription)

        # Extract potential questions
        sentences = list(doc.sents)
        qa_pairs = []

        # Try to generate diverse questions
        question_types = [
            "What",
            "How",
            "Why",
            "When",
            "Where"
        ]

        for _ in range(min(num_questions, len(sentences))):
            # Select a random sentence
            sentence = random.choice(sentences)
            
            # Generate a question
            question_type = random.choice(question_types)
            question = f"{question_type} {sentence.text.split()[1:4]}?".replace('[', '').replace(']', '')
            
            qa_pairs.append({
                "context": self.transcription,
                "question": question.strip(),
                "answer": sentence.text
            })

        return qa_pairs

    def to_json(self, format_type: str = "raw") -> str:
        """
        Convert transcription to different JSON formats
        """
        if format_type == "raw":
            return json.dumps({
                "video_title": self.video_title,
                "transcription": self.transcription
            }, indent=2)
        
        elif format_type == "qa":
            qa_pairs = self.generate_qa_pairs()
            return json.dumps(qa_pairs, indent=2)
        
        elif format_type == "instruction":
            return json.dumps({
                "instruction": f"Provide insights about the video titled: {self.video_title}",
                "input": self.transcription,
                "output": "Detailed analysis and key points"
            }, indent=2)

    def to_csv(self) -> str:
        """
        Convert transcription to CSV
        """
        import io
        
        # Create a CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Video Title", "Transcription"])
        writer.writerow([self.video_title, self.transcription])
        
        return output.getvalue()

    def to_parquet(self) -> bytes:
        """
        Convert transcription to Parquet format
        """
        df = pd.DataFrame({
            "video_title": [self.video_title],
            "transcription": [self.transcription]
        })
        
        # Save to bytes
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()

# Function: Download YouTube audio and transcribe
def process_youtube_video(video_url):
    """
    Download and transcribe YouTube video
    """
    # Create temporary directory
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_filename = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
            video_title = info_dict.get('title', 'Unknown Title')
        
        # Transcribe
        model = whisper.load_model("base")
        result = model.transcribe(audio_filename)
        
        return video_title, result["text"], audio_filename
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None, None

# Streamlit App
def main():
    st.set_page_config(page_title="YouTube to LLM Training Data", page_icon="🤖")
    
    st.title("YouTube Transcription to LLM Training Data Converter")
    st.write("Convert YouTube video transcriptions into machine learning training formats")
    
    # YouTube URL input
    video_url = st.text_input("YouTube Video URL", placeholder="Enter YouTube URL")
    
    # Conversion format selection
    conversion_formats = {
        "JSON (Raw)": "raw",
        "JSON (QA Pairs)": "qa",
        "JSON (Instruction)": "instruction",
        "CSV": "csv",
        "Parquet": "parquet"
    }
    selected_format = st.selectbox("Select Conversion Format", list(conversion_formats.keys()))
    
    if st.button("Convert"):
        if not video_url:
            st.error("Please enter a YouTube URL")
            return
        
        with st.spinner("Processing video..."):
            video_title, transcription, audio_filename = process_youtube_video(video_url)
        
        if not transcription:
            st.error("Failed to process the video")
            return
        
        # Create converter
        converter = TranscriptionDataConverter(transcription, video_title)
        
        
        format_key = conversion_formats[selected_format]
        
        if format_key in ["raw", "qa", "instruction"]:
            converted_data = converter.to_json(format_key)
            file_extension = "json"
            mime_type = "application/json"
        elif format_key == "csv":
            converted_data = converter.to_csv()
            file_extension = "csv"
            mime_type = "text/csv"
        elif format_key == "parquet":
            converted_data = converter.to_parquet()
            file_extension = "parquet"
            mime_type = "application/parquet"
        
        
        st.subheader("Converted Data Preview")
        st.code(converted_data[:1000] + "..." if len(converted_data) > 1000 else converted_data)
        

        st.download_button(
            label=f"Download {selected_format}",
            data=converted_data,
            file_name=f"training_data.{file_extension}",
            mime=mime_type
        )
        
        # Cleanup
        try:
            os.remove(audio_filename)
        except:
            pass

if __name__ == "__main__":
    main()