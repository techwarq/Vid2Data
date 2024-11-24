import os
import streamlit as st
import yt_dlp
import whisper
import json
import csv
import pandas as pd
import io
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class LLMEnhancedConverter:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.1-70b-versatile"
        )

    def enhance_transcription(self, transcription: str, video_title: str) -> dict:
        prompt = PromptTemplate.from_template(
            """
            ### VIDEO TITLE:
            {title}

            ### TRANSCRIPTION:
            {transcription}

            ### INSTRUCTION:
            Analyze the video transcription and create a structured summary that includes:
            1. A clean, properly formatted version of the transcription
            2. Main topics discussed
            3. Key insights and takeaways
            4. Any technical terms or concepts mentioned

            Return the results in JSON format with these keys: 
            - cleaned_transcription
            - topics
            - key_insights
            - technical_terms

            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        chain = prompt | self.llm
        res = chain.invoke({"title": video_title, "transcription": transcription})
        
        try:
            json_parser = JsonOutputParser()
            return json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse enhanced transcription")

    def generate_qa_pairs(self, transcription: str, video_title: str) -> List[Dict]:
        prompt = PromptTemplate.from_template(
            """
            ### VIDEO TITLE:
            {title}

            ### TRANSCRIPTION:
            {transcription}

            ### INSTRUCTION:
            Generate 10 high-quality question-answer pairs from the video transcription that would be suitable for training language models. For each pair:
            1. Create diverse question types (what, why, how, etc.)
            2. Ensure answers are comprehensive and accurate
            3. Include context when necessary
            4. Focus on important concepts and key points
            5. Avoid trivial or overly simple questions

            Return the results as a JSON array of objects with these keys:
            - question: The full question text
            - answer: A detailed answer
            - context: Relevant portion of transcription
            - type: Question type (conceptual, factual, analytical, etc.)

            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        chain = prompt | self.llm
        res = chain.invoke({"title": video_title, "transcription": transcription})
        
        try:
            json_parser = JsonOutputParser()
            return json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse QA pairs")

    def generate_instruction_format(self, transcription: str, video_title: str) -> List[Dict]:
        prompt = PromptTemplate.from_template(
            """
            ### VIDEO TITLE:
            {title}

            ### TRANSCRIPTION:
            {transcription}

            ### INSTRUCTION:
            Create 5 high-quality instruction-input-output triplets suitable for training language models. For each triplet:
            1. Create a clear, specific instruction
            2. Provide relevant context as input
            3. Generate a detailed, helpful output
            4. Focus on different aspects (summary, analysis, explanation, etc.)
            5. Ensure outputs demonstrate reasoning and depth

            Return results as a JSON array with these keys:
            - instruction: Clear task description
            - input: Relevant context from transcription
            - output: Detailed response
            - type: Task type (summarization, analysis, explanation, etc.)

            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        chain = prompt | self.llm
        res = chain.invoke({"title": video_title, "transcription": transcription})
        
        try:
            json_parser = JsonOutputParser()
            return json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse instruction format")

    def generate_csv_format(self, transcription: str, video_title: str) -> str:
        prompt = PromptTemplate.from_template(
            """
            ### VIDEO TITLE:
            {title}

            ### TRANSCRIPTION:
            {transcription}

            ### INSTRUCTION:
            Process this video transcription into structured training data segments. For each segment:
            1. Break down the content into logical chunks/segments (2-3 minutes each)
            2. Identify the main topic/theme of each segment
            3. Add relevant tags and categories
            4. Include difficulty level (basic, intermediate, advanced)
            5. Generate a clean, well-formatted version of each segment
            6. Create appropriate metadata

            Return a JSON array of objects with these keys:
            - segment_id: Numerical ID
            - timestamp: Approximate timestamp in minutes
            - content: The cleaned segment text
            - topic: Main topic/theme of the segment
            - tags: Array of relevant tags
            - category: General category (e.g., tutorial, explanation, discussion)
            - difficulty: Level of complexity
            - speaker: Speaker identifier if available
            - keywords: Key terms in the segment
            - summary: Brief summary of the segment

            ### VALID JSON (NO PREAMBLE):
            """
        )
        
        chain = prompt | self.llm
        res = chain.invoke({"title": video_title, "transcription": transcription})
        
        try:
            json_parser = JsonOutputParser()
            segments = json_parser.parse(res.content)
            
        
            output = io.StringIO()
            writer = csv.writer(output)
            
            
            headers = [
                "segment_id",
                "timestamp",
                "content",
                "topic",
                "tags",
                "category",
                "difficulty",
                "speaker",
                "keywords",
                "summary",
                "source_title",
                "data_type"
            ]
            writer.writerow(headers)
            
            
            for segment in segments:
                writer.writerow([
                    segment.get("segment_id", ""),
                    segment.get("timestamp", ""),
                    segment.get("content", ""),
                    segment.get("topic", ""),
                    "|".join(segment.get("tags", [])),
                    segment.get("category", ""),
                    segment.get("difficulty", ""),
                    segment.get("speaker", ""),
                    "|".join(segment.get("keywords", [])),
                    segment.get("summary", ""),
                    video_title,
                    "video_transcript"
                ])
            
            return output.getvalue()
            
        except OutputParserException:
            raise OutputParserException("Unable to parse CSV format")

def process_youtube_video(video_url):
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
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
        
        model = whisper.load_model("base")
        result = model.transcribe(audio_filename)
        
        return video_title, result["text"], audio_filename
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="Enhanced YouTube to LLM Training Data",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("YouTube to LLM Training Data Converter")
    st.write("Convert YouTube videos into high-quality LLM training data formats")
   
    with st.sidebar:
        st.header("About")
        st.write("""
        This tool converts YouTube videos into various formats suitable for training language models:
        - Enhanced JSON: Structured content with topics and insights
        - QA Pairs: Question-answer pairs for training
        - Instruction Format: Instruction-following format
        - CSV Training Data: Industry-standard segmented format
        """)
        
        st.header("Instructions")
        st.write("""
        1. Paste a YouTube URL
        2. Select your desired output format
        3. Click Convert to process the video
        4. Download the generated training data
        """)
    
    
    video_url = st.text_input("YouTube Video URL", placeholder="Enter YouTube URL")
    
   
    col1, col2 = st.columns([2, 1])
    
    with col1:
        conversion_formats = {
            "Enhanced JSON": "enhanced",
            "QA Pairs": "qa",
            "Instruction Format": "instruction",
            "CSV Training Data": "csv"
        }
        selected_format = st.selectbox(
            "Select Training Data Format",
            list(conversion_formats.keys())
        )
    
    if st.button("Convert", type="primary"):
        if not video_url:
            st.error("Please enter a YouTube URL")
            return
        
        
        progress_container = st.empty()
        
        with progress_container.container():
            st.write("Step 1: Downloading and transcribing video...")
            video_title, transcription, audio_filename = process_youtube_video(video_url)
            
        if not transcription:
            st.error("Failed to process the video")
            return
        
        converter = LLMEnhancedConverter()
        
        try:
            with progress_container.container():
                st.write("Step 2: Enhancing content with LLaMA...")
                format_key = conversion_formats[selected_format]
                
                if format_key == "csv":
                    converted_data = converter.generate_csv_format(transcription, video_title)
                    mime_type = "text/csv"
                    file_extension = "csv"
                    
                   
                    st.subheader("Preview (First 1000 characters)")
                    st.text(converted_data[:1000] + "..." if len(converted_data) > 1000 else converted_data)
                    
                  
                    st.subheader("Sample Data Structure")
                    df = pd.read_csv(io.StringIO(converted_data))
                    st.dataframe(df.head(1))
                else:
                    if format_key == "enhanced":
                        converted_data = converter.enhance_transcription(transcription, video_title)
                    elif format_key == "qa":
                        converted_data = converter.generate_qa_pairs(transcription, video_title)
                    else:
                        converted_data = converter.generate_instruction_format(transcription, video_title)
                    
                   
                    converted_data = json.dumps(converted_data, indent=2)
                    mime_type = "application/json"
                    file_extension = "json"
                    
                   
                    st.subheader("Preview")
                    st.json(json.loads(converted_data))
                
              
                st.success("Processing completed!")
                st.download_button(
                    label=f"Download {selected_format}",
                    data=converted_data,
                    file_name=f"llm_training_data_{format_key}.{file_extension}",
                    mime=mime_type
                )
        
        except Exception as e:
            st.error(f"Error enhancing content: {e}")
        
        finally:
            
            try:
                if audio_filename and os.path.exists(audio_filename):
                    os.remove(audio_filename)
            except Exception as e:
                st.warning(f"Warning: Could not clean up temporary files: {e}")

if __name__ == "__main__":
    main()