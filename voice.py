from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import tempfile
import base64
from gtts import gTTS
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ERPNext API credentials
ERP_URL = "http://grofresh.crm-doctor.com:8000"
API_KEY = "d296ff153c0517b"
API_SECRET = "cd5acb77227b874"

# Global variables
df = None
model = None
tokenizer = None
query_count = 0

class VoiceAssistant:
    def __init__(self):
        self.online_available = True
        print("Voice Assistant initialized")
    
    def generate_speech(self, text):
        """Generate speech using gTTS and return file path"""
        try:

            tts = gTTS(text=text, lang='en', slow=False)
            

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()
            
            return temp_file.name
                
        except Exception as e:
            print(f"TTS error: {e}")
            return None


voice_assistant = VoiceAssistant()

def fetch_leads():
    """Fetch lead data from ERPNext"""
    try:
        endpoint = f"{ERP_URL}/api/resource/Lead"
        params = {
            "fields": '["lead_name", "email", "custom_interested_courses_", "status"]'
        }
        response = requests.get(endpoint, params=params, auth=HTTPBasicAuth(API_KEY, API_SECRET))
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df.rename(columns={"lead_name":"student","email":"email","custom_interested_courses_":"course"}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching leads: {e}")
        return pd.DataFrame()

def fetch_courses():
    """Fetch course data from ERPNext"""
    try:
        endpoint = f"{ERP_URL}/api/resource/Course"
        params = {
            "fields": '["course_name", "course_fee"]'
        }
        response = requests.get(endpoint, params=params, auth=HTTPBasicAuth(API_KEY, API_SECRET))
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df.rename(columns={"course_name":"course","course_fee":"fee"}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching courses: {e}")
        return pd.DataFrame()

def load_latest_data():
    """Fetch and merge the latest data from ERPNext"""
    try:
        print("Fetching data from ERP...")
        leads_df = fetch_leads()
        courses_df = fetch_courses()
        
        if leads_df.empty or courses_df.empty:
            print("No data received from ERP")
            return None

        df = leads_df.merge(courses_df, on="course", how="left")
        df['fee'] = df['fee'].fillna('0').astype(str).str.replace(',', '').astype(float).astype(int)
        
        print(f"Loaded {len(df)} records from ERP")
        return df
    except Exception as e:
        print(f"Data loading error: {e}")
        return None

def answer_question_strict(question, current_df):
    """Generate answer for the given question using ERP data"""
    global query_count, df
    
    query_count += 1
    if query_count % 10 == 0:
        print("Auto-refreshing ERP data...")
        fresh_data = load_latest_data()
        if fresh_data is not None and not fresh_data.empty:
            df = fresh_data
            current_df = df
            print("ERP data auto-refreshed!")
        else:
            print("Auto-refresh failed - using cached ERP data")

    if current_df.empty:
        return "Sorry, no ERP data is available. Please check your connection and try refreshing the data."

    # Precompute statistics
    enquiries = current_df['course'].value_counts(dropna=True).to_dict()
    revenue = {}
    for c, g in current_df.groupby('course'):
        if not g.empty and len(g) > 0:
            course_fee = g['fee'].iloc[0] if 'fee' in g.columns else 0
            paid_students = len(g[g['status'].str.lower().str.contains('paid fees', na=False)])
            revenue[c] = int(course_fee * paid_students)

    # More precise status filtering
    total_paid = len(current_df[current_df['status'].str.lower().str.contains('paid fees', na=False)])
    total_pending = len(current_df[current_df['status'].str.lower().str.contains('not paid', na=False)])

    # Rule-based responses
    question_lower = question.lower()

    # Course enquiry questions
    for course in enquiries.keys():
        if course.lower() in question_lower and ("enquir" in question_lower or "inquiry" in question_lower):
            count = enquiries[course]
            return f"The number of enquiries for the {course} course is {count}."

    # Revenue questions
    if "highest revenue" in question_lower or "most revenue" in question_lower:
        if revenue:
            top_course = max(revenue, key=revenue.get)
            top_amount = revenue[top_course]
            return f"The course with the highest revenue is {top_course} with revenue of {top_amount} rupees."

    # Enquiry questions
    if ("highest enquir" in question_lower or "most enquir" in question_lower):
        if enquiries:
            top_course = max(enquiries, key=enquiries.get)
            count = enquiries[top_course]
            return f"The course with the highest number of enquiries is {top_course} with {count} enquiries."

    # Payment status questions
    if ("paid" in question_lower and "not" in question_lower) or ("paid" in question_lower and "pending" in question_lower) or "payment status" in question_lower:
        paid_count = len(current_df[current_df['status'].str.lower().str.contains('paid fees', na=False)])
        pending_count = len(current_df[current_df['status'].str.lower().str.contains('not paid', na=False)])
        
        return f"{paid_count} students have paid fees and {pending_count} students have not paid fees."

    # AI-generated response for other questions
    try:
        prompt = f"""Answer this question about student enrollment data clearly and concisely:

Data Summary:
- Course enquiries: {dict(list(enquiries.items())[:3])}
- Total paid students: {total_paid}
- Total pending students: {total_pending}
- Available courses: {list(enquiries.keys())[:5]}

Question: {question}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer:")[-1].strip()
        answer = answer.split('\n')[0].strip()
        answer = re.sub(r'\s+', ' ', answer)

        if len(answer) < 5:
            answer = "I need more specific information to answer that question properly."

        return answer
    except Exception as e:
        return f"I encountered an issue processing your question. Please try rephrasing it."

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main API endpoint to ask questions and get voice responses"""
    global df
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question parameter',
                'message': 'Please provide a question in JSON format: {"question": "your question"}'
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                'error': 'Empty question',
                'message': 'Please provide a non-empty question'
            }), 400
        
        # Check if we have ERP data, try to load if not available
        if df is None or df.empty:
            print("No data available, attempting to load ERP data...")
            df = load_latest_data()
            
            if df is None or df.empty:
                return jsonify({
                    'error': 'No ERP data available',
                    'message': 'Please check ERP connection and credentials',
                    'text_response': 'No ERP data available. Please check your connection.',
                    'audio_base64': None
                }), 503
        
        # Generate text response
        print(f"Processing question: {question}")
        text_response = answer_question_strict(question, df)
        print(f"Generated response: {text_response}")
        
        # Generate voice response
        audio_file_path = voice_assistant.generate_speech(text_response)
        
        response_data = {
            'question': question,
            'text_response': text_response,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # If audio generation successful, include audio data
        if audio_file_path:
            try:
                # Read the audio file and encode as base64
                with open(audio_file_path, 'rb') as audio_file:
                    audio_content = audio_file.read()
                    audio_base64 = base64.b64encode(audio_content).decode('utf-8')
                
                response_data['audio_base64'] = audio_base64
                response_data['audio_format'] = 'mp3'
                
                # Clean up temporary file
                os.unlink(audio_file_path)
                
            except Exception as e:
                print(f"Error processing audio file: {e}")
                response_data['audio_base64'] = None
                response_data['audio_error'] = str(e)
        else:
            response_data['audio_base64'] = None
            response_data['audio_error'] = 'Failed to generate audio'
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'failed'
        }), 500

def initialize_app():
    """Initialize the application with model and data loading"""
    global df, model, tokenizer
    
    print("Initializing ERP Voice Assistant API...")
    

    try:
        print("🤖 Loading AI model...")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("AI model loaded successfully")
    except Exception as e:
        print(f"Failed to initialize AI model: {e}")
        return False
    
    # Load initial ERP data
    print("Loading initial ERP data...")
    df = load_latest_data()
    
    if df is None or df.empty:
        print("Warning: No ERP data loaded initially. Will try to load on first request.")
    else:
        print(f"Initial data loaded: {len(df)} records")
    
    print("API initialization complete!")
    return True

if __name__ == '__main__':
    if initialize_app():
        print("\nStarting Flask server...")
        print("API Endpoint: POST /ask")
        print("\nExample usage with curl:")
        print('   curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d \'{"question": "Which course has the highest revenue?"}\'')
        print("\n" + "="*60)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("Failed to initialize application. Exiting...")