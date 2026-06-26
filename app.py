from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import gc  # Import garbage collector for manual cleanup
from docx import Document
from PyPDF2 import PdfReader
import uvicorn
import os

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Text Summarization System",
    description="Summarize dialogues with T5!",
    version="1.0"
)

# -----------------------------
# Lazy Model Loading
# -----------------------------
model = None
tokenizer = None
# Force CPU device configuration entirely to save framework footprint overheads
device = torch.device("cpu")

def get_model():
    global model, tokenizer
    if model is None:
        print("Loading model from Hugging Face Hub...")
        model_id = "BASANT1896/saved_summary_model"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        
        # OPTIMIZATION: Load tensors using 16-bit brain floating points (bfloat16) to cut footprint in half
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        model.to("cpu")
        model.eval()
    return model, tokenizer

# -----------------------------
# Templates
# -----------------------------
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Input schema
# -----------------------------
class DialogueInput(BaseModel):
    dialogue: str

# -----------------------------
# Utilities
# -----------------------------
# OPTIMIZATION: Reduce text cap from 6000 down to 2500 characters to prevent huge processing memory peaks
MAX_INPUT_CHARS = 2500 

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

# -----------------------------
# Summarization logic
# -----------------------------
def summarize_dialogue(dialogue: str) -> str:
    # Free stray memory slots right before generation overhead starts
    gc.collect()
    
    current_model, current_tokenizer = get_model() # Trigger lazy load
    
    dialogue = clean_text(dialogue)
    dialogue = dialogue[:MAX_INPUT_CHARS]
    dialogue = "summarize: " + dialogue

    inputs = current_tokenizer(
        dialogue,
        return_tensors="pt",
        truncation=True,
        max_length=512  # OPTIMIZATION: Reduced internal tokenization context max from 1024 to 512
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    input_length = input_ids.shape[1]
    
    # OPTIMIZATION: Cap summary ranges tightly to restrict token tracking matrix memory usage
    max_summary_len = max(60, int(input_length * 0.3))
    max_summary_len = min(max_summary_len, 140)
    min_summary_len = max(30, int(max_summary_len * 0.5))

    with torch.no_grad():
        outputs = current_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_summary_len,
            min_length=min_summary_len,
            num_beams=2,  # OPTIMIZATION: Dropped search width from 4 down to 2 to minimize memory tracking tracks
            length_penalty=1.1,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary_text = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Complete aggressive sweep instantly after text generation finishes
    gc.collect()
    
    return summary_text

# -----------------------------
# API Routes
# -----------------------------
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    text = ""
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + " "
    elif file.filename.lower().endswith(".docx"):
        doc = Document(file.file)
        for para in doc.paragraphs: text += para.text + " "
    elif file.filename.lower().endswith(".txt"):
        text = (await file.read()).decode("utf-8", errors="ignore")
    
    text = clean_text(text)[:MAX_INPUT_CHARS]
    return {"text": text}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Fixed compatibility route signature structure for Python 3.14 + FastAPI
    return templates.TemplateResponse(request, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
