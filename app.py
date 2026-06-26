from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from docx import Document
from PyPDF2 import PdfReader

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global model, tokenizer
    if model is None:
        print("Loading model for the first time...")
        # Ensure path matches the folder created by your download_model.sh script
        tokenizer = AutoTokenizer.from_pretrained("./saved_summary_model", use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained("./saved_summary_model")
        model.to(device)
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
MAX_INPUT_CHARS = 6000 

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

# -----------------------------
# Summarization logic
# -----------------------------
def summarize_dialogue(dialogue: str) -> str:
    current_model, current_tokenizer = get_model() # Trigger lazy load
    
    dialogue = clean_text(dialogue)
    dialogue = dialogue[:MAX_INPUT_CHARS]
    dialogue = "summarize: " + dialogue

    inputs = current_tokenizer(
        dialogue,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    input_length = input_ids.shape[1]
    max_summary_len = max(80, int(input_length * 0.4))
    max_summary_len = min(max_summary_len, 300)
    min_summary_len = max(40, int(max_summary_len * 0.5))

    with torch.no_grad():
        outputs = current_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_summary_len,
            min_length=min_summary_len,
            num_beams=4,
            length_penalty=1.1,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return current_tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    return templates.TemplateResponse("index.html", {"request": request})
