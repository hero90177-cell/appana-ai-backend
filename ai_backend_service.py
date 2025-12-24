# ai_backend_service.py
# Appana AI – Final Unified Server (vFinal)
# Integrates OCR, AI Chat, Memory, and Advanced Prompts
# Run with: python ai_backend_service.py

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
import pdfplumber
from PIL import Image, ImageOps
import io
import uvicorn
import os
import httpx
import json

app = FastAPI()

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Set your keys here or in system environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# Simple In-Memory Storage for Chat History (Replaces Cloudflare KV)
# Note: This clears when you restart the server.
CHAT_MEMORY = {} 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_LANGS = "eng+hin"

# ---------------------------------------------------------
# 2. OCR HELPERS
# ---------------------------------------------------------

def process_image_for_ocr(image):
    # Grayscale & Upscale for better accuracy
    img = image.convert("L")
    img = img.resize((int(img.width * 2), int(img.height * 2)), Image.Resampling.LANCZOS)
    return img

# ---------------------------------------------------------
# 3. OCR ENDPOINTS
# ---------------------------------------------------------

@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))
        processed = process_image_for_ocr(image)
        text = pytesseract.image_to_string(processed, lang=OCR_LANGS)
        
        if not text.strip():
            return {"status": "warning", "text": "(No text detected. Try a clearer image.)"}
            
        return {"status": "success", "text": text}
    except Exception as e:
        print(f"OCR Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        full_text = ""
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try text extraction first
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 5:
                    full_text += page_text + "\n"
                else:
                    # Fallback to OCR
                    try:
                        im = page.to_image(resolution=300).original
                        processed = process_image_for_ocr(im)
                        ocr_text = pytesseract.image_to_string(processed, lang=OCR_LANGS)
                        full_text += ocr_text + "\n"
                    except Exception as ocr_err:
                        full_text += f"[Page {i+1}: Image Scan Failed]\n"

        return {"status": "success", "text": full_text if full_text.strip() else "(No text found)"}

    except Exception as e:
        print(f"PDF Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------------------------------------
# 4. AI CHAT ENDPOINT (Restored Logic from JS)
# ---------------------------------------------------------

@app.post("/api/ai-chat")
async def ai_chat_endpoint(request: Request):
    try:
        body = await request.json()
        
        # 1. Ping Check
        if body.get("type") == "ping":
            return {"status": "ok", "backend": "python-unified"}

        # 2. Extract Data
        message = body.get("message", "")
        uid = body.get("uid", "guest")
        subject = body.get("subject", "General")
        language = body.get("language", "English")
        examMode = body.get("examMode", "normal")
        goal = body.get("goal", "")

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        # 3. Retrieve Memory
        previous_context = CHAT_MEMORY.get(uid, "")

        # 4. Build Professional System Prompt
        tone = "friendly, encouraging, exam-focused mentor"
        format_req = "clear and concise bullet points"

        if examMode == "teacher": 
            tone = "strict, formal, precise Indian syllabus teacher"
        elif examMode == "2marks": 
            format_req = "2–3 sentences, exam-oriented, precise"
        elif examMode == "5marks": 
            format_req = "structured paragraph with 5 key points"
        elif examMode == "8marks": 
            format_req = "detailed essay with introduction, body, and conclusion"

        system_prompt = f"""
You are Appana AI.
Role: {tone}
Subject: {subject}
Language: {language}
Exam Mode: {examMode}
Goal: {goal}
Format Requirement: {format_req}

Context History:
{previous_context}

Instructions:
1. Use provided Context/Large Subjects automatically.
2. Be Indian syllabus aware (CBSE/ICSE/State Boards).
3. Keep explanations clear, accurate, and exam-relevant.
4. Use emojis sparingly (max one per response).
5. Analyze any provided file text first.
"""
        full_prompt = f"{system_prompt}\n\nStudent: {message}"

        # 5. Inject Large Subjects
        if "largeSubjects" in body and isinstance(body["largeSubjects"], list):
            extra = "\n\n".join([f"[{s['name']}]\n{s['content']}" for s in body["largeSubjects"]])
            full_prompt += f"\n\n[LARGE SUBJECT CONTEXT]:\n{extra}"

        reply = None

        # 6. Call AI Providers (Gemini -> Groq -> Cohere -> HF)
        async with httpx.AsyncClient(timeout=40.0) as client:
            
            # --- GEMINI ---
            if not reply and GEMINI_API_KEY:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
                    r = await client.post(url, json=payload)
                    data = r.json()
                    reply = data['candidates'][0]['content']['parts'][0]['text']
                except Exception as e:
                    print(f"Gemini Error: {e}")

            # --- GROQ ---
            if not reply and GROQ_API_KEY:
                try:
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": full_prompt}]
                    }
                    r = await client.post(url, json=payload, headers=headers)
                    data = r.json()
                    reply = data['choices'][0]['message']['content']
                except Exception as e:
                    print(f"Groq Error: {e}")

            # --- COHERE ---
            if not reply and COHERE_API_KEY:
                try:
                    url = "https://api.cohere.ai/v1/chat"
                    headers = {
                        "Authorization": f"Bearer {COHERE_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {"model": "command-r-plus", "message": full_prompt}
                    r = await client.post(url, json=payload, headers=headers)
                    data = r.json()
                    reply = data['text']
                except Exception as e:
                    print(f"Cohere Error: {e}")

        if not reply:
            return {"reply": "⚠️ All AI providers failed. Check your API Keys."}

        # 7. Post-Processing & Memory Update
        if examMode not in ["teacher", "2marks", "5marks", "8marks"]:
            lower = reply.lower()
            emoji = ""
            if "important" in lower: emoji = "📌"
            elif "remember" in lower: emoji = "💡"
            elif "excellent" in lower: emoji = "✅"
            if emoji: reply = f"{emoji} {reply}"

        # Save to Memory (Keep last 2000 chars)
        new_mem = f"{previous_context}\nQ: {message}\nA: {reply}"
        if len(new_mem) > 2000: new_mem = new_mem[-2000:]
        CHAT_MEMORY[uid] = new_mem

        return {"reply": reply}

    except Exception as e:
        print(f"Chat Error: {e}")
        return JSONResponse(status_code=500, content={"reply": f"🔥 Server Error: {str(e)}"})

# ---------------------------------------------------------
# 5. STATIC FILES (MUST BE LAST)
# ---------------------------------------------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
