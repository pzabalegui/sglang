#!/usr/bin/env python3
"""
Simple FastAPI server for abliterated GLM-4.7-Flash model with proper chat format
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading abliterated model...")
    model_path = "./output_v3_final"
    
    tokenizer = AutoTokenizer.from_pretrained(
        "zai-org/GLM-4.7-Flash",
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!")
    yield

app = FastAPI(title="Abliterated GLM-4.7-Flash", lifespan=lifespan)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Format as GLM chat
    messages = [{"role": "user", "content": request.prompt}]
    
    # Use tokenizer's chat template
    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else 0.01,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return ChatResponse(response=response)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
