"""
FastAPI Backend — AirBnB Support Chatbot
Streaming chat endpoint powered by Mistral via Ollama + RAG context.
"""

import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import ollama

from rag import get_context

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="AirBnB Support Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful and friendly AirBnB customer support assistant.

RULES:
1. Answer ONLY using the provided context below. Do not make up information.
2. If the context doesn't contain enough information, say "I don't have specific information about that, but I'd recommend contacting Airbnb support directly for the most accurate answer."
3. Be concise, friendly, and professional.
4. Use bullet points for multi-part answers.
5. If the user greets you, respond warmly and ask how you can help with their AirBnB experience.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

# ---------------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: Request):
    """
    Streaming chat endpoint.
    Accepts: { "message": "user question" }
    Returns: Server-Sent Events stream of response chunks.
    """
    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return {"error": "Message is required"}

    # Retrieve relevant context from RAG
    context = get_context(user_message)

    # Build the prompt with RAG context
    system_msg = SYSTEM_PROMPT.format(context=context)

    def generate():
        """Stream response tokens from Mistral via Ollama."""
        stream = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_message},
            ],
            stream=True,
            options={
                "num_predict": 512,      # Max tokens
                "temperature": 0.3,      # Low temp for factual responses
                "top_p": 0.9,
            },
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            # Send as SSE format
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": "mistral"}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
