from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from havoc_core.config import InferenceConfig
from havoc_inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None


# ============================================================================
# Request/Response Models
# ============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    prompt: str = Field(..., description="Input prompt text")
    max_new_tokens: Optional[int] = Field(512, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(50, description="Top-k sampling threshold")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop_sequences: Optional[list[str]] = Field(None, description="Sequences that stop generation")


class CompletionResponse(BaseModel):
    """Response for text completion."""

    text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    model: str = Field("havoc-7b", description="Model name")
    usage: dict = Field(..., description="Token usage statistics")


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str = Field(..., description="Message role (system/user/assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""

    messages: list[ChatMessage] = Field(..., description="List of chat messages")
    max_new_tokens: Optional[int] = Field(512, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(50, description="Top-k sampling threshold")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop_sequences: Optional[list[str]] = Field(None, description="Sequences that stop generation")


class ChatResponse(BaseModel):
    """Response for chat completion."""

    message: ChatMessage = Field(..., description="Generated message")
    model: str = Field("havoc-7b", description="Model name")
    usage: dict = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    device: str = Field(..., description="Device (cuda/cpu)")
    timestamp: float = Field(..., description="Current timestamp")


# ============================================================================
# FastAPI App
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    global inference_engine
    config = app.state.config
    logger.info("Starting inference engine...")
    inference_engine = InferenceEngine(config)
    logger.info("Inference engine ready!")

    yield

    # Shutdown
    logger.info("Shutting down inference engine...")
    inference_engine = None


def create_app(config: InferenceConfig) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="HAVOC-7B Inference API",
        description="REST API for HAVOC-7B language model inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint."""
        return {
            "message": "HAVOC-7B Inference API",
            "version": "0.1.0",
            "endpoints": {
                "completion": "/completion",
                "chat": "/chat",
                "health": "/health",
            },
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Liveness endpoint."""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not initialized")

        return HealthResponse(
            status="ok",
            model="havoc-7b",
            device=str(inference_engine.device),
            timestamp=time.time(),
        )

    @app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
    async def ready():
        """Readiness endpoint."""
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not initialized")

        return HealthResponse(
            status="ready",
            model="havoc-7b",
            device=str(inference_engine.device),
            timestamp=time.time(),
        )

    @app.post("/completion", response_model=CompletionResponse, tags=["Completion"])
    async def completion(request: CompletionRequest):
        """
        Text completion endpoint.

        Generate text from a prompt using the HAVOC-7B model.
        """
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not initialized")

        try:
            if request.stream:
                def token_iterator():
                    for token in inference_engine.generate_stream(
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                        do_sample=request.do_sample,
                        stop_sequences=request.stop_sequences,
                    ):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    iterate_in_threadpool(token_iterator()),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Non-streaming response
                generated_text = inference_engine.generate(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    stop_sequences=request.stop_sequences,
                )

                return CompletionResponse(
                    text=generated_text,
                    prompt=request.prompt,
                    model="havoc-7b",
                    usage={
                        "prompt_tokens": len(request.prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(request.prompt.split()) + len(generated_text.split()),
                    },
                )
        except Exception as e:
            logger.error(f"Error in completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate", response_model=CompletionResponse, tags=["Completion"])
    async def generate_alias(request: CompletionRequest):
        """Alias for /completion to satisfy tooling that expects /generate."""
        return await completion(request)

    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """
        Chat completion endpoint.

        Generate a chat response from a list of messages.
        """
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not initialized")

        try:
            # Format messages into a prompt
            prompt = format_chat_prompt(request.messages)

            if request.stream:
                def token_iterator():
                    for token in inference_engine.generate_stream(
                        prompt=prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                        do_sample=request.do_sample,
                        stop_sequences=request.stop_sequences,
                    ):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    iterate_in_threadpool(token_iterator()),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Non-streaming response
                generated_text = inference_engine.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    stop_sequences=request.stop_sequences,
                )

                return ChatResponse(
                    message=ChatMessage(role="assistant", content=generated_text),
                    model="havoc-7b",
                    usage={
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(prompt.split()) + len(generated_text.split()),
                    },
                )
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def format_chat_prompt(messages: list[ChatMessage]) -> str:
    """
    Format chat messages into a single prompt.

    Uses a simple template that can be customized for your use case.
    """
    prompt_parts = []

    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}\n")
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}\n")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}\n")

    # Add assistant prefix for next response
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)
