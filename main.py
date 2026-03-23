import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from agentic_workflow import run_agentic_workflow
from classifier import classify_query
from data_retriever import assign_mentor, get_user_context
from rbac import detect_role
from response_formatter import (
    create_excel,
    create_pdf,
    create_text_file,
    create_word,
    format_text_response,
)


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodle-ai-assistant")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
MENTOR_ASSIGNMENTS_PATH = BASE_DIR / "data" / "mentor_assignments.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Moodle AI Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    format: Literal["text", "txt", "pdf", "excel", "word"] = "text"


class MentorAssignmentRequest(BaseModel):
    actor_user_id: str = Field(..., min_length=1)
    student_id: str = Field(..., min_length=1)
    mentor_name: str = Field(..., min_length=1)
    mentor_email: str = ""
    mentor_phone: str = ""


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or still using the placeholder value in .env")
    return Groq(api_key=api_key)


def _chat_sync(system_prompt: str, user_prompt: str, model: str) -> str:
    client = _get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


async def ask_groq(system_prompt: str, user_prompt: str, model: str) -> str:
    return await asyncio.to_thread(_chat_sync, system_prompt, user_prompt, model)


def _cleanup_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _build_download_response(file_path: Path, media_type: str, filename: str) -> FileResponse:
    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(_cleanup_temp_file, file_path),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/user-context/{user_id}")
async def user_context(user_id: str):
    role = detect_role(user_id)
    context = get_user_context(
        data_path=str(DATA_PATH),
        user_id=user_id,
        role=role,
        assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
    )
    return JSONResponse(context)


@app.get("/")
async def home():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.post("/mentor/assign")
async def mentor_assignment(payload: MentorAssignmentRequest):
    try:
        actor_role = detect_role(payload.actor_user_id)
        result = assign_mentor(
            data_path=str(DATA_PATH),
            assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
            actor_role=actor_role,
            actor_user_id=payload.actor_user_id,
            student_id=payload.student_id,
            mentor_name=payload.mentor_name,
            mentor_email=payload.mentor_email,
            mentor_phone=payload.mentor_phone,
        )
        return JSONResponse({"role": actor_role, **result})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ask")
async def ask(payload: AskRequest):
    try:
        result = await run_agentic_workflow(
            user_id=payload.user_id,
            query=payload.query,
            data_path=str(DATA_PATH),
            assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
            classify_query=classify_query,
            ask_groq=ask_groq,
        )
        logger.info("Agent trace: %s", json.dumps(result.trace))

        if payload.format == "text":
            return JSONResponse(
                {
                    "answer": format_text_response(result.answer),
                    "role": result.role,
                    "classification": result.classification,
                    "user_context": result.user_context,
                    "agent_trace": result.trace,
                }
            )

        if payload.format == "txt":
            file_path = create_text_file(result.answer)
            return _build_download_response(
                file_path=file_path,
                media_type="text/plain; charset=utf-8",
                filename="moodle_ai_response.txt",
            )

        if payload.format == "pdf":
            file_path = create_pdf(result.answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/pdf",
                filename="moodle_ai_response.pdf",
            )

        if payload.format == "excel":
            file_path = create_excel(result.answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="moodle_ai_response.xlsx",
            )

        if payload.format == "word":
            file_path = create_word(result.answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename="moodle_ai_response.docx",
            )

        raise HTTPException(status_code=400, detail="Invalid format")
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to process /ask")
        raise HTTPException(status_code=500, detail=f"Server error: {exc}")
