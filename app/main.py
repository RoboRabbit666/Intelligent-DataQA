#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI

from app.api.routers import chat, knoq, knowledge

app = FastAPI(title="AI_PLATFORM", openapi_url=f"/api/v1/openapi.json")

app.include_router(knoq.router)
app.include_router(knowledge.router)
app.include_router(chat.router)