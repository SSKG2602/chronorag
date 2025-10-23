from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="ChronoRAG API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    from app.routes import answer, incident, ingest, policy, retrieve  # noqa: WPS433

    app.include_router(ingest.router)
    app.include_router(retrieve.router)
    app.include_router(answer.router)
    app.include_router(policy.router)
    app.include_router(incident.router)
    return app


app = create_app()
