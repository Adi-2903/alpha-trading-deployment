"""
api/index.py -- Vercel serverless entry point.

Vercel's @vercel/python builder looks for an `app` variable in this file.
We simply re-export the FastAPI application from api/server.py so all
routes, middleware, and CORS settings defined there are preserved.

Run locally (dev):
    uvicorn api.index:app --reload --port 8000
Deploy to Vercel:
    vercel deploy  (or git push to a connected repo)
"""
from api.server import app  # noqa: F401  -- re-exported for Vercel
