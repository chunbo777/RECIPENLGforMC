# "main" module, e.g. import app.main

from fastapi import FastAPI
from core.config import settings
from html.html import getContents
from generation.run_generation import main as gen
from fastapi.responses import HTMLResponse
app = FastAPI()
# from fastapi import APIRouter, Depends, HTTPException
# from .dependencies import get_token_header

# router = APIRouter(
#     prefix="/items",
#     tags=["items"],
#     dependencies=[Depends(get_token_header)],
#     responses={404: {"description": "Not found"}},
# )

@app.get("/")
async def root():

    contents = getContents('html/main.html')

    return HTMLResponse(contents, status_code=200)


@app.post("/item/{items}")
async def read_item(items):

    # recipe = gen('milk')
    recipe = str(items)
    return recipe