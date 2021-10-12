# "main" module, e.g. import app.main

from fastapi import FastAPI
from core.config import settings
from html.html import getContents

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


    return f'{items}{settings.PROJECT_NAME}을(를) 이용한 레시피가 생성되었습니다.'