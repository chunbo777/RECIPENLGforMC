# "main" module, e.g. import app.main

from fastapi import FastAPI
from core.config import settings
from html.html import getContents
from generation.run_generation import main
from fastapi.responses import HTMLResponse
import json
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

    # recipe = str(items)+'!@#$!@#$'
    recipe = main(items)
    # recipe = test
    # recipe = testfunc()
    # print(recipe)
    # print(json.dumps(recipe))
    return json.dumps(recipe)


import uvicorn


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True, access_log=False)
    

