# "main" module, e.g. import app.main

from fastapi import FastAPI
from core.config import settings
from html.html import getContents
from generation.run_generation import main
from fastapi.responses import HTMLResponse
import json
app = FastAPI()

@app.get("/")
async def root():

    contents = getContents('html/main.html')

    return HTMLResponse(contents, status_code=200)


@app.get("/ingredients/{ingredients}")
async def read_item(ingredients):

    recipe = str(ingredients)
    # recipe = main(ingredients)
    return json.dumps(recipe)


    
import uvicorn

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)# You must pass the application as an import string to enable 'reload' or 'workers'.
    uvicorn.run(app, host="0.0.0.0", port=8080)
    

