# "main" module, e.g. import app.main

from fastapi import FastAPI, File, UploadFile
from core.config import settings
from html.html import getHTML
from js.js import getScript
from generation.run_generation import main
from generation.get_entities import detect_text_uri, get_tag_from_db
from fastapi.responses import HTMLResponse
import json
import re
app = FastAPI()
@app.get("/")
async def root():

    html_contents = getHTML('html/main.html')
    js_contents = getScript('js/javascript.js')
    contents = re.sub('<script></script>',f'<script>{js_contents}</script>',html_contents)

    return HTMLResponse(contents, status_code=200)


@app.get("/ingredients/{ingredients}")
async def get_recipe(ingredients):

    # recipe = str(ingredients)
    recipe = main(ingredients)
    return json.dumps(recipe)

@app.get("/get_tag/{word}")
async def get_tag(word):
    # recipe = str(ingredients)
    # recipe = main(ingredients)
    tag = get_tag_from_db(word)
    return tag


from typing import List

# @app.post("/files/")
# async def create_files(files: List[bytes] = File(...)):
#     return {"file_sizes": [len(file) for file in files]}


import os
@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    result = {}
    for uploadFile in files:
        # with open(f'{os.path.dirname(__file__)}/generation/resources/{uploadFile.filename}.png', 'wb') as f:
        #     f.write(uploadFile.file.read())
        entities = detect_text_uri(uploadFile.file)
        result.update({uploadFile.filename :entities})
    # return {"filenames": [file.filename for file in files]}
    return result


import uvicorn

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)# You must pass the application as an import string to enable 'reload' or 'workers'.
    uvicorn.run(app, host="0.0.0.0", port=8080)
    

