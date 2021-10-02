from fastapi import FastAPI

import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# sys.path.insert(0, os.path.abspath('..'))
# from RECIPENLGforMC.generation_prac.run_generation import main
# from generation_prac.run_generation import main
from generation_prac.run_generation import main
app = FastAPI()



@app.get("/")
async def root():
    # return {"message": "Hello World "}
    return '안녕하세요'

# txt = main('우유')
# print(txt)

@app.get("/{item_id}")
async def read_item(item_id):
    # test = main('우유')
    # print(item_id)
    # # test = main(item_id)
    return f'안녕하세요 {item_id}님!'