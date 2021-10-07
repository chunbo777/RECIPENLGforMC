# "main" module, e.g. import app.main

from fastapi import FastAPI
from core.config import settings
from fastapi.responses import HTMLResponse
app = FastAPI()
# from ..html import getContents
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
    # return {"message": "Hello World "}
    # contents = getContents('main.html')
    contents = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /*
        * Globals
        */

        /* Links */
        a,
        a:focus,
        a:hover {
        color: #fff;
        }
        /*
        * Base structure
        */

        html,
        body {
        height: 100%;
        background-color: #333;
        }

        body {
        color: #fff;
        }


    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <!-- Latest compiled and minified CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.1/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Latest compiled JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.1/dist/js/bootstrap.bundle.min.js"></script>

</head>
<script>
    whenSpanClicked = (el)=>{
        inputs = $(el.parentElement).find('input')
        if (inputs.length >0){
            alert('수정중인 식재료를 저장해주세요')
            return
        }
        div = document.createElement('div')
        el.parentElement.append(div)
        input = document.createElement('input')
        div.append(input)
        input.value = el.innerHTML
        btn = document.createElement('button')
        div.append(btn)
        btn.innerHTML = '저장'
        btn.onclick =(e)=>{
            span = document.createElement('span')
            e.target.parentElement.parentElement.appendChild(span)
            span.innerHTML = $(e.target.parentElement).find('input')[0].value
            spanclasses = ['primary','secondary','success', 'danger', 'warning', 'info', 'dark']
            span.className = 'badge bg-'+spanclasses[Math.floor(spanclasses.length*Math.random())]
            span.onclick = (event)=>{
                whenSpanClicked(event.target)
            }
            e.target.parentElement.remove()
        }
        el.remove()
    }


    $(()=>{
        $("span.badge").on("dblclick", (e)=>{
            console.log(e.target)
            e.target.remove()
        });
        $("span.badge").on("click", (e)=>{
            whenSpanClicked(e.target)
            // inputs = $(e.parentElement).find('input')
            // if (inputs.length >0){
            //     alert('수정중인 식재료를 저장해주세요')
            //     return
            // }
            // div = document.createElement('div')
            // e.target.parentElement.append(div)
            // input = document.createElement('input')
            // div.append(input)
            // input.value = e.target.innerHTML
            // btn = document.createElement('button')
            // div.append(btn)
            // btn.innerHTML = '저장'
            // btn.onclick =(e)=>{
            //     span = document.createElement('span')
            //     e.target.parentElement.parentElement.appendChild(span)
            //     span.innerHTML = $(e.target.parentElement).find('input')[0].value
            //     spanclasses = ['primary','secondary','success', 'danger', 'warning', 'info', 'dark']
            //     span.className = 'badge bg-'+spanclasses[Math.floor(spanclasses.length*Math.random())]
            //     e.target.parentElement.remove()
            // }
            // e.target.remove()
        });

        $("#getRecipe.btn.btn-primary").on("click", ()=>{
            //alert("The paragraph was clicked.");
            console.log($('span.badge'))
            const array = $('span.badge')
            let foods = new Array
            for (let index = 0; index < array.length; index++) {
                const element = array[index].innerHTML;
                foods.push(element)
            }
            
            const xhttp = new XMLHttpRequest();
            xhttp.onload = function() {
                document.getElementById("Recipe").innerHTML = this.responseText;
            }
            xhttp.open("POST", "http://127.0.0.1:8000/item/"+foods.join(', '));

            xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
            xhttp.send("fname=Henry&lname=Ford");            
        });        
    })
</script>
<body>
    <div class="container">
        <h1 class='.display-1'>셀럽시피&발명식당!</h1>
        <p>This is some text.</p>
        <div class="row">
            <div class="col-sm-4">
                <img class="img-fluid" src="https://th.bing.com/th/id/R.374b01f37e7993217f13b14ee9916223?rik=t5QB3IDWf93yAw&riu=http%3a%2f%2fpostfiles6.naver.net%2f20151116_53%2funesco1128_1447659335467lqOdR_GIF%2fnl9vkuA.gif%3ftype%3dw2&ehk=EqdpOrfDkLun5HLx1KzJwdoETDEZiG8Cwzw7qdhktFA%3d&risl=&pid=ImgRaw&r=0" alt="냠냠">
            </div>
            <div class="col-sm-8">
                <div class="row">
                    <div class="col-sm-8" id="params">
                        <span class="badge bg-primary">공심채</span>
                        <span class="badge bg-secondary">간마늘</span>
                        <span class="badge bg-success">페퍼톤치노</span>
                        <span class="badge bg-danger">치킨스톡</span>
                        <span class="badge bg-warning">피쉬소스</span>
                        <span class="badge bg-info">굴소스</span>
                        <span class="badge bg-dark">참기름</span>
                    </div>
                    <div class="col-sm-4"><button type="button" class="btn btn-primary" id="getRecipe">레시피 생성</button></div>


                </div>
                <div class="row">
                    <div class="progress">
                        <div class="progress-bar" style="width:70%">70%</div>
                    </div>
                    <div class="row" id='Recipe'>Recipe</div>
                </div>

            </div>
        </div>
    </div>
</body>
</html>
    '''
    return HTMLResponse(contents, status_code=200)

# txt = main('우유')
# print(txt)

@app.post("/item/{items}")
async def read_item(items):
    # test = main('우유')
    # print(item_id)
    # # test = main(item_id)
    return f'{items}을(를) 이용한 레시피가 생성되었습니다.'