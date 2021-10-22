bgclasses = ['bg-primary','bg-secondary','bg-success', 'bg-danger', 'bg-warning', 'bg-info', 'bg-dark']
groupColors = ['list-group-item-success','list-group-item-secondary','list-group-item-info'
, 'list-group-item-warning', 'list-group-item-danger', 'list-group-item-primary'
, 'list-group-item-dark', 'list-group-item-light']

whenSpanClicked = (root, text)=>{

    div = document.createElement('div')
    root.append(div)
    div.className = 'input-group m-1 p-2'

    input = document.createElement('input')
    div.append(input)
    if (text != null){
        input.value = text
    }
    input.type = 'text'
    input.className = 'form-control'
    input.placeholder = '재료를 입력해주세요'

    save_btn = document.createElement('button')
    div.append(save_btn)
    save_btn.innerHTML = '저장'
    save_btn.className = 'btn btn-primary'
    save_btn.type = 'button'

    save_btn.onclick =(e)=>{
        const xhttp = new XMLHttpRequest();
        xhttp.onload = function() {
            jsonData = JSON.parse(this.responseText)
            span = document.createElement('span')
            e.target.parentElement.parentElement.appendChild(span)
            span.innerHTML = jsonData[0]
            span.title = jsonData[jsonData.length-2]
            span.setAttribute('data-bs-toggle','tooltip') 

            // span.className = 'badge '+bgclasses[Math.floor(bgclasses.length*Math.random())]
            span.className = 'badge '+bgclasses[jsonData[jsonData.length-1] == undefined? 0 : jsonData[jsonData.length-1]]
            e.target.parentElement.remove()

            // Note: Tooltips must be initialized with JavaScript to work.
            let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
            let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
              return new bootstrap.Tooltip(tooltipTriggerEl)
            })

        }
        xhttp.open("GET", location.href + "get_tag/"+$(e.target.parentElement).find('input')[0].value);
        xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        xhttp.send();

    }

    del_btn = document.createElement('button')
    div.append(del_btn)
    del_btn.innerHTML = '삭제'
    del_btn.onclick =(e)=>{
        e.target.parentElement.remove()
    }
    del_btn.className = 'btn btn-danger'
    del_btn.type = 'button'
}

$(()=>{
    $(document).on('change',(e)=>{
        if(e.target.tagName == 'INPUT' & e.target.type == 'file'){
            if(e.target.files.length>0){
                document.getElementById('img').src = URL.createObjectURL(e.target.files[0])
            }
        }
    })
    $(document).on('submit',(e)=>{
        e.preventDefault()
        if(e.target.elements[0].files.length==0){
            alert('파일을 선택해주세요')
            return
        }

        $(e.target.elements[1]).empty() 
        span = document.createElement('span')
        e.target.elements[1].appendChild(span)
        span.className = 'spinner-border spinner-border-sm'
        e.target.elements[1].innerHTML += 'Loading..'
        e.target.elements[1].disabled = true

        const xhttp = new XMLHttpRequest();
        xhttp.onload = function() {
            $(e.target.elements[1]).empty() 
            e.target.elements[1].className = 'btn btn-success'
            e.target.elements[1].innerHTML = '식재료 추출'
            e.target.elements[1].disabled = false


            jsonData = JSON.parse(this.responseText)
            for (const key in jsonData) {
                for(let n =0;n<jsonData[key].length;n++){
                    data = jsonData[key][n]
                    span = document.createElement('span')
                    document.getElementById('params').appendChild(span)
                    span.innerHTML = data[0]
                    span.title = data[data.length-2]
                    // span.setAttribute('data','bs-toggle="tooltip"') 
                    span.setAttribute('data-bs-toggle','tooltip')  
                    // span.className = 'badge '+bgclasses[Math.floor(bgclasses.length*Math.random())]
                    span.className = 'badge '+bgclasses[data[data.length-1] == undefined? 0 : data[data.length-1]]
                }
            }
            // Note: Tooltips must be initialized with JavaScript to work.
            let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
            let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
              return new bootstrap.Tooltip(tooltipTriggerEl)
            })


            document.getElementById('getentities').style.display = 'none'
            document.getElementById('params').parentElement.style.removeProperty('display')
        }
        xhttp.open("POST", location.href +"uploadfiles/");
        xhttp.setRequestHeader("enctype", "multipart/form-data");
        xhttp.send(new FormData(e.target));
    })

    $(document.body).on("click", (e)=>{
        
        console.log($(e))
        if(e.target.className.includes('badge') & e.target.tagName == 'SPAN'){
            inputs = $(e.target.parentElement).find('input')
            if (inputs.length >0){
                alert('수정중인 식재료를 저장해주세요')
                return
            }

            whenSpanClicked(e.target.parentElement, e.target.innerHTML)

            e.target.remove()

        }else if (e.target.id.includes('getRecipe') & e.target.tagName == 'BUTTON'){

            $(e.target).empty()
            span = document.createElement('span')
            e.target.appendChild(span)
            span.className = 'spinner-border spinner-border-sm'
            e.target.innerHTML += 'Loading..'
            e.target.disabled = true

            const array =  $('div#params span.badge')
            let foods = new Array
            for (let index = 0; index < array.length; index++) {
                const element = array[index].innerHTML;
                foods.push(element)
            }
            
            const xhttp = new XMLHttpRequest();
            xhttp.onload = function() {

                $(e.target).empty() 
                e.target.className = 'btn btn-primary'
                e.target.innerHTML = '레시피 생성'
                e.target.disabled = false

                const jsonData = JSON.parse(JSON.parse(this.responseText))// parse
                let root = document.getElementById("Recipe")

                $(root).empty();

                for (let k in jsonData){
                    let div = document.createElement('div')
                    root.appendChild(div)
                    let h3 = document.createElement('h3')
                    h3.innerHTML = k
                    div.appendChild(h3)
                    let contents = document.createElement('div')
                    div.appendChild(contents)
                    if (Array.isArray(jsonData[k])){
                        let innerDiv = document.createElement('div')
                        contents.appendChild(innerDiv)
                        if (k=='INSTR'){
                            innerDiv.className = 'list-group'
                            for (let n =0 ; n<jsonData[k].length; n++){
                                let a = document.createElement('a')
                                innerDiv.appendChild(a)
                                a.className = 'list-group-item list-group-item-action '+groupColors[Math.floor(groupColors.length*Math.random())]
                                a.innerHTML = (n+1)+ ') '+jsonData[k][n]
                            }
                        }else {                            
                            for (let n =0 ; n<jsonData[k].length; n++){
                                let span = document.createElement('span')
                                innerDiv.appendChild(span)
                                span.className = 'm-3 badge '+bgclasses[Math.floor(bgclasses.length*Math.random())]

                                let h4 = document.createElement('h4')
                                span.appendChild(h4)
                                h4.innerHTML = jsonData[k][n]
                            }
                        }
                    }else{
                        let span = document.createElement('span')
                        contents.appendChild(span)
                        span.innerHTML = jsonData[k]
                    }// if
                }// for (let k in jsonData){
            }// xhttp.onload = function() {
            xhttp.open("GET", location.href + "ingredients/"+foods.join(', '));
            // xhttp.open("POST", "http://127.0.0.1:8000/item/"+foods.join(', '));
            xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
            xhttp.send();
        }else if (e.target.id.includes('params') & e.target.tagName == 'DIV'){
            whenSpanClicked(e.target, null)
        }
    });
})