bgclasses = ['bg-primary','bg-secondary','bg-success', 'bg-danger', 'bg-warning', 'bg-info', 'bg-dark']
groupColors = ['list-group-item-success','list-group-item-secondary','list-group-item-info'
, 'list-group-item-warning', 'list-group-item-danger', 'list-group-item-primary'
, 'list-group-item-dark', 'list-group-item-light']
borderColors = ['border-primary','border-secondary','border-success','border-danger','border-warning'
,'border-info','border-light','border-dark','border-white']

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
            span.title = jsonData[jsonData.length-2]
            span.setAttribute('data-bs-toggle','tooltip') 
            span.className = 'm-2 badge '+bgclasses[jsonData[jsonData.length-1] == undefined? 0 : jsonData[jsonData.length-1]]
            span.id = 'saved'+$(e.target.parentElement.parentElement).find('span').length

            let h4 = document.createElement('h4')
            span.appendChild(h4)
            h4.innerHTML = jsonData[0]
            h4.setAttribute('draggable','true')

            h4.ondragstart =(e)=>{drag(e)}
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

allowDrop =(ev)=>{
    ev.preventDefault();
}
  
drag=(ev)=>{
    ev.dataTransfer.setData("text", ev.target.parentElement.id);
}
  
drop=(ev)=>{
    ev.preventDefault();
    let data = ev.dataTransfer.getData("text");
    draggedEl = document.getElementById(data)
    ev.target.appendChild(draggedEl);
    
    // // param 영역으로 옮겨진 el 수정시 tooltip이 삭제되지 않는 문제 해결
    // if(draggedEl.hasAttribute('data-bs-toggle')){
    //     draggedEl.setAttribute('data-bs-toggle','tooltip')  

    // }else{
    //     draggedEl.removeAttribute('data-bs-toggle')
    //     draggedEl.removeAttribute('data-bs-original-title')
    //     docu
    //     draggedEl.removeAttribute('aria-describedby')
    // }

    // let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    // let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    //     return new bootstrap.Tooltip(tooltipTriggerEl)
    // })

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
        e.target.elements[1].innerHTML += ' Loading..'
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
                    // document.getElementById('params').appendChild(span)
                    document.getElementById('detectedFromImg').appendChild(span)
                    span.title = data[data.length-2]
                    span.setAttribute('data-bs-toggle','tooltip')  
                    span.className = 'm-2 badge '+bgclasses[data[data.length-1] == undefined? 0 : data[data.length-1]]
                    span.id = 'detectedFromImg'+n

                    let h4 = document.createElement('h4')
                    span.appendChild(h4)
                    h4.innerHTML = data[0]
                    h4.setAttribute('draggable','true')
                    
                    h4.ondragstart =(e)=>{drag(e)} 
                    // h4.setAttribute('ondragstart','drag(event)')
                }
            }
            // Note: Tooltips must be initialized with JavaScript to work.
            let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
            let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
              return new bootstrap.Tooltip(tooltipTriggerEl)
            })


            document.getElementById('getentities').style.display = 'none'
            document.getElementById('detectedFromImg').parentElement.style.removeProperty('display')
            document.getElementById('params').parentElement.style.removeProperty('display')
        }
        xhttp.open("POST", location.href +"uploadfiles/");
        xhttp.setRequestHeader("enctype", "multipart/form-data");
        xhttp.send(new FormData(e.target));
    })

    $(document.body).on("click", (e)=>{
        
        console.log($(e))
        if($(e.target).parents('div#params').length>0 
         & e.target.tagName == 'H4' & e.target.parentElement.className.includes('badge') ){
            inputs = $(e.target.parentElement.parentElement).find('input')
            if (inputs.length >0){
                alert('수정중인 식재료를 저장해주세요')
                return
            }

            whenSpanClicked(e.target.parentElement.parentElement, e.target.innerHTML)
            document.getElementById(e.target.parentElement.getAttribute('aria-describedby')).remove()
            e.target.remove()
        }else if (e.target.id.includes('params') & e.target.tagName == 'DIV'){
            // 신규 추가
            whenSpanClicked(e.target, null)
        }else if (e.target.id.includes('getRecipe') & e.target.tagName == 'BUTTON'){

            const array =  $('div#params span.badge h4')
            if(array.length==0){
                alert('하나이상의 입력값이 필요합니다')
                return
            }else{
                
                $(e.target).empty()
                span = document.createElement('span')
                e.target.appendChild(span)
                span.className = 'spinner-border spinner-border-sm'
                e.target.innerHTML += ' Loading..'
                e.target.disabled = true
            }
                
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
                    div.className = 'mt-5 border border-5 rounded-3'
                    let h3 = document.createElement('h3')
                    h3.innerHTML = k
                    div.appendChild(h3)
                    let contents = document.createElement('div')
                    div.appendChild(contents)
                    contents.className = 'm-3 border border-5 rounded-3 '+borderColors[Math.floor(borderColors.length*Math.random())]
                    if (Array.isArray(jsonData[k])){
                        let innerDiv = document.createElement('div')
                        contents.appendChild(innerDiv)
                        if (k=='INSTR'){
                            innerDiv.className = 'list-group'
                            for (let n =0 ; n<jsonData[k].length; n++){
                                let a = document.createElement('a')
                                innerDiv.appendChild(a)
                                a.className = 'list-group-item list-group-item-action '+groupColors[Math.floor(groupColors.length*Math.random())]

                                let h4 = document.createElement('h4')
                                a.appendChild(h4)
                                // h5.className = 'text-dark'
                                h4.innerHTML = (n+1)+ ') '+jsonData[k][n]
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
                        span.className = 'm-3 badge '+bgclasses[Math.floor(bgclasses.length*Math.random())]
                        let h4 = document.createElement('h4')
                        span.appendChild(h4)
                        h4.innerHTML = jsonData[k]
                    }// if
                }// for (let k in jsonData){
            }// xhttp.onload = function() {
            xhttp.open("GET", location.href + "ingredients/"+foods.join(', '));
            // xhttp.open("POST", "http://127.0.0.1:8000/item/"+foods.join(', '));
            xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
            xhttp.send();
        }
    });
})