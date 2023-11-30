let operator="";
let firstnumber="";
let secondnumber="";
function appendNumber(num){
if (operator===''){
    firstnumber +=num;
    document.getElementById("result").value=firstnumber;
}
else{
    secondnumber +=num;
    document.getElementById("result").value=secondnumber;
}
}
let result;
function calculate(){
    
    switch(operator){
        case '+':
            result=parseInt(firstnumber)+parseInt(secondnumber);
        break;
        case '-':
            result=parseInt(firstnumber)-parseInt(secondnumber);
        break;
        case '*':
            result=parseInt(firstnumber)*parseInt(secondnumber);
        break;
        case '/':
            result=parseInt(firstnumber)/parseInt(secondnumber);
        break;
    }
    document.getElementById("result").value=result;
}
function setoperator(op){
    operator=op;
}
function clearResult(){
    operator='';
    firstnumber='';
    secondnumber='';
    document.getElementById('result').value='';
}
