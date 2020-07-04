let testInfo = document.getElementById("testInfo");
let iterationProgress = document.getElementById("iterationProgress");
let showIterations = document.getElementById("showIterations");
let modelLoad = document.getElementById("modelLoad");
let modelProgress = document.getElementById("modelProgress");
let showModel = document.getElementById("showModel");

let runButton = document.getElementById("runButton");
runButton.addEventListener("click", run);
// Run test only after opencv.js loaded
cv.onRuntimeInitialized = () => {
    runButton.disabled = false;
}

let imgelement = document.getElementById("imgsrc");
let inputelement = document.getElementById("fileinput");
inputelement.addEventListener("change", (e) =>{
    imgelement.src = URL.createObjectURL(e.target.files[0]);
}, false);

/*************************   CONTROL PARAMETERS   **************************/

// Record if the model have been loaded
let modelLoaded = [];
// The forward iterations set by user
let iterations = Number(document.querySelector('#iterations').value);
// Flag for first run
let initFlag = true;
// Count for iterations finished
let calIteration = 0;
// Number of top result to show
let topNum = 5;
// Save each forward time 
let timeSum = [];
// Lables for image classification result
let labels;
// Top result to show
let classes;

let colors;
let modelName = 'opt_deeplabv3_mnv2_224.pb';
let inputSize = [224, 224];

//Detect the click, init the UI and control parameters,
//check if the model have been loaded, then run the test.
function run(){
    initPara();

    colors = [0,0,0];
    while(colors.length<21*3){
        colors.push( Math.round( (Math.random()*255 + colors[colors.length-3]) / 2 ) );
    }

    let index = modelLoaded.indexOf(modelName);
    if(index === -1) {        
        showModel.innerHTML = 'Model loading...';
        createFileFromUrl(modelName, modelName, compute, modelState);
        modelLoaded.push(modelName);
    } else{
        modelLoaded.innerHTML = `${modelName} has been loaded before.`;
        compute();
    };
}

function getModelById(id){
    for(const modelInfo of modelZoo){
        if (id === modelInfo.name) {
            return modelInfo;
        };
    };
    return {};
}

//Init the UI and the control parameters.
function initPara(){
    iterations = Number(document.querySelector('#iterations').value);
    calIteration = 0;
    timeSum = [];

    testInfo.innerHTML = '';
    modelLoad.innerHTML = '';
    showModel.innerHTML = '';
    modelProgress.value = 0;
    modelProgress.style.visibility = 'hidden';
    iterationProgress.style.visibility = 'hidden';
    showIterations.innerHTML = '';

    if(initFlag){
        loadLables();
        initFlag = false;
    };
}

//Load labels from the txt for the first run.
function loadLables(){
    let url = 'labels1000.txt';
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.onload = function(ev) {
        if (request.readyState ===4 ) {
            if(request.status === 200) {
                labels = request.response;
                labels = labels.split('\n')
            };
        };
    };
    request.send();
}

//The whole compute pipeline.
function compute (){
    let mat = cv.imread("imgsrc");
    let matC3 = new cv.Mat(mat.matSize[0],mat.matSize[1],cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let matdata = matC3.data;
    let stddata = [];
    for(var i=0; i<mat.matSize[0]*mat.matSize[1]; ++i){
        stddata.push( (matdata[3*i]) ); 
        stddata.push( (matdata[3*i+1]) ); 
        stddata.push( (matdata[3*i+2]) );
    };
    let inputMat = cv.matFromArray(mat.matSize[0],mat.matSize[1],cv.CV_32FC3,stddata);         

    let net = cv.readNet(modelName);
    console.log('Start inference...')
    let input = cv.blobFromImage(inputMat, 0.007843, new cv.Size(inputSize[0], inputSize[1]), new cv.Scalar(127.5, 127.5, 127.5), true, false);
    net.setInput(input);

    let result = multiForward(net, 10);
    let finalResult = summarize(timeSum);
    console.log(`inference time: ${finalResult.mean.toFixed(2)} ± ${finalResult.std.toFixed(2)}ms`)

    timeSum = []; 
    result = multiForward(net, 100);
    finalResult = summarize(timeSum);
    console.log(`inference time: ${finalResult.mean.toFixed(2)} ± ${finalResult.std.toFixed(2)}ms`)

    C = result.matSize[1];
    H = result.matSize[2];
    W = result.matSize[3];

    let resultData = result.data32F;

    let classId = [];
    let argmax = [];

    let imgSize = H*W;
    for (i = 0; i<imgSize; ++i){
        tmp = 0;
        for (j = 0; j<C; ++j){
            if(resultData[j*imgSize+i] > resultData[tmp*imgSize+i]){
                tmp = j;
            }
        }
        argmax.push(tmp);
        classId.push(colors[tmp*3]);
        classId.push(colors[tmp*3+1]);
        classId.push(colors[tmp*3+2]);
        classId.push(255);
    }

    output = cv.matFromArray(H,W,cv.CV_8UC4,classId);


    cv.imshow('output',output)


    
    console.log('Test finished!');
    net.delete();
    result.delete();
}

function multiForward(net, num) {
    let result;
    for(let i=0; i<num+1; ++i) {
        let start = performance.now();
        result =net.forward();
        let elapsed = performance.now()-start;
        timeSum.push(elapsed);
        console.log(`num ${i} forward, time ${elapsed.toFixed(2)}ms`)
    }
    return result;    
}

//Load the file from the filesystem.
function createFileFromUrl(path, url, callback, onprogress){
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function(ev) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                showModel.innerHTML = 'Model loaded.';
                callback();
            } else {
                console.log('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
    request.onprogress = onprogress;
};

//Show the model status when load the model.
function modelState(ev){
    modelProgress.style.visibility = 'visible';
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    modelLoad.innerHTML = `${loadedSize.toFixed(2)}/${totalSize.toFixed(2)}MB ${percentComplete.toFixed(2)}%`;
    modelProgress.value = percentComplete;
}

function summarize(results) {
    if (results.length !== 0) {
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        let d = results.reduce((d, v) => {
            d.sum += v;
            d.sum2 += v * v;
            return d;
        }, {
            sum: 0,
            sum2: 0
        });
        let mean = d.sum / results.length;
        let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
        return {
            mean: mean,
            std: std
        };
    } else {
        return null;
    };
}
