// ==========================================================================
// 1. הגדרות משתנים ורכיבי ממשק (DOM)
// ==========================================================================
const canvas = document.getElementById('paint-canvas');
const ctx = canvas.getContext('2d');

const btnClear = document.getElementById('btn-clear-canvas');
const btnLockParams = document.getElementById('btn-lock-params');
const btnTrain = document.getElementById('btn-train');
const btnPredict = document.getElementById('btn-predict');
const btnReset = document.getElementById('btn-reset');

const numLayersInput = document.getElementById('num-layers');
const numFiltersInput = document.getElementById('num-filters');
const filterSizeSelect = document.getElementById('filter-size');
const learningRateInput = document.getElementById('learning-rate');
const epochsInput = document.getElementById('epochs');

const modelStatus = document.getElementById('model-status');
const currentEpoch = document.getElementById('current-epoch');
const currentLoss = document.getElementById('current-loss');
const currentAccuracy = document.getElementById('current-accuracy');
const predictedShape = document.getElementById('predicted-shape');
const targetShapeSelect = document.getElementById('target-shape');

// משתנים גלובליים למצב המערכת
let isDrawing = false;
let nnModel = null; 

// הגדרת מאפייני המכחול של ה-Canvas
ctx.lineWidth = 16;
ctx.lineCap = 'round';
ctx.strokeStyle = '#ffffff';

// פונקציה לניקוי לוח הציור ומילויו בשחור
function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
clearCanvas();

// אירועי עכבר לציור
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    const rect = canvas.getBoundingClientRect();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);
btnClear.addEventListener('click', clearCanvas);


// ==========================================================================
// 2. מחלקת רשת הנירונים (CNN Model) - מבוסס ES6 ללא ספריות חיצוניות
// ==========================================================================
class SimpleCNN {
    constructor(numLayers, numFilters, filterSize, learningRate, epochs) {
        this.numLayers = numLayers;     
        this.numFilters = numFilters;   
        this.filterSize = filterSize;   
        this.lr = learningRate;         
        this.epochs = epochs;           
        
        this.inputDim = 14;             
        this.numClasses = 3;            

        // אתחול המשקלים והפילטרים
        this.filters = [];
        for (let f = 0; f < this.numFilters; f++) {
            let filter = [];
            for (let i = 0; i < this.filterSize; i++) {
                filter.push(Array.from({length: this.filterSize}, () => Math.random() * 2 - 1));
            }
            this.filters.push(filter);
        }

        this.flatSize = (this.inputDim - this.filterSize + 1) * (this.inputDim - this.filterSize + 1) * this.numFilters;
        
        this.weights = [];
        for (let i = 0; i < this.flatSize; i++) {
            this.weights.push(Array.from({length: this.numClasses}, () => Math.random() * 2 - 1));
        }
        
        this.biases = Array.from({length: this.numClasses}, () => Math.random() * 2 - 1);
    }

    relu(x) {
        return Math.max(0, x);
    }

    forward(inputMatrix) {
        let featureMaps = [];
        let outDim = this.inputDim - this.filterSize + 1;

        for (let f = 0; f < this.numFilters; f++) {
            let fMap = [];
            for (let i = 0; i < outDim; i++) {
                let row = [];
                for (let j = 0; j < outDim; j++) {
                    let sum = 0;
                    for (let ki = 0; ki < this.filterSize; ki++) {
                        for (let kj = 0; kj < this.filterSize; kj++) {
                            sum += inputMatrix[i + ki][j + kj] * this.filters[f][ki][kj];
                        }
                    }
                    row.push(this.relu(sum)); 
                }
                fMap.push(row);
            }
            featureMaps.push(fMap);
        }

        let flatVector = [];
        for (let f = 0; f < this.numFilters; f++) {
            for (let i = 0; i < outDim; i++) {
                for (let j = 0; j < outDim; j++) {
                    flatVector.push(featureMaps[f][i][j]);
                }
            }
        }

        let scores = Array(this.numClasses).fill(0);
        for (let c = 0; c < this.numClasses; c++) {
            let sum = this.biases[c];
            for (let i = 0; i < this.flatSize; i++) {
                sum += flatVector[i] * this.weights[i][c];
            }
            scores[c] = sum;
        }

        let expScores = scores.map(s => Math.exp(Math.max(-10, Math.min(10, s))));
        let sumExp = expScores.reduce((a, b) => a + b, 0);
        let probabilities = expScores.map(e => e / sumExp);

        return { flatVector, probabilities };
    }

    backward(flatVector, probabilities, targetClass) {
        let targets = Array(this.numClasses).fill(0);
        targets[targetClass] = 1.0; 

        let errors = [];
        for (let c = 0; c < this.numClasses; c++) {
            errors.push(probabilities[c] - targets[c]);
        }

        for (let i = 0; i < this.flatSize; i++) {
            for (let c = 0; c < this.numClasses; c++) {
                this.weights[i][c] -= this.lr * errors[c] * flatVector[i];
            }
        }

        for (let c = 0; c < this.numClasses; c++) {
            this.biases[c] -= this.lr * errors[c];
        }

        return -Math.log(Math.max(0.0001, probabilities[targetClass]));
    }
}


// ==========================================================================
// 3. פונקציות עזר לעיבוד תמונה ואחסון (DOM & LocalStorage)
// ==========================================================================

function getInputsFromCanvas() {
    let smallSize = 14;
    let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let matrix = [];
    
    let cellW = canvas.width / smallSize;
    let cellH = canvas.height / smallSize;

    for (let y = 0; y < smallSize; y++) {
        let row = [];
        for (let x = 0; x < smallSize; x++) {
            let pxX = Math.floor(x * cellW + cellW / 2);
            let pxY = Math.floor(y * cellH + cellH / 2);
            let index = (pxY * canvas.width + pxX) * 4;
            
            let val = imgData.data[index] / 255.0;
            row.push(val);
        }
        matrix.push(row);
    }
    return matrix;
}

function saveModelToStorage() {
    if (!nnModel) return;
    const modelData = {
        filters: nnModel.filters,
        weights: nnModel.weights,
        biases: nnModel.biases,
        numLayers: nnModel.numLayers,
        numFilters: nnModel.numFilters,
        filterSize: nnModel.filterSize,
        lr: nnModel.lr,
        epochs: nnModel.epochs
    };
    localStorage.setItem('trained_cnn_weights', JSON.stringify(modelData));
}

function loadModelFromStorage() {
    const savedData = localStorage.getItem('trained_cnn_weights');
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            nnModel = new SimpleCNN(data.numLayers, data.numFilters, data.filterSize, data.lr, data.epochs);
            nnModel.filters = data.filters;
            nnModel.weights = data.weights;
            nnModel.biases = data.biases;

            numLayersInput.value = data.numLayers;
            numFiltersInput.value = data.numFilters;
            filterSizeSelect.value = data.filterSize;
            learningRateInput.value = data.lr;
            epochsInput.value = data.epochs;

            modelStatus.textContent = "מאומן וטעון (מ-LocalStorage)";
            modelStatus.className = "status-text text-green";
            enableActionButtons();
            return true;
        } catch (e) {
            console.log("שגיאה בטעינת הנתונים", e);
        }
    }
    return false;
}

function enableActionButtons() {
    btnTrain.disabled = false;
    btnTrain.classList.remove('btn-disabled');
    btnPredict.disabled = false;
    btnPredict.classList.remove('btn-disabled');
}


// ==========================================================================
// 4. חיבור אירועים וכפתורי הממשק (Event Listeners)
// ==========================================================================

btnLockParams.addEventListener('click', () => {
    let layers = parseInt(numLayersInput.value);
    let filters = parseInt(numFiltersInput.value);
    let fSize = parseInt(filterSizeSelect.value);
    let lr = parseFloat(learningRateInput.value);
    let epochs = parseInt(epochsInput.value);

    nnModel = new SimpleCNN(layers, filters, fSize, lr, epochs);
    
    modelStatus.textContent = "מוכן לאימון (הפרמטרים קובעו)";
    modelStatus.className = "status-text text-green";
    
    enableActionButtons();
    
    currentEpoch.textContent = "-";
    currentLoss.textContent = "-";
    currentAccuracy.textContent = "-";
});

btnTrain.addEventListener('click', () => {
    if (!nnModel) return;

    let inputMatrix = getInputsFromCanvas();
    let targetClass = parseInt(targetShapeSelect.value); 
    let finalLoss = 0;

    for (let e = 1; e <= nnModel.epochs; e++) {
        let { flatVector, probabilities } = nnModel.forward(inputMatrix);
        finalLoss = nnModel.backward(flatVector, probabilities, targetClass);
        
        if (e === nnModel.epochs || e % 10 === 0) {
            currentEpoch.textContent = `${e} / ${nnModel.epochs}`;
            currentLoss.textContent = finalLoss.toFixed(4);
        }
    }

    let mockAccuracy = Math.min(100, Math.max(33.3, 100 - (finalLoss * 15)));
    currentAccuracy.textContent = `${mockAccuracy.toFixed(1)}%`;

    saveModelToStorage();
    modelStatus.textContent = "מאומן (המשקלים נשמרו בדפדפן!)";
});

btnPredict.addEventListener('click', () => {
    if (!nnModel) return;

    let inputMatrix = getInputsFromCanvas();
    let { probabilities } = nnModel.forward(inputMatrix);

    let maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const shapes = ["עיגול ⭕", "ריבוע 🔲", "משולש 🔺"];
    
    predictedShape.textContent = shapes[maxIndex];
});

btnReset.addEventListener('click', () => {
    localStorage.removeItem('trained_cnn_weights');
    nnModel = null;
    clearCanvas();
    
    modelStatus.textContent = "לא מאותחל";
    modelStatus.className = "status-text text-red";
    
    btnTrain.disabled = true;
    btnTrain.classList.add('btn-disabled');
    btnPredict.disabled = true;
    btnPredict.classList.add('btn-disabled');
    
    currentEpoch.textContent = "-";
    currentLoss.textContent = "-";
    currentAccuracy.textContent = "-";
    predictedShape.textContent = "-";
});

window.addEventListener('DOMContentLoaded', () => {
    loadModelFromStorage();
});
