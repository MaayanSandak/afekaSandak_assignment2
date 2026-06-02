// ==========================================================================
//  מודל CNN מאפס בצד הלקוח - JavaScript נקי, ללא ספריות
//  מבנה: Convolution -> ReLU -> Max Pooling -> Flatten -> Dense -> Softmax
// ==========================================================================

// --------------------------------------------------------------------------
// 1. רכיבי ממשק (DOM)
// --------------------------------------------------------------------------
const canvas = document.getElementById('paint-canvas');
const ctx = canvas.getContext('2d');

const btnClear = document.getElementById('btn-clear-canvas');
const btnLockParams = document.getElementById('btn-lock-params');
const btnAddSample = document.getElementById('btn-add-sample');
const btnTrain = document.getElementById('btn-train');
const btnPredict = document.getElementById('btn-predict');
const btnReset = document.getElementById('btn-reset');
const btnExport = document.getElementById('btn-export');

const numLayersInput = document.getElementById('num-layers');
const numNeuronsInput = document.getElementById('num-neurons');
const numFiltersInput = document.getElementById('num-filters');
const filterSizeSelect = document.getElementById('filter-size');
const learningRateInput = document.getElementById('learning-rate');
const epochsInput = document.getElementById('epochs');
const paramInputs = [numLayersInput, numNeuronsInput, numFiltersInput, filterSizeSelect, learningRateInput, epochsInput];

const modelStatus = document.getElementById('model-status');
const modelArch = document.getElementById('model-arch');
const datasetSize = document.getElementById('dataset-size');
const currentEpoch = document.getElementById('current-epoch');
const currentLoss = document.getElementById('current-loss');
const currentAccuracy = document.getElementById('current-accuracy');
const predictedShape = document.getElementById('predicted-shape');
const probBars = document.getElementById('prob-bars');
const featureMapsDiv = document.getElementById('feature-maps');
const targetShapeSelect = document.getElementById('target-shape');

const GRID = 14;                    // התמונה מוקטנת לרשת 14x14
const NUM_CLASSES = 3;              // 0=עיגול, 1=ריבוע, 2=משולש
const SHAPE_NAMES = ["עיגול ⭕", "ריבוע 🔲", "משולש 🔺"];

let nnModel = null;
let dataset = { X: [], Y: [] };     // סט האימון (דוגמאות שהמשתמש מוסיף + דוגמאות אוטומטיות)

// --------------------------------------------------------------------------
// 2. הציור: מצייר חזותית על ה-Canvas וגם רושם לרשת לוגית 14x14
// --------------------------------------------------------------------------
let isDrawing = false;
let drawGrid = createEmptyGrid();   // מטריצת הקלט הלוגית של הציור הנוכחי

function createEmptyGrid() {
    let g = [];
    for (let y = 0; y < GRID; y++) g.push(new Array(GRID).fill(0));
    return g;
}

ctx.lineWidth = 16;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#ffffff';

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid = createEmptyGrid();
}
clearCanvas();

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

// רושם נקודה לרשת הלוגית 14x14 לפי מיקום הסמן
function markGridFromPoint(x, y) {
    let cellSize = canvas.width / GRID;
    let gx = Math.floor(x / cellSize);
    let gy = Math.floor(y / cellSize);
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID) drawGrid[gy][gx] = 1;
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    const pos = getMousePos(e);
    ctx.moveTo(pos.x, pos.y);
    markGridFromPoint(pos.x, pos.y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    markGridFromPoint(pos.x, pos.y);
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);
btnClear.addEventListener('click', clearCanvas);

// מחזיר עותק של מטריצת הקלט של הציור הנוכחי
function getInputGrid() {
    return drawGrid.map(row => row.slice());
}

// --------------------------------------------------------------------------
// 3. מחלקת ה-CNN מאפס (קונבולוציה + pooling + שכבות מלאות + backprop)
// --------------------------------------------------------------------------
function relu(x) { return x > 0 ? x : 0; }

class ConvNet {
    constructor(numFilters, filterSize, numHiddenLayers, hiddenSize, lr) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.numHiddenLayers = numHiddenLayers;
        this.hiddenSize = hiddenSize;
        this.lr = lr;

        this.convOut = GRID - filterSize + 1;        // קונבולוציית "Valid" (ללא ריפוד)
        this.poolOut = Math.floor(this.convOut / 2); // Max Pooling 2x2, stride 2
        this.flatSize = numFilters * this.poolOut * this.poolOut;

        // אתחול פילטרים והיסטים של שכבת הקונבולוציה (He initialization)
        this.filters = [];
        this.fBias = [];
        const fanIn = filterSize * filterSize;
        for (let f = 0; f < numFilters; f++) {
            let m = [];
            for (let r = 0; r < filterSize; r++) {
                let row = [];
                for (let c = 0; c < filterSize; c++) row.push((Math.random() * 2 - 1) * Math.sqrt(2 / fanIn));
                m.push(row);
            }
            this.filters.push(m);
            this.fBias.push(0);
        }

        // גדלי השכבות המלאות: [flat, hidden..., classes]
        this.sizes = [this.flatSize];
        for (let i = 0; i < numHiddenLayers; i++) this.sizes.push(hiddenSize);
        this.sizes.push(NUM_CLASSES);

        // אתחול משקלים והיסטים של השכבות המלאות
        this.W = [];
        this.B = [];
        for (let i = 0; i < this.sizes.length - 1; i++) {
            let inS = this.sizes[i], outS = this.sizes[i + 1];
            let w = [];
            for (let a = 0; a < inS; a++) {
                let row = [];
                for (let b = 0; b < outS; b++) row.push((Math.random() * 2 - 1) * Math.sqrt(2 / inS));
                w.push(row);
            }
            this.W.push(w);
            this.B.push(new Array(outS).fill(0));
        }
    }

    // ---- מעבר קדימה (Forward Pass) ----
    forward(grid) {
        // שכבת קונבולוציה + ReLU
        let convPre = [], convAct = [];
        for (let f = 0; f < this.numFilters; f++) {
            let pre = [], act = [];
            for (let i = 0; i < this.convOut; i++) {
                let preRow = [], actRow = [];
                for (let j = 0; j < this.convOut; j++) {
                    let s = this.fBias[f];
                    for (let di = 0; di < this.filterSize; di++)
                        for (let dj = 0; dj < this.filterSize; dj++)
                            s += grid[i + di][j + dj] * this.filters[f][di][dj];
                    preRow.push(s);
                    actRow.push(relu(s));
                }
                pre.push(preRow); act.push(actRow);
            }
            convPre.push(pre); convAct.push(act);
        }

        // שכבת Max Pooling 2x2
        let pool = [], argMax = [];
        for (let f = 0; f < this.numFilters; f++) {
            let pm = [], am = [];
            for (let pi = 0; pi < this.poolOut; pi++) {
                let pr = [], ar = [];
                for (let pj = 0; pj < this.poolOut; pj++) {
                    let best = -Infinity, bestR = 0, bestC = 0;
                    for (let r = 0; r < 2; r++)
                        for (let c = 0; c < 2; c++) {
                            let rr = pi * 2 + r, cc = pj * 2 + c;
                            let v = convAct[f][rr][cc];
                            if (v > best) { best = v; bestR = rr; bestC = cc; }
                        }
                    pr.push(best); ar.push([bestR, bestC]);
                }
                pm.push(pr); am.push(ar);
            }
            pool.push(pm); argMax.push(am);
        }

        // שיטוח (Flatten)
        let flat = [];
        for (let f = 0; f < this.numFilters; f++)
            for (let pi = 0; pi < this.poolOut; pi++)
                for (let pj = 0; pj < this.poolOut; pj++)
                    flat.push(pool[f][pi][pj]);

        // שכבות מלאות (Dense)
        let acts = [flat], cur = flat;
        for (let i = 0; i < this.W.length; i++) {
            let next = new Array(this.sizes[i + 1]).fill(0);
            for (let o = 0; o < this.sizes[i + 1]; o++) {
                let s = this.B[i][o];
                for (let a = 0; a < this.sizes[i]; a++) s += cur[a] * this.W[i][a][o];
                next[o] = (i === this.W.length - 1) ? s : relu(s);
            }
            acts.push(next);
            cur = next;
        }

        // נרמול Softmax (יציב מספרית)
        let scores = acts[acts.length - 1];
        let mx = Math.max(...scores);
        let ex = scores.map(s => Math.exp(s - mx));
        let sum = ex.reduce((a, b) => a + b, 0);
        let probs = ex.map(e => e / (sum || 1));

        this.cache = { grid, convPre, convAct, argMax, flat, acts, probs };
        return probs;
    }

    // ---- הפצה לאחור (Backpropagation) + עדכון משקלים ----
    backward(target) {
        const C = this.cache;

        // 1. גרדיאנט שכבת הפלט (Softmax + Cross-Entropy)
        let grads = [];
        let last = this.W.length - 1;
        let dScore = C.probs.slice();
        dScore[target] -= 1;
        grads[last] = dScore;

        // 2. הפצה לאחור דרך השכבות המלאות
        for (let i = last - 1; i >= 0; i--) {
            let g = new Array(this.sizes[i + 1]).fill(0);
            for (let j = 0; j < this.sizes[i + 1]; j++) {
                let s = 0;
                for (let k = 0; k < this.sizes[i + 2]; k++) s += grads[i + 1][k] * this.W[i + 1][j][k];
                g[j] = C.acts[i + 1][j] > 0 ? s : 0;   // נגזרת ReLU
            }
            grads[i] = g;
        }

        // 3. גרדיאנט ביחס לקלט השכבה המלאה הראשונה (הווקטור המשוטח)
        let dFlat = new Array(this.flatSize).fill(0);
        for (let k = 0; k < this.flatSize; k++) {
            let s = 0;
            for (let o = 0; o < this.sizes[1]; o++) s += grads[0][o] * this.W[0][k][o];
            dFlat[k] = s;
        }

        // 4. עדכון משקלי השכבות המלאות
        for (let i = 0; i < this.W.length; i++) {
            for (let a = 0; a < this.sizes[i]; a++)
                for (let o = 0; o < this.sizes[i + 1]; o++)
                    this.W[i][a][o] -= this.lr * grads[i][o] * C.acts[i][a];
            for (let o = 0; o < this.sizes[i + 1]; o++)
                this.B[i][o] -= this.lr * grads[i][o];
        }

        // 5. פירוק dFlat חזרה למבנה מפות ה-Pooling
        let dPool = [];
        let idx = 0;
        for (let f = 0; f < this.numFilters; f++) {
            let m = [];
            for (let pi = 0; pi < this.poolOut; pi++) {
                let row = [];
                for (let pj = 0; pj < this.poolOut; pj++) row.push(dFlat[idx++]);
                m.push(row);
            }
            dPool.push(m);
        }

        // 6. ניתוב הגרדיאנט דרך ה-Max Pooling אל מפת הקונבולוציה (כולל נגזרת ReLU)
        let dConvPre = [];
        for (let f = 0; f < this.numFilters; f++) {
            let m = [];
            for (let i = 0; i < this.convOut; i++) m.push(new Array(this.convOut).fill(0));
            dConvPre.push(m);
        }
        for (let f = 0; f < this.numFilters; f++)
            for (let pi = 0; pi < this.poolOut; pi++)
                for (let pj = 0; pj < this.poolOut; pj++) {
                    let pos = C.argMax[f][pi][pj];
                    let r = pos[0], c = pos[1];
                    if (C.convPre[f][r][c] > 0) dConvPre[f][r][c] += dPool[f][pi][pj];
                }

        // 7. עדכון פילטרים והיסטים של שכבת הקונבולוציה
        for (let f = 0; f < this.numFilters; f++) {
            let db = 0;
            let dW = [];
            for (let di = 0; di < this.filterSize; di++) dW.push(new Array(this.filterSize).fill(0));
            for (let i = 0; i < this.convOut; i++)
                for (let j = 0; j < this.convOut; j++) {
                    let d = dConvPre[f][i][j];
                    if (d === 0) continue;
                    db += d;
                    for (let di = 0; di < this.filterSize; di++)
                        for (let dj = 0; dj < this.filterSize; dj++)
                            dW[di][dj] += d * C.grid[i + di][j + dj];
                }
            for (let di = 0; di < this.filterSize; di++)
                for (let dj = 0; dj < this.filterSize; dj++)
                    this.filters[f][di][dj] -= this.lr * dW[di][dj];
            this.fBias[f] -= this.lr * db;
        }

        // החזרת ערך ה-Loss (Cross-Entropy)
        return -Math.log(Math.max(1e-7, C.probs[target]));
    }

    predict(grid) {
        let p = this.forward(grid);
        let mi = 0;
        for (let i = 1; i < p.length; i++) if (p[i] > p[mi]) mi = i;
        return { cls: mi, probs: p };
    }

    // המרת המודל לאובייקט לשמירה
    serialize() {
        return {
            numFilters: this.numFilters, filterSize: this.filterSize,
            numHiddenLayers: this.numHiddenLayers, hiddenSize: this.hiddenSize,
            lr: this.lr, sizes: this.sizes,
            filters: this.filters, fBias: this.fBias, W: this.W, B: this.B
        };
    }

    // בנייה מחדש מאובייקט שמור
    static fromData(d) {
        let net = new ConvNet(d.numFilters, d.filterSize, d.numHiddenLayers, d.hiddenSize, d.lr);
        net.sizes = d.sizes;
        net.filters = d.filters;
        net.fBias = d.fBias;
        net.W = d.W;
        net.B = d.B;
        return net;
    }
}

// --------------------------------------------------------------------------
// 4. מחולל סט אימון אוטומטי (צורות סינתטיות: עיגול, ריבוע, משולש)
// --------------------------------------------------------------------------
function blankGrid() { return createEmptyGrid(); }
function setPixel(g, x, y) { if (x >= 0 && x < GRID && y >= 0 && y < GRID) g[y][x] = 1; }

// אלגוריתם קו (Bresenham) לציור קצוות הצורות
function drawLine(g, x0, y0, x1, y1) {
    let dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
    let sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
    while (true) {
        setPixel(g, x0, y0);
        if (x0 === x1 && y0 === y1) break;
        let e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}
function rnd(a, b) { return a + Math.random() * (b - a); }

function genCircle() {
    let g = blankGrid(), cx = rnd(6, 8), cy = rnd(6, 8), r = rnd(3.5, 5.2);
    for (let a = 0; a < 360; a += 6) {
        let x = Math.round(cx + r * Math.cos(a * Math.PI / 180));
        let y = Math.round(cy + r * Math.sin(a * Math.PI / 180));
        setPixel(g, x, y);
    }
    return g;
}
function genSquare() {
    let g = blankGrid(), r = Math.round(rnd(3, 5)), cx = Math.round(rnd(6, 8)), cy = Math.round(rnd(6, 8));
    drawLine(g, cx - r, cy - r, cx + r, cy - r);
    drawLine(g, cx + r, cy - r, cx + r, cy + r);
    drawLine(g, cx + r, cy + r, cx - r, cy + r);
    drawLine(g, cx - r, cy + r, cx - r, cy - r);
    return g;
}
function genTriangle() {
    let g = blankGrid(), r = Math.round(rnd(3.5, 5.5)), cx = Math.round(rnd(6, 8)), cy = Math.round(rnd(6, 8));
    let ax = cx, ay = cy - r, bx = cx - r, by = cy + r, dx = cx + r, dy = cy + r;
    drawLine(g, ax, ay, bx, by);
    drawLine(g, bx, by, dx, dy);
    drawLine(g, dx, dy, ax, ay);
    return g;
}

// מוסיף דוגמאות אוטומטיות מתויגות לסט האימון
function seedDataset(perClass) {
    for (let i = 0; i < perClass; i++) {
        dataset.X.push(genCircle()); dataset.Y.push(0);
        dataset.X.push(genSquare()); dataset.Y.push(1);
        dataset.X.push(genTriangle()); dataset.Y.push(2);
    }
}

// --------------------------------------------------------------------------
// 5. אימון, דיוק ושמירה
// --------------------------------------------------------------------------
function evaluateAccuracy() {
    if (dataset.X.length === 0) return 0;
    let ok = 0;
    for (let i = 0; i < dataset.X.length; i++)
        if (nnModel.predict(dataset.X[i]).cls === dataset.Y[i]) ok++;
    return 100 * ok / dataset.X.length;
}

function saveModelToStorage() {
    if (!nnModel) return;
    localStorage.setItem('trained_cnn_weights', JSON.stringify(nnModel.serialize()));
}

function loadModelFromObject(data) {
    nnModel = ConvNet.fromData(data);
    numLayersInput.value = data.numHiddenLayers;
    numNeuronsInput.value = data.hiddenSize;
    numFiltersInput.value = data.numFilters;
    filterSizeSelect.value = data.filterSize;
    learningRateInput.value = data.lr;
    updateArchLabel();
}

function updateArchLabel() {
    if (!nnModel) { modelArch.textContent = "-"; return; }
    modelArch.textContent =
        `קלט 14x14 ← Conv ${nnModel.numFilters}×${nnModel.filterSize}x${nnModel.filterSize} ← ReLU ← Pool 2x2 ← Flatten(${nnModel.flatSize}) ← ` +
        `${nnModel.numHiddenLayers}×Dense(${nnModel.hiddenSize}) ← Softmax(3)`;
}

function enableActionButtons() {
    btnTrain.disabled = false; btnTrain.classList.remove('btn-disabled');
    btnPredict.disabled = false; btnPredict.classList.remove('btn-disabled');
}

function setParamsLocked(locked) {
    paramInputs.forEach(inp => inp.disabled = locked);
    btnLockParams.textContent = locked ? "אתחל מחדש (שחרר פרמטרים)" : "קבע פרמטרים ואתחל רשת";
}

// --------------------------------------------------------------------------
// 6. תצוגת תוצאות: ברי הסתברות + מפות מאפיינים (Feature Maps)
// --------------------------------------------------------------------------
function renderProbBars(probs) {
    probBars.innerHTML = "";
    for (let i = 0; i < NUM_CLASSES; i++) {
        let pct = (probs[i] * 100).toFixed(1);
        let row = document.createElement('div');
        row.className = 'prob-row';
        let label = document.createElement('span');
        label.className = 'prob-label';
        label.textContent = SHAPE_NAMES[i];
        let track = document.createElement('div');
        track.className = 'prob-track';
        let fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.style.width = pct + '%';
        let val = document.createElement('span');
        val.className = 'prob-val';
        val.textContent = pct + '%';
        track.appendChild(fill);
        row.appendChild(label); row.appendChild(track); row.appendChild(val);
        probBars.appendChild(row);
    }
}

// מצייר את מפת המאפיינים של כל פילטר על Canvas קטן
function renderFeatureMaps() {
    featureMapsDiv.innerHTML = "";
    if (!nnModel || !nnModel.cache) return;
    const convAct = nnModel.cache.convAct;
    const size = nnModel.convOut;
    const scale = 6;
    for (let f = 0; f < nnModel.numFilters; f++) {
        // מציאת הערך המקסימלי לנירמול התצוגה
        let mx = 0;
        for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) if (convAct[f][i][j] > mx) mx = convAct[f][i][j];
        let fmCanvas = document.createElement('canvas');
        fmCanvas.width = size * scale;
        fmCanvas.height = size * scale;
        fmCanvas.className = 'feature-map';
        let fctx = fmCanvas.getContext('2d');
        for (let i = 0; i < size; i++)
            for (let j = 0; j < size; j++) {
                let v = mx > 0 ? convAct[f][i][j] / mx : 0;
                let c = Math.round(v * 255);
                fctx.fillStyle = `rgb(${c},${c},${c})`;
                fctx.fillRect(j * scale, i * scale, scale, scale);
            }
        let wrap = document.createElement('div');
        wrap.className = 'feature-map-wrap';
        let cap = document.createElement('div');
        cap.className = 'feature-map-cap';
        cap.textContent = 'פילטר ' + (f + 1);
        wrap.appendChild(fmCanvas);
        wrap.appendChild(cap);
        featureMapsDiv.appendChild(wrap);
    }
}

// --------------------------------------------------------------------------
// 7. אירועים וכפתורי הממשק
// --------------------------------------------------------------------------
btnLockParams.addEventListener('click', () => {
    let layers = parseInt(numLayersInput.value);
    let neurons = parseInt(numNeuronsInput.value);
    let filters = parseInt(numFiltersInput.value);
    let fSize = parseInt(filterSizeSelect.value);
    let lr = parseFloat(learningRateInput.value);

    // יצירת רשת חדשה לפי הבחירה של המשתמש
    nnModel = new ConvNet(filters, fSize, layers, neurons, lr);

    // אתחול סט אימון עם דוגמאות אוטומטיות (אפשר להוסיף עוד ידנית)
    dataset = { X: [], Y: [] };
    seedDataset(20);

    setParamsLocked(true);
    modelStatus.textContent = "מוכן לאימון (הפרמטרים קובעו)";
    modelStatus.className = "status-text text-green";
    updateArchLabel();
    datasetSize.textContent = dataset.X.length + " דוגמאות";
    enableActionButtons();

    currentEpoch.textContent = "-";
    currentLoss.textContent = "-";
    currentAccuracy.textContent = "-";
    predictedShape.textContent = "-";
    probBars.innerHTML = "";
    featureMapsDiv.innerHTML = "";
});

btnAddSample.addEventListener('click', () => {
    let grid = getInputGrid();
    // בדיקה שהציור אינו ריק
    let sum = 0;
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) sum += grid[y][x];
    if (sum < 3) { alert("נא לצייר צורה לפני ההוספה לסט האימון."); return; }
    dataset.X.push(grid);
    dataset.Y.push(parseInt(targetShapeSelect.value));
    datasetSize.textContent = dataset.X.length + " דוגמאות";
    modelStatus.textContent = "נוספה דוגמה לסט האימון - יש לאמן מחדש";
    clearCanvas();
});

btnTrain.addEventListener('click', () => {
    if (!nnModel) return;
    let epochs = parseInt(epochsInput.value);
    modelStatus.textContent = "מאמן...";

    // לולאת אימון: SGD על כל הדוגמאות, עם ערבוב בכל Epoch
    let finalLoss = 0;
    for (let e = 1; e <= epochs; e++) {
        let order = [];
        for (let i = 0; i < dataset.X.length; i++) order.push(i);
        for (let i = order.length - 1; i > 0; i--) {       // ערבוב Fisher-Yates
            let j = Math.floor(Math.random() * (i + 1));
            let t = order[i]; order[i] = order[j]; order[j] = t;
        }
        let epochLoss = 0;
        for (let k = 0; k < order.length; k++) {
            let ii = order[k];
            nnModel.forward(dataset.X[ii]);
            epochLoss += nnModel.backward(dataset.Y[ii]);
        }
        finalLoss = epochLoss / dataset.X.length;
        currentEpoch.textContent = `${e} / ${epochs}`;
        currentLoss.textContent = finalLoss.toFixed(4);
    }

    // חישוב דיוק אמיתי מול סט האימון
    let acc = evaluateAccuracy();
    currentAccuracy.textContent = acc.toFixed(1) + "%";

    saveModelToStorage();
    modelStatus.textContent = "מאומן (המשקלים נשמרו ב-LocalStorage)";
    modelStatus.className = "status-text text-green";
});

btnPredict.addEventListener('click', () => {
    if (!nnModel) return;
    let grid = getInputGrid();
    let sum = 0;
    for (let y = 0; y < GRID; y++) for (let x = 0; x < GRID; x++) sum += grid[y][x];
    if (sum < 3) { alert("נא לצייר צורה לפני הבדיקה."); return; }

    let result = nnModel.predict(grid);
    predictedShape.textContent = SHAPE_NAMES[result.cls];
    renderProbBars(result.probs);
    renderFeatureMaps();
});

btnReset.addEventListener('click', () => {
    localStorage.removeItem('trained_cnn_weights');
    nnModel = null;
    dataset = { X: [], Y: [] };
    clearCanvas();
    setParamsLocked(false);

    modelStatus.textContent = "לא מאותחל";
    modelStatus.className = "status-text text-red";
    modelArch.textContent = "-";
    datasetSize.textContent = "-";

    btnTrain.disabled = true; btnTrain.classList.add('btn-disabled');
    btnPredict.disabled = true; btnPredict.classList.add('btn-disabled');

    currentEpoch.textContent = "-";
    currentLoss.textContent = "-";
    currentAccuracy.textContent = "-";
    predictedShape.textContent = "-";
    probBars.innerHTML = "";
    featureMapsDiv.innerHTML = "";
});

// ייצוא המשקלים לקובץ JSON (להעלאה ל-GitHub כגיבוי)
btnExport.addEventListener('click', () => {
    if (!nnModel) { alert("אין מודל לייצוא. יש לקבוע פרמטרים ולאמן."); return; }
    let dataStr = "window.PRETRAINED_MODEL = " + JSON.stringify(nnModel.serialize()) + ";\n";
    let blob = new Blob([dataStr], { type: 'text/javascript' });
    let url = URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url;
    a.download = 'pretrained.js';
    a.click();
    URL.revokeObjectURL(url);
});

// --------------------------------------------------------------------------
// 8. טעינה ראשונית: LocalStorage, אחרת המודל המאומן מ-GitHub (pretrained.js)
// --------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
    let loaded = false;
    let saved = localStorage.getItem('trained_cnn_weights');
    if (saved) {
        try {
            loadModelFromObject(JSON.parse(saved));
            modelStatus.textContent = "מאומן וטעון (מ-LocalStorage)";
            modelStatus.className = "status-text text-green";
            loaded = true;
        } catch (err) { console.log("שגיאה בטעינה מ-LocalStorage", err); }
    }
    if (!loaded && typeof window.PRETRAINED_MODEL !== 'undefined') {
        loadModelFromObject(window.PRETRAINED_MODEL);
        modelStatus.textContent = "מאומן וטעון (מ-GitHub / pretrained.js)";
        modelStatus.className = "status-text text-green";
        loaded = true;
    }
    if (loaded) {
        seedDataset(20);                 // סט הערכה לדיוק
        datasetSize.textContent = dataset.X.length + " דוגמאות";
        setParamsLocked(true);
        enableActionButtons();
        currentAccuracy.textContent = evaluateAccuracy().toFixed(1) + "%";
    }
});
