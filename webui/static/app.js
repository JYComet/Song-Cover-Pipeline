/**
 * Song Cover Pipeline -- Web UI Application
 * Vanilla JS, no framework dependencies
 * Supports both single-file and batch processing
 */

// =========================================================================
// State
// =========================================================================
const STATE = {
    taskId: null,
    uploadId: null,
    ws: null,
    status: 'idle',  // idle | uploading | ready | processing | done | error
    audioCtx: null,
    playing: false,
    _activeSteps: null,
    _pollTimer: null,
    // Batch mode
    mode: null,           // null | 'single' | 'batch'
    batchId: null,
    batchWs: null,
    _batchPollTimer: null,
    files: [],            // [{uploadId, name, sizeMb, status}]
};

let _pendingFiles = [];  // temporary File references for sequential upload

// =========================================================================
// DOM References (cached on init)
// =========================================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let DOM = {};

function cacheDom() {
    DOM = {
        // Header
        gpuBadge: $('#gpu-badge'),
        diskBadge: $('#disk-badge'),
        serverStatus: $('#server-status'),

        // Sidebar
        modelSelect: $('#model-select'),
        uploadZone: $('#upload-zone'),
        fileInput: $('#file-input'),
        btnBrowse: $('#btn-browse'),
        btnStart: $('#btn-start'),
        fileQueue: $('#file-queue'),
        fileQueueList: $('#file-queue-list'),
        queueCount: $('#queue-count'),
        btnClearQueue: $('#btn-clear-queue'),

        // Main states
        welcomeState: $('#welcome-state'),
        processingState: $('#processing-state'),
        resultsState: $('#results-state'),
        errorState: $('#error-state'),

        // Progress
        progressFill: $('#progress-fill'),
        progressPercent: $('#progress-percent'),
        progressStage: $('#progress-stage'),
        progressMessage: $('#progress-message'),
        stageIndicators: $('#stage-indicators'),
        batchProgressSection: $('#batch-progress-section'),
        batchCurrentFile: $('#batch-current-file'),
        batchFileList: $('#batch-file-list'),

        // Single result
        singleResult: $('#single-result'),
        audioPlayer: $('#audio-player'),
        btnPlay: $('#btn-play'),
        audioSeek: $('#audio-seek'),
        audioVolume: $('#audio-volume'),
        audioCurrent: $('#audio-current'),
        audioDuration: $('#audio-duration'),
        btnDownload: $('#btn-download'),
        btnNewTask: $('#btn-new-task'),
        btnShowLog: $('#btn-show-log'),
        outputLinks: $('#output-links'),
        outputFilesList: $('#output-files-list'),

        // Batch results
        batchResults: $('#batch-results'),
        batchSummary: $('#batch-summary'),
        batchResultsList: $('#batch-results-list'),
        btnNewTaskBatch: $('#btn-new-task-batch'),

        // Shared log viewer
        logViewerResults: $('#log-viewer-results'),
        logContentResults: $('#log-content-results'),
        btnCloseLogResults: $('#btn-close-log-results'),

        // Error
        errorMessage: $('#error-message'),
        btnRetry: $('#btn-retry'),
        btnShowErrorLog: $('#btn-show-error-log'),
        logViewerError: $('#log-viewer-error'),
        logContentError: $('#log-content-error'),
        btnCloseLogError: $('#btn-close-log-error'),

        // Toast
        toastContainer: $('#toast-container'),
    };
}

// =========================================================================
// Helpers
// =========================================================================

function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    DOM.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

async function apiGet(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
}

async function apiPost(url, body) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
}

// =========================================================================
// State switching
// =========================================================================

function showState(state) {
    STATE.status = state;
    DOM.welcomeState.style.display = state === 'idle' ? '' : 'none';
    DOM.processingState.style.display = state === 'processing' ? '' : 'none';
    DOM.resultsState.style.display = state === 'done' ? '' : 'none';
    DOM.errorState.style.display = state === 'error' ? '' : 'none';
}

// =========================================================================
// Health check & GPU badge
// =========================================================================

async function checkHealth() {
    try {
        const data = await apiGet('/api/health');
        DOM.serverStatus.className = 'status-dot online';

        if (data.gpu && data.gpu.available) {
            DOM.gpuBadge.className = 'badge gpu-ok';
            DOM.gpuBadge.textContent = `🖥 ${data.gpu.device || 'GPU'} (${data.gpu.memory_used_gb || 0}GB/${data.gpu.memory_total_gb || '?'}GB)`;
        } else {
            DOM.gpuBadge.className = 'badge gpu-no';
            DOM.gpuBadge.textContent = '⚠ CPU only';
        }

        if (data.disk && !data.disk.error) {
            const d = data.disk;
            const diskEl = $('#disk-badge');
            if (d.warning) {
                diskEl.className = 'badge gpu-no';
                diskEl.textContent = `💾 ⚠ ${d.uploads_size_gb}GB / ${d.max_uploads_gb}GB`;
                diskEl.title = `Uploads directory is full. Free: ${d.disk_free_gb}GB. Old tasks auto-clean after 24h.`;
            } else {
                diskEl.className = 'badge';
                diskEl.textContent = `💾 ${d.uploads_size_gb}GB used`;
                diskEl.title = `Free disk: ${d.disk_free_gb}GB of ${d.disk_total_gb}GB`;
            }
        }

        if (data.task_running) {
            DOM.serverStatus.className = 'status-dot busy';
        }
    } catch (err) {
        DOM.serverStatus.className = 'status-dot error';
        DOM.gpuBadge.className = 'badge gpu-no';
        DOM.gpuBadge.textContent = '⚠ Server error';
    }
}

// =========================================================================
// Load models
// =========================================================================

async function loadModels() {
    try {
        const data = await apiGet('/api/models');
        const select = DOM.modelSelect;
        select.innerHTML = '';

        if (data.models.length === 0) {
            select.innerHTML = '<option value="">未找到模型</option>';
            return;
        }

        data.models.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = `${model.name} (${model.size_mb} MB)`;
            if (model.path === data.default) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    } catch (err) {
        DOM.modelSelect.innerHTML = '<option value="">加载失败</option>';
        showToast('加载模型列表失败: ' + err.message);
    }
}

// =========================================================================
// Load parameter defaults
// =========================================================================

let DEFAULT_PARAMS = {};

async function loadDefaults() {
    try {
        const data = await apiGet('/api/config/defaults');
        DEFAULT_PARAMS = data;
        setRangeVal('param-infer_step', data.timbre_conversion.infer_step);
        setRangeVal('param-t_start', data.timbre_conversion.t_start);
        setRangeVal('param-key', data.timbre_conversion.key);
        setSelectVal('param-method', data.timbre_conversion.method);
        setSelectVal('param-pitch_extractor', data.timbre_conversion.pitch_extractor);
        setRangeVal('param-vocal_gain', data.mixing.vocal_gain);
        setRangeVal('param-instrumental_gain', data.mixing.instrumental_gain);
        setRangeVal('param-reverb_gain', data.mixing.reverb_gain);
        setRangeVal('param-formant_shift', data.timbre_conversion.formant_shift);
        setRangeVal('param-segment_batch_size', data.timbre_conversion.segment_batch_size);
        setRangeVal('param-chunk_batch_harmony', data.separation.chunk_batch_harmony);
        setRangeVal('param-chunk_batch_reverb', data.separation.chunk_batch_reverb);
        setRangeVal('param-f0_min', data.timbre_conversion.f0_min);
        setRangeVal('param-f0_max', data.timbre_conversion.f0_max);
        setRangeVal('param-threshold', data.timbre_conversion.threshold);
        setCheckbox('param-use_compile', true);
        setCheckbox('param-normalize_output', data.mixing.normalize_output);
    } catch (err) {
        showToast('加载默认参数失败: ' + err.message);
    }
}

function setRangeVal(id, value) {
    const el = $(`#${id}`);
    if (el) { el.value = value; updateRangeLabel(id, value); }
}

function setSelectVal(id, value) {
    const el = $(`#${id}`);
    if (el) el.value = value;
}

function setCheckbox(id, checked) {
    const el = $(`#${id}`);
    if (el) el.checked = checked;
}

function updateRangeLabel(id, value) {
    const label = $(`#val-${id.replace('param-', '')}`);
    if (label) label.textContent = value;
}

// =========================================================================
// Collect current parameters from form
// =========================================================================

function collectParams() {
    const get = (id, coerce) => {
        const el = $(`#${id}`);
        if (!el) return undefined;
        return coerce(el.value);
    };
    const getCheck = (id) => $(`#${id}`)?.checked ?? undefined;

    return {
        model_ckpt: DOM.modelSelect.value,
        infer_step: get('param-infer_step', parseInt),
        t_start: get('param-t_start', parseFloat),
        method: get('param-method', String),
        pitch_extractor: get('param-pitch_extractor', String),
        key: get('param-key', parseFloat),
        formant_shift: get('param-formant_shift', parseFloat),
        vocal_register_shift: 0,
        vocal_gain: get('param-vocal_gain', parseFloat),
        instrumental_gain: get('param-instrumental_gain', parseFloat),
        reverb_gain: get('param-reverb_gain', parseFloat),
        normalize_output: getCheck('param-normalize_output'),
        use_compile: getCheck('param-use_compile'),
        segment_batch_size: get('param-segment_batch_size', parseInt),
        chunk_batch_harmony: get('param-chunk_batch_harmony', parseInt),
        chunk_batch_reverb: get('param-chunk_batch_reverb', parseInt),
        f0_min: get('param-f0_min', parseFloat),
        f0_max: get('param-f0_max', parseFloat),
        threshold: get('param-threshold', parseFloat),
    };
}

// =========================================================================
// File upload with drag & drop (multi-file support)
// =========================================================================

const ALLOWED_EXTS = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'aiff',
                      'mp4', 'mkv', 'avi', 'mov', 'flv', 'webm', 'wmv', 'm4v'];

function initUpload() {
    const zone = DOM.uploadZone;
    const input = DOM.fileInput;

    DOM.btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        input.click();
    });

    zone.addEventListener('click', () => input.click());

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            queueFiles(input.files);
            input.value = '';
        }
    });

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            queueFiles(e.dataTransfer.files);
        }
    });

    DOM.btnClearQueue.addEventListener('click', () => {
        if (STATE.status === 'processing') return;
        STATE.files = [];
        _pendingFiles = [];
        renderFileQueue();
        updateStartButton();
    });
}

function queueFiles(fileList) {
    const MAX_BATCH = 50;

    for (const file of fileList) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTS.includes(ext)) {
            showToast(`不支持的文件格式: ${file.name} (.${ext})`, 'warning');
            continue;
        }

        if (STATE.files.length + _pendingFiles.length >= MAX_BATCH) {
            showToast(`最多支持 ${MAX_BATCH} 个文件`, 'warning');
            break;
        }

        const allNames = [...STATE.files.map(f => f.name), ..._pendingFiles.map(f => f.name)];
        if (allNames.includes(file.name)) {
            showToast(`文件已存在: ${file.name}`, 'warning');
            continue;
        }

        _pendingFiles.push(file);
    }

    if (_pendingFiles.length > 0) {
        drainPendingUploads();
    }
}

async function drainPendingUploads() {
    while (_pendingFiles.length > 0) {
        const file = _pendingFiles.shift();

        const entry = {
            uploadId: null,
            name: file.name,
            sizeMb: (file.size / (1024 * 1024)).toFixed(1),
            status: 'uploading',
        };
        STATE.files.push(entry);
        renderFileQueue();
        updateStartButton();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await resp.json();
            entry.uploadId = data.upload_id;
            entry.name = data.original_name;
            entry.sizeMb = data.size_mb;
            entry.status = 'uploaded';
        } catch (err) {
            entry.status = 'error';
            showToast(`上传失败: ${entry.name} - ${err.message}`);
        }
        renderFileQueue();
        updateStartButton();
    }
}

function renderFileQueue() {
    if (STATE.files.length === 0) {
        DOM.fileQueue.style.display = 'none';
        DOM.fileQueueList.innerHTML = '';
        return;
    }

    DOM.fileQueue.style.display = 'block';
    DOM.queueCount.textContent = STATE.files.length;

    DOM.fileQueueList.innerHTML = STATE.files.map((f, i) => {
        let statusChip = '';
        let removeBtn = '';
        const canRemove = STATE.status !== 'processing';

        switch (f.status) {
            case 'uploading':
                statusChip = '<span class="status-chip chip-uploading">⏳ 上传中</span>';
                break;
            case 'uploaded':
                statusChip = '<span class="status-chip chip-uploaded">✅ 已上传</span>';
                break;
            case 'error':
                statusChip = '<span class="status-chip chip-error">❌ 失败</span>';
                break;
            default:
                statusChip = `<span class="status-chip">${f.status}</span>`;
        }

        if (canRemove) {
            removeBtn = `<button class="btn-icon" onclick="removeFile(${i})" title="移除">✕</button>`;
        }

        return `
            <div class="file-queue-item">
                <span class="file-icon-small">🎵</span>
                <span class="file-name" title="${f.name}">${f.name}</span>
                <span class="file-size-small">${f.sizeMb} MB</span>
                ${statusChip}
                ${removeBtn}
            </div>
        `;
    }).join('');
}

function removeFile(index) {
    if (STATE.status === 'processing') return;
    STATE.files.splice(index, 1);
    renderFileQueue();
    updateStartButton();
}

function updateStartButton() {
    const hasModel = DOM.modelSelect.value && DOM.modelSelect.value !== '';
    const hasFiles = STATE.files.length > 0 && STATE.files.every(f => f.status === 'uploaded');
    const notRunning = STATE.status !== 'processing';
    const canStart = hasModel && hasFiles && notRunning;

    DOM.btnStart.disabled = !canStart;

    if (STATE.status === 'processing') {
        DOM.btnStart.textContent = '⏳ 处理中...';
    } else if (STATE.files.length > 1) {
        DOM.btnStart.textContent = `🚀 开始批量处理 (${STATE.files.length} 个文件)`;
    } else {
        DOM.btnStart.textContent = '🚀 开始处理';
    }
}

// =========================================================================
// Start processing (single or batch)
// =========================================================================

async function startProcessing() {
    if (STATE.status === 'processing') return;

    const uploadedFiles = STATE.files.filter(f => f.status === 'uploaded');
    if (uploadedFiles.length === 0) {
        showToast('请先上传文件');
        return;
    }
    if (!DOM.modelSelect.value) {
        showToast('请选择音色模型');
        return;
    }

    const params = collectParams();
    if (!params.model_ckpt) {
        showToast('请选择音色模型');
        return;
    }

    if (uploadedFiles.length === 1) {
        STATE.mode = 'single';
        STATE.uploadId = uploadedFiles[0].uploadId;
        await startSingleProcessing(uploadedFiles[0], params);
    } else {
        STATE.mode = 'batch';
        await startBatchProcessing(uploadedFiles, params);
    }
}

async function startSingleProcessing(fileEntry, params) {
    try {
        const keepIntermediates = document.getElementById('param-keep_intermediates')?.checked || false;
        const steps = {
            harmony: document.getElementById('step-harmony')?.checked ?? true,
            reverb: document.getElementById('step-reverb')?.checked ?? true,
            timbre: document.getElementById('step-timbre')?.checked ?? true,
        };
        const data = await apiPost('/api/tasks', {
            upload_id: fileEntry.uploadId,
            params: params,
            keep_intermediates: keepIntermediates,
            steps: steps,
        });

        STATE.taskId = data.task_id;
        STATE._pollTimer = null;
        STATE._activeSteps = steps;
        showState('processing');
        DOM.btnStart.disabled = true;
        DOM.btnStart.textContent = '⏳ 处理中...';
        DOM.serverStatus.className = 'status-dot busy';

        DOM.stageIndicators.style.display = '';
        DOM.batchProgressSection.style.display = 'none';

        DOM.progressFill.style.width = '0%';
        DOM.progressPercent.textContent = '0%';
        DOM.progressStage.textContent = '';
        DOM.progressMessage.textContent = '正在初始化...';
        setupStageDots(steps);

        connectWebSocket(data.task_id);
    } catch (err) {
        showToast('启动任务失败: ' + err.message);
        showState('error');
        DOM.errorMessage.textContent = err.message;
        updateStartButton();
    }
}

async function startBatchProcessing(uploadedFiles, params) {
    try {
        const keepIntermediates = document.getElementById('param-keep_intermediates')?.checked || false;
        const steps = {
            harmony: document.getElementById('step-harmony')?.checked ?? true,
            reverb: document.getElementById('step-reverb')?.checked ?? true,
            timbre: document.getElementById('step-timbre')?.checked ?? true,
        };

        const data = await apiPost('/api/batch', {
            upload_ids: uploadedFiles.map(f => f.uploadId),
            params: params,
            keep_intermediates: keepIntermediates,
            steps: steps,
        });

        STATE.batchId = data.batch_id;
        STATE._batchPollTimer = null;
        STATE._activeSteps = steps;
        showState('processing');
        DOM.btnStart.disabled = true;
        DOM.btnStart.textContent = '⏳ 处理中...';
        DOM.serverStatus.className = 'status-dot busy';

        DOM.stageIndicators.style.display = 'none';
        DOM.batchProgressSection.style.display = '';

        DOM.progressFill.style.width = '0%';
        DOM.progressPercent.textContent = '0%';
        DOM.progressStage.textContent = '';
        DOM.progressMessage.textContent = '正在初始化批量处理...';

        connectBatchWebSocket(data.batch_id);
    } catch (err) {
        showToast('启动批量任务失败: ' + err.message);
        showState('error');
        DOM.errorMessage.textContent = err.message;
        updateStartButton();
    }
}

// =========================================================================
// WebSocket progress (single task)
// =========================================================================

function connectWebSocket(taskId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/tasks/${taskId}`;
    let completed = false;

    const pollTimer = setInterval(async () => {
        if (completed) { clearInterval(pollTimer); return; }
        try {
            const resp = await fetch(`/api/tasks/${taskId}`);
            if (!resp.ok) return;
            const task = await resp.json();
            handleProgressMessage({
                type: 'progress',
                task_id: task.task_id,
                status: task.status,
                progress: task.progress,
                stage: task.stage,
                message: task.message,
                output_files: task.output_files,
                error: task.error,
            });
            if (task.status === 'completed' || task.status === 'failed') {
                completed = true;
                clearInterval(pollTimer);
                if (STATE.ws) { STATE.ws.close(); STATE.ws = null; }
            }
        } catch (e) { /* ignore */ }
    }, 2000);
    STATE._pollTimer = pollTimer;

    const ws = new WebSocket(wsUrl);
    STATE.ws = ws;

    ws.onopen = () => console.log('WebSocket connected for task', taskId);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleProgressMessage(data);
            if (data.status === 'completed' || data.status === 'failed') {
                completed = true;
                if (pollTimer) clearInterval(pollTimer);
            }
        } catch (err) {
            console.error('Invalid WS message:', err);
        }
    };

    ws.onerror = () => console.warn('WebSocket error -- REST polling fallback active');
    ws.onclose = () => { console.log('WebSocket closed for task', taskId); STATE.ws = null; };

    const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        else clearInterval(pingInterval);
    }, 10000);
}

// =========================================================================
// WebSocket progress (batch)
// =========================================================================

function connectBatchWebSocket(batchId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/batches/${batchId}`;
    let completed = false;

    const pollTimer = setInterval(async () => {
        if (completed) { clearInterval(pollTimer); return; }
        try {
            const resp = await fetch(`/api/batches/${batchId}`);
            if (!resp.ok) return;
            const data = await resp.json();
            handleBatchMessage(data);
            if (data.status === 'completed' || data.status === 'failed') {
                completed = true;
                clearInterval(pollTimer);
                if (STATE.batchWs) { STATE.batchWs.close(); STATE.batchWs = null; }
            }
        } catch (e) { /* ignore */ }
    }, 2000);
    STATE._batchPollTimer = pollTimer;

    const ws = new WebSocket(wsUrl);
    STATE.batchWs = ws;

    ws.onopen = () => console.log('Batch WebSocket connected for', batchId);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleBatchMessage(data);
            if (data.status === 'completed' || data.status === 'failed') {
                completed = true;
                if (pollTimer) clearInterval(pollTimer);
            }
        } catch (err) {
            console.error('Invalid batch WS message:', err);
        }
    };

    ws.onerror = () => console.warn('Batch WebSocket error -- REST polling fallback active');
    ws.onclose = () => { console.log('Batch WebSocket closed for', batchId); STATE.batchWs = null; };

    const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        else clearInterval(pingInterval);
    }, 10000);
}

// =========================================================================
// Progress message handlers
// =========================================================================

function handleProgressMessage(data) {
    if (data.type !== 'progress') return;

    const { status, progress, stage, message, output_files, error } = data;
    const isLoading = /加载|loading/i.test(message || '');
    const pct = Math.round(progress);

    if (isLoading) {
        DOM.progressFill.style.width = '100%';
        DOM.progressFill.classList.add('indeterminate');
        DOM.progressPercent.textContent = '...';
    } else {
        DOM.progressFill.classList.remove('indeterminate');
        DOM.progressFill.style.width = `${pct}%`;
        DOM.progressPercent.textContent = `${pct}%`;
    }
    DOM.progressStage.textContent = stage ? formatStageName(stage) : '';
    DOM.progressMessage.textContent = message || '';
    DOM.progressMessage.className = 'progress-message' + (isLoading ? ' loading' : '');

    updateStageIndicators(stage, status);

    if (status === 'completed') {
        onTaskComplete(data);
    } else if (status === 'failed' || error) {
        onTaskError(data);
    }
}

function handleBatchMessage(data) {
    const { status, progress, current_index, message, error, files } = data;
    const isLoading = /加载|loading/i.test(message || '');
    const pct = Math.round(progress);

    if (isLoading) {
        DOM.progressFill.style.width = '100%';
        DOM.progressFill.classList.add('indeterminate');
        DOM.progressPercent.textContent = '...';
    } else {
        DOM.progressFill.classList.remove('indeterminate');
        DOM.progressFill.style.width = `${pct}%`;
        DOM.progressPercent.textContent = `${pct}%`;
    }

    if (files && files.length > 0 && current_index < files.length) {
        const current = files[current_index];
        DOM.batchCurrentFile.textContent = `📄 文件 ${current_index + 1}/${files.length}: ${current.input_filename}`;
        DOM.progressStage.textContent = current.stage ? formatStageName(current.stage) : '';
        DOM.progressMessage.textContent = current.message || message || '';
        DOM.progressMessage.className = 'progress-message' + (isLoading ? ' loading' : '');
    } else {
        DOM.progressMessage.textContent = message || '';
    }

    // Render per-file progress rows
    if (files && DOM.batchFileList) {
        DOM.batchFileList.innerHTML = files.map((f, i) => {
            let chipClass = '', chipText = '';
            switch (f.status) {
                case 'completed': chipClass = 'chip-done'; chipText = '✅ 完成'; break;
                case 'running': chipClass = 'chip-processing'; chipText = '⏳ 处理中'; break;
                case 'failed': chipClass = 'chip-error'; chipText = '❌ 失败'; break;
                default: chipText = '等待中';
            }

            const isCurrent = i === current_index && f.status === 'running';
            const rowClass = isCurrent ? 'batch-file-row processing' : 'batch-file-row';

            let miniBar = '';
            if (f.status === 'running') {
                miniBar = `<div class="mini-progress-bg"><div class="mini-progress-fill" style="width:${Math.round(f.progress)}%"></div></div>`;
            } else if (f.status === 'completed') {
                miniBar = `<div class="mini-progress-bg"><div class="mini-progress-fill" style="width:100%"></div></div>`;
            }

            let errorText = '';
            if (f.status === 'failed' && f.error) {
                errorText = `<div class="batch-file-error">${f.error}</div>`;
            }

            return `
                <div class="${rowClass}">
                    <div class="batch-file-row-header">
                        <span class="file-name">🎵 ${f.input_filename}</span>
                        <span class="status-chip ${chipClass}">${chipText}</span>
                    </div>
                    ${miniBar}
                    ${errorText}
                </div>
            `;
        }).join('');
    }

    if (status === 'completed') {
        onBatchComplete(data);
    } else if (status === 'failed') {
        onBatchError(data);
    }
}

function formatStageName(stage) {
    const names = {
        'extract_audio': '音频提取',
        'harmony_separation': '和声分离',
        'reverb_separation': '混响分离',
        'linked_separation': '联合分离',
        'timbre_conversion': '音色替换',
        'mixing': '混音输出',
        'error': '错误',
        'starting': '初始化',
    };
    return names[stage] || stage;
}

function setupStageDots(steps) {
    const visibility = {
        'extract_audio': true,
        'harmony_separation': steps.harmony !== false,
        'reverb_separation': steps.reverb !== false,
        'timbre_conversion': steps.timbre !== false,
        'mixing': true,
    };
    $$('.stage-dot').forEach(dot => {
        const ds = dot.dataset.stage;
        dot.style.display = visibility[ds] !== false ? '' : 'none';
        dot.className = 'stage-dot';
    });
}

function updateStageIndicators(currentStage, status) {
    const stages = ['extract_audio', 'harmony_separation', 'reverb_separation', 'timbre_conversion', 'mixing'];
    const linkedCompletes = new Set(['harmony_separation', 'reverb_separation']);
    let passed = true;

    $$('.stage-dot').forEach(dot => {
        if (dot.style.display === 'none') return;
        const ds = dot.dataset.stage;
        if (currentStage === 'linked_separation' && linkedCompletes.has(ds) && passed) {
            dot.className = 'stage-dot active';
            passed = false;
        } else if (ds === currentStage) {
            dot.className = 'stage-dot active';
            passed = false;
        } else if (passed && stages.includes(ds)) {
            dot.className = 'stage-dot completed';
        }
    });
}

function resetStageIndicators() {
    $$('.stage-dot').forEach(dot => {
        dot.className = 'stage-dot';
        dot.style.display = '';
    });
}

// =========================================================================
// Single task completion
// =========================================================================

function onTaskComplete(data) {
    STATE.status = 'done';
    DOM.serverStatus.className = 'status-dot online';
    updateStartButton();

    if (data.output_files && data.output_files.cover) {
        setupAudioPlayer(data.task_id);
        setupDownload(data.task_id);
    }
    if (data.output_files) {
        setupOutputFiles(data.task_id, data.output_files);
    }

    setupLogButtons(data.task_id);
    DOM.singleResult.style.display = '';
    DOM.batchResults.style.display = 'none';
    showState('done');
}

function onTaskError(data) {
    STATE.status = 'error';
    DOM.serverStatus.className = 'status-dot error';
    updateStartButton();
    showState('error');
    DOM.errorMessage.textContent = data.error || data.message || '未知错误';
    setupLogButtons(data.task_id);
}

// =========================================================================
// Batch completion
// =========================================================================

function onBatchComplete(data) {
    STATE.status = 'done';
    DOM.serverStatus.className = 'status-dot online';
    updateStartButton();

    DOM.singleResult.style.display = 'none';
    DOM.batchResults.style.display = '';

    const files = data.files || [];
    const succeeded = files.filter(f => f.status === 'completed').length;
    const failed = files.filter(f => f.status === 'failed').length;
    DOM.batchSummary.textContent = `✅ ${succeeded} 成功 · ❌ ${failed} 失败 · 📁 共 ${files.length} 个文件`;

    DOM.batchResultsList.innerHTML = files.map(f => {
        if (f.status === 'completed') {
            const coverFile = f.output_files?.cover || '';
            const previewUrl = `/api/tasks/${f.file_id}/preview`;
            const downloadUrl = coverFile
                ? `/api/tasks/${f.file_id}/output/${encodeURIComponent(coverFile)}`
                : previewUrl;
            const coverName = coverFile ? coverFile.split('/').pop() : 'cover.wav';

            // Build intermediate files list (everything except cover)
            const intermediates = f.output_files
                ? Object.entries(f.output_files).filter(([k]) => k !== 'cover')
                : [];
            let intermediatesHtml = '';
            if (intermediates.length > 0) {
                const labelMap = {
                    'converted_vocals': '音色转换人声',
                    'vocals': '原始人声',
                    'instrumental': '伴奏',
                    'noreverb': '干声(无混响)',
                    'reverb': '混响',
                };
                const rows = intermediates.map(([key, relPath]) => {
                    const dlUrl = `/api/tasks/${f.file_id}/output/${encodeURIComponent(relPath)}`;
                    const fname = relPath.split('/').pop();
                    const label = labelMap[key] || key;
                    return `<div class="output-link">
                        <span>📄 ${label}: ${fname}</span>
                        <a href="${dlUrl}" download>⬇ 下载</a>
                    </div>`;
                }).join('');
                intermediatesHtml = `
                    <details class="intermediates-toggle">
                        <summary>📂 中间文件 (${intermediates.length})</summary>
                        <div class="intermediates-list">${rows}</div>
                    </details>`;
            }

            return `
                <div class="result-card">
                    <div class="result-card-header">
                        <span class="file-name">🎵 ${f.input_filename}</span>
                        <span class="status-chip chip-done">✅ 完成</span>
                    </div>
                    <audio controls preload="none" src="${previewUrl}" class="result-audio"></audio>
                    <div class="result-card-actions">
                        <a href="${downloadUrl}" download="${coverName}" class="btn btn-primary btn-sm">💾 下载翻唱</a>
                        <button class="btn btn-secondary btn-sm" onclick="fetchAndShowLog('${f.file_id}', DOM.logViewerResults, DOM.logContentResults)">📋 查看日志</button>
                    </div>
                    ${intermediatesHtml}
                </div>
            `;
        } else {
            return `
                <div class="result-card result-card-error-card">
                    <div class="result-card-header">
                        <span class="file-name">🎵 ${f.input_filename}</span>
                        <span class="status-chip chip-error">❌ 失败</span>
                    </div>
                    <div class="batch-file-error">${f.error || '未知错误'}</div>
                    <div class="result-card-actions">
                        <button class="btn btn-secondary btn-sm" onclick="fetchAndShowLog('${f.file_id}', DOM.logViewerResults, DOM.logContentResults)">📋 查看日志</button>
                    </div>
                </div>
            `;
        }
    }).join('');

    DOM.logViewerResults.style.display = 'none';
    showState('done');
}

function onBatchError(data) {
    STATE.status = 'error';
    DOM.serverStatus.className = 'status-dot error';
    updateStartButton();

    if (data.files && data.files.length > 0) {
        onBatchComplete(data);
        showToast('批量处理失败: ' + (data.error || '未知错误'), 'error');
    } else {
        showState('error');
        DOM.errorMessage.textContent = data.error || data.message || '批量处理失败';
    }
}

// =========================================================================
// Audio player (single mode)
// =========================================================================

function setupAudioPlayer(taskId) {
    const audio = DOM.audioPlayer;
    audio.src = `/api/tasks/${taskId}/preview`;
    audio.crossOrigin = 'anonymous';
    audio.load();

    const newBtn = DOM.btnPlay.cloneNode(true);
    DOM.btnPlay.parentNode.replaceChild(newBtn, DOM.btnPlay);
    DOM.btnPlay = newBtn;

    DOM.btnPlay.addEventListener('click', () => {
        if (audio.paused) {
            audio.play();
            DOM.btnPlay.textContent = '⏸';
            DOM.btnPlay.classList.add('playing');
            STATE.playing = true;
        } else {
            audio.pause();
            DOM.btnPlay.textContent = '▶';
            DOM.btnPlay.classList.remove('playing');
            STATE.playing = false;
        }
    });

    const canvas = document.getElementById('waveform-canvas');
    let waveformData = null;
    let waveformDrawn = false;

    function drawWaveform() {
        if (!canvas || waveformDrawn || !audio.duration) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
        const H = canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);

        if (waveformData) {
            ctx.clearRect(0, 0, W, H);
            const mid = H / 2;
            const barW = Math.max(1, W / waveformData.length);
            const playedColor = '#8b7cf0';
            const pendingColor = '#4a4a6a';
            const progress = audio.duration ? (audio.currentTime / audio.duration) : 0;
            for (let i = 0; i < waveformData.length; i++) {
                const h = waveformData[i] * mid * 0.85;
                const x = i * barW;
                ctx.fillStyle = (i / waveformData.length <= progress) ? playedColor : pendingColor;
                ctx.fillRect(x, mid - h, Math.max(1, barW - 1), Math.max(1, h * 2));
            }
            waveformDrawn = true;
        } else {
            ctx.fillStyle = '#6a6a8a';
            ctx.font = `${Math.min(14, H * 0.4)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillText('加载波形中...', W / 2, H / 2);
        }
    }

    audio.onloadedmetadata = () => {
        DOM.audioDuration.textContent = formatTime(audio.duration);
        if (!waveformData && window.AudioContext) {
            try {
                const actx = new AudioContext();
                fetch(audio.src)
                    .then(r => r.arrayBuffer())
                    .then(buf => actx.decodeAudioData(buf))
                    .then(decoded => {
                        const ch = decoded.getChannelData(0);
                        const peaks = 200;
                        const step = Math.floor(ch.length / peaks);
                        const data = [];
                        for (let i = 0; i < peaks; i++) {
                            let max = 0;
                            const start = i * step;
                            for (let j = 0; j < step; j++) {
                                max = Math.max(max, Math.abs(ch[start + j] || 0));
                            }
                            data.push(max);
                        }
                        waveformData = data;
                        drawWaveform();
                    })
                    .catch(() => {});
            } catch (e) {}
        }
        drawWaveform();
    };

    audio.ontimeupdate = () => {
        DOM.audioCurrent.textContent = formatTime(audio.currentTime);
        const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
        DOM.audioSeek.value = pct;
        if (waveformData) { waveformDrawn = false; drawWaveform(); }
    };
    audio.onended = () => {
        DOM.btnPlay.textContent = '▶';
        DOM.btnPlay.classList.remove('playing');
        STATE.playing = false;
        waveformDrawn = false;
        drawWaveform();
    };
    audio.onplay = () => { if (!waveformDrawn) drawWaveform(); };

    window.addEventListener('resize', () => { waveformDrawn = false; drawWaveform(); });
    drawWaveform();

    DOM.audioSeek.oninput = () => {
        if (audio.duration) {
            audio.currentTime = (DOM.audioSeek.value / 100) * audio.duration;
            waveformDrawn = false;
            drawWaveform();
        }
    };

    DOM.audioVolume.oninput = () => { audio.volume = DOM.audioVolume.value / 100; };
    audio.volume = DOM.audioVolume.value / 100;
}

function setupDownload(taskId) {
    const newBtn = DOM.btnDownload.cloneNode(true);
    DOM.btnDownload.parentNode.replaceChild(newBtn, DOM.btnDownload);
    DOM.btnDownload = newBtn;

    DOM.btnDownload.addEventListener('click', () => {
        const a = document.createElement('a');
        a.href = `/api/tasks/${taskId}/preview`;
        a.download = `cover_${taskId}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
}

function setupOutputFiles(taskId, outputFiles) {
    DOM.outputFilesList.style.display = 'block';
    DOM.outputLinks.innerHTML = '';

    Object.entries(outputFiles).forEach(([name, relPath]) => {
        const div = document.createElement('div');
        div.className = 'output-link';
        const fileName = relPath.split('/').pop();
        div.innerHTML = `
            <span>📄 ${name}: ${fileName}</span>
            <a href="/api/tasks/${taskId}/output/${encodeURIComponent(relPath)}" download>⬇ 下载</a>
        `;
        DOM.outputLinks.appendChild(div);
    });
}

// =========================================================================
// Task log viewer
// =========================================================================

async function fetchAndShowLog(taskId, containerEl, contentEl) {
    contentEl.textContent = '加载中...';
    containerEl.style.display = 'block';

    try {
        const data = await apiGet(`/api/tasks/${taskId}/log`);
        const logText = data.log || '(暂无日志记录)';
        const highlighted = logText
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\[ERROR\]/g, '<span class="log-ERROR">[ERROR]</span>')
            .replace(/\[WARNING\]/g, '<span class="log-WARNING">[WARNING]</span>')
            .replace(/\[INFO\]/g, '<span class="log-INFO">[INFO]</span>')
            .replace(/\[DEBUG\]/g, '<span class="log-DEBUG">[DEBUG]</span>');
        contentEl.innerHTML = highlighted;
        contentEl.scrollTop = contentEl.scrollHeight;

        if (data.result && data.result.elapsed_total_s) {
            const existing = containerEl.querySelector('.log-summary');
            if (!existing) {
                const summary = document.createElement('div');
                summary.className = 'log-summary';
                summary.innerHTML = `<strong>总耗时:</strong> ${data.result.elapsed_total_s.toFixed(1)}s | <strong>日志行数:</strong> ${data.log_lines}`;
                containerEl.querySelector('.log-viewer-header').appendChild(summary);
            }
        }
    } catch (err) {
        contentEl.textContent = '无法加载日志: ' + err.message;
    }
}

function setupLogButtons(taskId) {
    DOM.btnShowLog.addEventListener('click', () => {
        fetchAndShowLog(taskId, DOM.logViewerResults, DOM.logContentResults);
    });
    DOM.btnCloseLogResults.addEventListener('click', () => {
        DOM.logViewerResults.style.display = 'none';
    });

    fetchAndShowLog(taskId, DOM.logViewerError, DOM.logContentError);
    DOM.btnShowErrorLog.addEventListener('click', () => {
        fetchAndShowLog(taskId, DOM.logViewerError, DOM.logContentError);
    });
    DOM.btnCloseLogError.addEventListener('click', () => {
        DOM.logViewerError.style.display = 'none';
    });
}

// =========================================================================
// Reset
// =========================================================================

function resetAll() {
    STATE.taskId = null;
    STATE.uploadId = null;
    STATE.playing = false;
    STATE.mode = null;
    if (STATE.ws) { STATE.ws.close(); STATE.ws = null; }
    if (STATE._pollTimer) { clearInterval(STATE._pollTimer); STATE._pollTimer = null; }

    STATE.batchId = null;
    if (STATE.batchWs) { STATE.batchWs.close(); STATE.batchWs = null; }
    if (STATE._batchPollTimer) { clearInterval(STATE._batchPollTimer); STATE._batchPollTimer = null; }

    if (DOM.audioPlayer) {
        DOM.audioPlayer.pause();
        DOM.audioPlayer.src = '';
    }

    DOM.progressFill.style.width = '0%';
    DOM.progressFill.classList.remove('indeterminate');
    DOM.progressPercent.textContent = '0%';
    DOM.progressStage.textContent = '';
    DOM.progressMessage.textContent = '';
    DOM.batchCurrentFile.textContent = '';
    resetStageIndicators();

    DOM.stageIndicators.style.display = '';
    DOM.batchProgressSection.style.display = 'none';
    DOM.singleResult.style.display = '';
    DOM.batchResults.style.display = 'none';
    DOM.batchSummary.textContent = '';
    DOM.batchResultsList.innerHTML = '';

    showState('idle');
    updateStartButton();

    DOM.btnPlay.textContent = '▶';
    DOM.btnPlay.classList.remove('playing');
    DOM.audioCurrent.textContent = '0:00';
    DOM.audioDuration.textContent = '0:00';
    DOM.audioSeek.value = 0;
    DOM.outputFilesList.style.display = 'none';
    DOM.outputLinks.innerHTML = '';
    if (DOM.logViewerResults) DOM.logViewerResults.style.display = 'none';
    if (DOM.logViewerError) DOM.logViewerError.style.display = 'none';

    STATE.files = [];
    _pendingFiles = [];
    renderFileQueue();
    checkHealth();
}

// =========================================================================
// Event bindings
// =========================================================================

function bindEvents() {
    DOM.btnStart.addEventListener('click', startProcessing);
    DOM.btnNewTask.addEventListener('click', resetAll);
    DOM.btnNewTaskBatch.addEventListener('click', resetAll);
    DOM.btnRetry.addEventListener('click', resetAll);
    DOM.modelSelect.addEventListener('change', updateStartButton);

    $$('.form-range').forEach(range => {
        range.addEventListener('input', () => {
            const id = range.id.replace('param-', '');
            const label = $(`#val-${id}`);
            if (label) label.textContent = range.value;
        });
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' ||
            e.target.tagName === 'TEXTAREA') return;
        if (e.code === 'Space' && STATE.status === 'done') {
            e.preventDefault();
            DOM.btnPlay.click();
        }
    });
}

// =========================================================================
// Initialization
// =========================================================================

async function init() {
    cacheDom();
    bindEvents();
    initUpload();

    await Promise.all([
        checkHealth(),
        loadModels(),
        loadDefaults(),
    ]);

    updateStartButton();
    setInterval(checkHealth, 30000);
    console.log('Song Cover Pipeline Web UI initialized (batch support)');
}

// Boot
document.addEventListener('DOMContentLoaded', init);
