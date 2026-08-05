/**
 * Song Cover Pipeline — Web UI Application
 * Vanilla JS, no framework dependencies
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
    _activeSteps: null,  // { harmony: bool, reverb: bool, timbre: bool }
    _pollTimer: null,
};

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
        uploadInfo: $('#upload-info'),
        uploadFilename: $('#upload-filename'),
        uploadSize: $('#upload-size'),
        fileInput: $('#file-input'),
        btnBrowse: $('#btn-browse'),
        btnClearFile: $('#btn-clear-file'),
        btnStart: $('#btn-start'),

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

        // Results
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

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
            DOM.gpuBadge.textContent = '⚠️ CPU only';
        }

        // Disk usage
        if (data.disk && !data.disk.error) {
            const d = data.disk;
            const diskEl = $('#disk-badge');
            if (d.warning) {
                diskEl.className = 'badge gpu-no';
                diskEl.textContent = `💾 ⚠️ ${d.uploads_size_gb}GB / ${d.max_uploads_gb}GB`;
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
        DOM.gpuBadge.textContent = '⚠️ Server error';
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

        // Apply defaults to form
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
        // MSST torch.compile defaults to ON (matches project config)
        setCheckbox('param-use_compile', true);
        setCheckbox('param-normalize_output', data.mixing.normalize_output);
    } catch (err) {
        showToast('加载默认参数失败: ' + err.message);
    }
}

function setRangeVal(id, value) {
    const el = $(`#${id}`);
    if (el) {
        el.value = value;
        updateRangeLabel(id, value);
    }
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
    const labelId = `val-${id.replace('param-', '')}`;
    const label = $(`#${labelId}`);
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
// File upload with drag & drop
// =========================================================================

function initUpload() {
    const zone = DOM.uploadZone;
    const input = DOM.fileInput;

    // Browse button
    DOM.btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        input.click();
    });

    // Click zone to browse
    zone.addEventListener('click', () => input.click());

    // File selected via browse
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            handleFile(input.files[0]);
        }
    });

    // Drag events
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
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear file
    DOM.btnClearFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });
}

async function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'aiff',
                         'mp4', 'mkv', 'avi', 'mov', 'flv', 'webm', 'wmv', 'm4v'];

    if (!allowedExts.includes(ext)) {
        showToast(`不支持的文件格式: .${ext}`);
        return;
    }

    // Show uploading state
    DOM.uploadZone.querySelector('.upload-icon').textContent = '⏳';
    DOM.uploadZone.querySelector('.upload-text').textContent = '上传中...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await resp.json();
        STATE.uploadId = data.upload_id;

        // Show uploaded state
        DOM.uploadZone.classList.add('has-file');
        DOM.uploadZone.querySelector('.upload-icon').textContent = '🎶';
        DOM.uploadZone.querySelector('.upload-text').textContent = '点击或拖拽替换文件';
        DOM.uploadInfo.style.display = 'flex';
        DOM.uploadFilename.textContent = data.original_name;
        DOM.uploadSize.textContent = `${data.size_mb} MB`;

        updateStartButton();
    } catch (err) {
        showToast('上传失败: ' + err.message);
        DOM.uploadZone.querySelector('.upload-icon').textContent = '🎶';
        DOM.uploadZone.querySelector('.upload-text').textContent = '拖拽音频/视频文件到此处';
    }
}

function clearFile() {
    STATE.uploadId = null;
    DOM.uploadZone.classList.remove('has-file');
    DOM.uploadInfo.style.display = 'none';
    DOM.uploadZone.querySelector('.upload-icon').textContent = '🎶';
    DOM.uploadZone.querySelector('.upload-text').textContent = '拖拽音频/视频文件到此处';
    DOM.fileInput.value = '';
    updateStartButton();
}

function updateStartButton() {
    const hasModel = DOM.modelSelect.value && DOM.modelSelect.value !== '';
    const hasFile = !!STATE.uploadId;
    const notRunning = STATE.status !== 'processing';

    DOM.btnStart.disabled = !(hasModel && hasFile && notRunning);

    if (hasModel && hasFile && notRunning) {
        DOM.btnStart.textContent = '🚀 开始处理';
    } else if (STATE.status === 'processing') {
        DOM.btnStart.textContent = '⏳ 处理中...';
    } else {
        DOM.btnStart.textContent = '🚀 开始处理 (请选择模型并上传文件)';
    }
}

// =========================================================================
// Start processing
// =========================================================================

async function startProcessing() {
    if (STATE.status === 'processing') return;
    if (!STATE.uploadId || !DOM.modelSelect.value) return;

    const params = collectParams();
    if (!params.model_ckpt) {
        showToast('请选择音色模型');
        return;
    }

    try {
        const keepIntermediates = document.getElementById('param-keep_intermediates')?.checked || false;
        // Collect step selections
        const steps = {
            harmony: document.getElementById('step-harmony')?.checked ?? true,
            reverb: document.getElementById('step-reverb')?.checked ?? true,
            timbre: document.getElementById('step-timbre')?.checked ?? true,
        };
        const data = await apiPost('/api/tasks', {
            upload_id: STATE.uploadId,
            params: params,
            keep_intermediates: keepIntermediates,
            steps: steps,
        });

        STATE.taskId = data.task_id;
        STATE._pollTimer = null;  // reset poll timer
        STATE._activeSteps = steps;  // remember which steps are enabled
        showState('processing');
        DOM.btnStart.disabled = true;
        DOM.btnStart.textContent = '⏳ 处理中...';
        DOM.serverStatus.className = 'status-dot busy';

        // Reset progress
        DOM.progressFill.style.width = '0%';
        DOM.progressPercent.textContent = '0%';
        DOM.progressStage.textContent = '';
        DOM.progressMessage.textContent = '正在初始化...';
        setupStageDots(steps);

        // Connect WebSocket for progress (with REST polling fallback)
        connectWebSocket(data.task_id);
    } catch (err) {
        showToast('启动任务失败: ' + err.message);
        showState('error');
        DOM.errorMessage.textContent = err.message;
        updateStartButton();
    }
}

// =========================================================================
// WebSocket progress (with REST polling fallback)
// =========================================================================

function connectWebSocket(taskId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/tasks/${taskId}`;

    let completed = false;

    // ---- Always run REST polling as a safety net ----
    // Even if WebSocket works, polling guarantees we never miss the
    // completion message (WS can drop when browser tab is backgrounded).
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

    // ---- WebSocket as primary (faster updates) ----
    const ws = new WebSocket(wsUrl);
    STATE.ws = ws;

    ws.onopen = () => {
        console.log('WebSocket connected for task', taskId);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleProgressMessage(data);
            // If WS delivers completion, stop polling on next cycle
            if (data.status === 'completed' || data.status === 'failed') {
                completed = true;
                if (pollTimer) clearInterval(pollTimer);
            }
        } catch (err) {
            console.error('Invalid WS message:', err);
        }
    };

    ws.onerror = () => {
        console.warn('WebSocket error — REST polling will keep progress alive');
    };

    ws.onclose = () => {
        console.log('WebSocket closed for task', taskId);
        STATE.ws = null;
        // Polling continues as fallback
    };

    // Keepalive ping
    const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
        } else {
            clearInterval(pingInterval);
        }
    }, 10000);
}

function handleProgressMessage(data) {
    if (data.type === 'progress') {
        const { status, progress, stage, message, output_files, error } = data;

        // Update progress bar
        const isLoading = /加载|loading/i.test(message || '');
        const pct = Math.round(progress);

        if (isLoading) {
            // Indeterminate animation while model loads
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

        // Update stage indicators
        updateStageIndicators(stage, status);

        // Check for completion or error
        // Only explicit 'completed' status triggers completion.
        // 'failed' tasks have a non-empty error field.
        if (status === 'completed') {
            onTaskComplete(data);
        } else if (status === 'failed' || error) {
            onTaskError(data);
        }
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

function resetStageIndicators() {
    $$('.stage-dot').forEach(dot => {
        dot.className = 'stage-dot';
        dot.style.display = '';  // restore visibility
    });
}

/**
 * Show/hide stage dots based on which steps the user selected.
 * Dots for disabled steps are hidden so the progress display only
 * shows the steps that will actually execute.
 */
function setupStageDots(steps) {
    // Always show extract_audio and mixing
    const visibility = {
        'extract_audio': true,
        'harmony_separation': steps.harmony !== false,
        'reverb_separation': steps.reverb !== false,
        'timbre_conversion': steps.timbre !== false,
        'mixing': true,
    };

    $$('.stage-dot').forEach(dot => {
        const ds = dot.dataset.stage;
        const visible = visibility[ds] !== false;
        dot.style.display = visible ? '' : 'none';
        dot.className = 'stage-dot';
    });
}

function updateStageIndicators(currentStage, status) {
    // Only consider visible stage dots
    const stages = ['extract_audio', 'harmony_separation', 'reverb_separation',
                    'timbre_conversion', 'mixing'];
    const linkedCompletes = new Set(['harmony_separation', 'reverb_separation']);
    let passed = true;

    $$('.stage-dot').forEach(dot => {
        if (dot.style.display === 'none') return;  // skip hidden dots

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
// Audio player
// =========================================================================

function setupAudioPlayer(taskId) {
    const audio = DOM.audioPlayer;
    audio.src = `/api/tasks/${taskId}/preview`;
    audio.crossOrigin = 'anonymous';
    audio.load();

    // Clone the play button to remove all existing listeners
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

    // ---- Waveform drawing ----
    const canvas = document.getElementById('waveform-canvas');
    let waveformData = null;
    let waveformDrawn = false;

    function drawWaveform() {
        if (!canvas || waveformDrawn || !audio.duration) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
        const H = canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);

        // If we have pre-decoded data, draw it; otherwise show loading bar
        if (waveformData) {
            ctx.clearRect(0, 0, W, H);
            const mid = H / 2;
            const barW = Math.max(1, W / waveformData.length);
            const playedColor = '#8b7cf0';  // accent-light
            const pendingColor = '#4a4a6a'; // muted
            const progress = audio.duration ? (audio.currentTime / audio.duration) : 0;
            for (let i = 0; i < waveformData.length; i++) {
                const h = waveformData[i] * mid * 0.85;
                const x = i * barW;
                ctx.fillStyle = (i / waveformData.length <= progress) ? playedColor : pendingColor;
                ctx.fillRect(x, mid - h, Math.max(1, barW - 1), Math.max(1, h * 2));
            }
            waveformDrawn = true;
        } else {
            // Placeholder before data loads
            ctx.fillStyle = '#6a6a8a';
            ctx.font = `${Math.min(14, H * 0.4)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillText('加载波形中...', W / 2, H / 2);
        }
    }

    // Try to decode waveform from audio
    audio.onloadedmetadata = () => {
        DOM.audioDuration.textContent = formatTime(audio.duration);
        // Use Web Audio API to extract peaks for waveform
        if (!waveformData && window.AudioContext) {
            try {
                const actx = new AudioContext();
                fetch(audio.src)
                    .then(r => r.arrayBuffer())
                    .then(buf => actx.decodeAudioData(buf))
                    .then(decoded => {
                        const ch = decoded.getChannelData(0);
                        const peaks = 200; // number of bars
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
                    .catch(() => { /* waveform not critical */ });
            } catch (e) { /* ignore */ }
        }
        drawWaveform();
    };

    // Update playhead on timeupdate
    audio.ontimeupdate = () => {
        DOM.audioCurrent.textContent = formatTime(audio.currentTime);
        const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
        DOM.audioSeek.value = pct;
        // Redraw waveform to update played region color
        if (waveformData) { waveformDrawn = false; drawWaveform(); }
    };
    audio.onended = () => {
        DOM.btnPlay.textContent = '▶';
        DOM.btnPlay.classList.remove('playing');
        STATE.playing = false;
        waveformDrawn = false;
        drawWaveform();
    };
    audio.onplay = () => {
        if (!waveformDrawn) drawWaveform();
    };

    // Redraw on resize
    window.addEventListener('resume', () => { waveformDrawn = false; drawWaveform(); });

    // Initial draw
    drawWaveform();

    // Seek
    DOM.audioSeek.oninput = () => {
        if (audio.duration) {
            audio.currentTime = (DOM.audioSeek.value / 100) * audio.duration;
            waveformDrawn = false;
            drawWaveform();
        }
    };

    // Volume
    DOM.audioVolume.oninput = () => {
        audio.volume = DOM.audioVolume.value / 100;
    };
    audio.volume = DOM.audioVolume.value / 100;
}

function setupDownload(taskId) {
    // Clone to remove old listeners
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
        // Basic syntax highlighting: colorize log level markers
        const highlighted = logText
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\[ERROR\]/g, '<span class="log-ERROR">[ERROR]</span>')
            .replace(/\[WARNING\]/g, '<span class="log-WARNING">[WARNING]</span>')
            .replace(/\[INFO\]/g, '<span class="log-INFO">[INFO]</span>')
            .replace(/\[DEBUG\]/g, '<span class="log-DEBUG">[DEBUG]</span>');
        contentEl.innerHTML = highlighted;
        contentEl.scrollTop = contentEl.scrollHeight;

        // Show summary if available
        if (data.result && data.result.elapsed_total_s) {
            const summary = document.createElement('div');
            summary.className = 'log-summary';
            summary.innerHTML = `<strong>总耗时:</strong> ${data.result.elapsed_total_s.toFixed(1)}s | <strong>日志行数:</strong> ${data.log_lines}`;
            if (!containerEl.querySelector('.log-summary')) {
                containerEl.querySelector('.log-viewer-header').appendChild(summary);
            }
        }
    } catch (err) {
        contentEl.textContent = '无法加载日志: ' + err.message;
    }
}

function setupLogButtons(taskId) {
    // Results log
    DOM.btnShowLog.addEventListener('click', () => {
        fetchAndShowLog(taskId, DOM.logViewerResults, DOM.logContentResults);
    });
    DOM.btnCloseLogResults.addEventListener('click', () => {
        DOM.logViewerResults.style.display = 'none';
    });

    // Error log (auto-show)
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
    if (STATE.ws) {
        STATE.ws.close();
        STATE.ws = null;
    }
    if (STATE._pollTimer) {
        clearInterval(STATE._pollTimer);
        STATE._pollTimer = null;
    }
    if (DOM.audioPlayer) {
        DOM.audioPlayer.pause();
        DOM.audioPlayer.src = '';
    }
    DOM.progressFill.style.width = '0%';
    DOM.progressPercent.textContent = '0%';
    DOM.progressStage.textContent = '';
    DOM.progressMessage.textContent = '';
    resetStageIndicators();
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
    clearFile();
    checkHealth();
}

// =========================================================================
// Event bindings
// =========================================================================

function bindEvents() {
    // Start button
    DOM.btnStart.addEventListener('click', startProcessing);

    // New task / Retry
    DOM.btnNewTask.addEventListener('click', resetAll);
    DOM.btnRetry.addEventListener('click', resetAll);

    // Model change updates start button
    DOM.modelSelect.addEventListener('change', updateStartButton);

    // Range sliders: update live value display
    $$('.form-range').forEach(range => {
        range.addEventListener('input', () => {
            const id = range.id.replace('param-', '');
            const label = $(`#val-${id}`);
            if (label) label.textContent = range.value;
        });
    });

    // Keyboard shortcut: Space for play/pause
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

    // Periodic health check
    setInterval(checkHealth, 30000);

    console.log('Song Cover Pipeline Web UI initialized');
}

// Boot
document.addEventListener('DOMContentLoaded', init);
