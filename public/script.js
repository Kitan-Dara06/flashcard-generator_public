document.addEventListener('DOMContentLoaded', function() {
    const passwordOverlay = document.getElementById('passwordOverlay');
    const passwordInput = document.getElementById('passwordInput');
    const passwordSubmit = document.getElementById('passwordSubmit');
    const passwordError = document.getElementById('passwordError');
    const fileInput = document.getElementById('fileInput');
    const generateBtn = document.getElementById('generateBtn');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const cancelBtn = document.getElementById('cancelBtn');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const flashcardsContainer = document.getElementById('flashcards');
    const resultsCount = document.getElementById('resultsCount');
    const downloadBtn = document.getElementById('downloadBtn');

    const MAX_UPLOAD_BYTES = 3 * 1024 * 1024;   // base64 inflates ~33%; Vercel caps bodies at 4.5 MB
    const POLL_INTERVAL_MS = 2000;
    const MAX_JOB_MS = 10 * 60 * 1000;
    const WARMUP_COOLDOWN_MS = 60 * 1000;

    let authToken = sessionStorage.getItem('authToken');
    if (authToken) {
        passwordOverlay.classList.add('hidden');
    }

    let currentFlashcards = [];
    let currentJobId = null;
    let pollTimer = null;
    let pollFailures = 0;
    let jobStartedAt = 0;
    let lastWarmupAt = 0;

    /* ---------------- auth ---------------- */

    passwordSubmit.addEventListener('click', checkPassword);
    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            checkPassword();
        }
    });

    async function checkPassword() {
        const enteredPassword = passwordInput.value;
        try {
            const response = await fetch('/api/verify_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    password: enteredPassword
                })
            });
            const data = await response.json();
            if (data.success && data.authenticated) {
                authToken = data.token;
                sessionStorage.setItem('authToken', authToken);
                passwordOverlay.classList.add('hidden');
                passwordError.classList.add('hidden');
                passwordInput.value = '';
                warmupNow();
            } else {
                passwordError.textContent = '❌ Incorrect password';
                passwordError.classList.remove('hidden');
                passwordInput.value = '';
                passwordInput.focus();
            }
        } catch (err) {
            passwordError.textContent = '❌ Error verifying password';
            passwordError.classList.remove('hidden');
            passwordInput.value = '';
        }
    }

    function expireAuth() {
        stopPolling();
        hideLoading();
        authToken = null;
        sessionStorage.removeItem('authToken');
        passwordOverlay.classList.remove('hidden');
        passwordInput.focus();
    }

    /* ---------------- pre-warm (Layer 4) ---------------- */

    function warmupNow() {
        if (!authToken) return;
        const now = Date.now();
        if (now - lastWarmupAt < WARMUP_COOLDOWN_MS) return;
        lastWarmupAt = now;
        fetch('/api/warmup', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + authToken }
        }).catch(function() { /* warmup is best-effort */ });
    }

    window.addEventListener('load', warmupNow);
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) warmupNow();
    });

    /* ---------------- file selection ---------------- */

    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            generateBtn.disabled = false;
            generateBtn.textContent = `🧠 Generate from ${file.name}`;
            warmupNow();
        } else {
            generateBtn.disabled = true;
            generateBtn.textContent = '🧠 Generate Flashcards';
        }
    });

    /* ---------------- generation flow ---------------- */

    generateBtn.addEventListener('click', async function() {
        const file = fileInput.files[0];
        if (!file) return;

        if (!authToken) {
            passwordOverlay.classList.remove('hidden');
            passwordInput.focus();
            return;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
            showError('File is too large for upload (max 3 MB). Try a smaller document.');
            return;
        }

        clearJob();
        showLoading('Uploading and extracting text…');

        try {
            const base64File = await fileToBase64(file);
            const response = await fetch('/api/generate_flashcards', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken,
                },
                body: JSON.stringify({
                    file_content: base64File.split(',')[1], // Remove data:mime;base64, prefix
                    file_type: file.type
                })
            });

            if (response.status === 401) { expireAuth(); return; }

            const data = await response.json();
            if (!response.ok || !data.success) {
                showError(data.message || data.error || 'Failed to start generation');
                return;
            }

            currentJobId = data.job_id;
            jobStartedAt = Date.now();
            pollFailures = 0;
            updateLoadingText('Queued…');
            schedulePoll(800);
        } catch (err) {
            showError('Failed to start generation: ' + err.message);
        }
    });

    cancelBtn.addEventListener('click', async function() {
        if (!currentJobId) return;
        const jobId = currentJobId;
        try {
            await fetch('/api/cancel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + authToken,
                },
                body: JSON.stringify({ job_id: jobId })
            });
        } catch (err) { /* the worker checks the flag between chunks anyway */ }
        stopPolling();
        hideLoading();
    });

    function schedulePoll(delay) {
        if (pollTimer) clearTimeout(pollTimer);
        pollTimer = setTimeout(pollStatus, delay);
    }

    async function pollStatus() {
        pollTimer = null;
        if (!currentJobId) return;

        if (Date.now() - jobStartedAt > MAX_JOB_MS) {
            stopPolling();
            hideLoading();
            showError('This is taking longer than expected. The job may still finish in the background.');
            return;
        }

        try {
            const response = await fetch('/api/status?job_id=' + encodeURIComponent(currentJobId), {
                headers: { 'Authorization': 'Bearer ' + authToken }
            });

            if (response.status === 401) { expireAuth(); return; }

            if (response.status === 404) {
                stopPolling();
                hideLoading();
                showError('Job not found or expired.');
                return;
            }
            if (!response.ok) throw new Error('status ' + response.status);

            pollFailures = 0;
            const data = await response.json();

            if (data.status === 'queued') {
                updateLoadingText('Queued…');
            } else if (data.status === 'processing') {
                const progress = data.progress || '';
                updateLoadingText(progress && progress !== '0/0'
                    ? `Generating flashcards… (chunk ${progress.replace('/', ' of ')})`
                    : 'Generating flashcards…');
            } else if (data.status === 'complete') {
                stopPolling();
                currentFlashcards = data.cards || [];
                showResults(currentFlashcards);
                return;
            } else if (data.status === 'error') {
                stopPolling();
                hideLoading();
                showError(data.error || 'Generation failed');
                return;
            } else if (data.status === 'cancelled') {
                stopPolling();
                hideLoading();
                return;
            }
        } catch (err) {
            pollFailures += 1;
            if (pollFailures >= 5) {
                stopPolling();
                hideLoading();
                showError('Lost connection to the server. Please try again.');
                return;
            }
        }
        schedulePoll(POLL_INTERVAL_MS);
    }

    function stopPolling() {
        if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
        }
        currentJobId = null;
    }

    function clearJob() {
        stopPolling();
    }

    /* ---------------- rendering ---------------- */

    downloadBtn.addEventListener('click', function() {
        if (currentFlashcards.length === 0) return;

        const dataStr = JSON.stringify(currentFlashcards, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'flashcards.json';
        link.click();
    });

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    function showLoading(text) {
        loading.classList.remove('hidden');
        cancelBtn.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');
        updateLoadingText(text || 'Working…');
        generateBtn.disabled = true;
    }

    function hideLoading() {
        loading.classList.add('hidden');
        cancelBtn.classList.add('hidden');
        generateBtn.disabled = !fileInput.files[0];
    }

    function updateLoadingText(text) {
        loadingText.textContent = text;
    }

    function showResults(flashcards) {
        hideLoading();
        results.classList.remove('hidden');
        error.classList.add('hidden');

        resultsCount.textContent = `✅ Generated ${flashcards.length} flashcards!`;

        flashcardsContainer.innerHTML = '';
        flashcards.forEach((card, index) => {
            const flashcardEl = createFlashcardElement(card, index + 1);
            flashcardsContainer.appendChild(flashcardEl);
        });
    }

    function showError(message) {
        hideLoading();
        results.classList.add('hidden');
        error.classList.remove('hidden');
        error.textContent = '❌ ' + message;
    }

    function createFlashcardElement(card, number) {
        const flashcard = document.createElement('div');
        flashcard.className = 'flashcard';

        flashcard.innerHTML = `
            <div class="flashcard-inner" onclick="this.classList.toggle('flip')">
                <div class="flashcard-front">
                    <div class="flashcard-number">Q${number}</div>
                    <div>${escapeHtml(card.question)}</div>
                </div>
                <div class="flashcard-back">
                    <div class="flashcard-number">A${number}</div>
                    <div>${escapeHtml(card.answer)}</div>
                </div>
            </div>
        `;

        return flashcard;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
});
