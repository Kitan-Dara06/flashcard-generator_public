document.addEventListener('DOMContentLoaded', function() {
    const passwordOverlay = document.getElementById('passwordOverlay');
    const passwordInput = document.getElementById('passwordInput');
    const passwordSubmit = document.getElementById('passwordSubmit');
    const passwordError = document.getElementById('passwordError');
    const fileInput = document.getElementById('fileInput');
    const uploadLabel = document.querySelector('.upload-label');
    const fileChip = document.getElementById('fileChip');
    const fileChipName = document.getElementById('fileChipName');
    const fileChipRemove = document.getElementById('fileChipRemove');
    const generateBtn = document.getElementById('generateBtn');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const progressBar = document.getElementById('progressBar');
    const cancelBtn = document.getElementById('cancelBtn');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const flashcardsContainer = document.getElementById('flashcards');
    const resultsCount = document.getElementById('resultsCount');
    const resultsStats = document.getElementById('resultsStats');
    const reviewToggle = document.getElementById('reviewToggle');
    const dueCount = document.getElementById('dueCount');
    const downloadBtn = document.getElementById('downloadBtn');
    const autoplayToggle = document.getElementById('autoplayToggle');

    const MAX_UPLOAD_BYTES = 3 * 1024 * 1024;   // base64 inflates ~33%; Vercel caps bodies at 4.5 MB
    const POLL_INTERVAL_MS = 2000;
    const MAX_JOB_MS = 20 * 60 * 1000;      // absolute ceiling (a big doc + cold start needs headroom)
    const STALL_MS = 6 * 60 * 1000;         // give up if *generation* stops progressing
    const QUEUED_GRACE_MS = 12 * 60 * 1000; // a cold boot may legitimately stay queued a while
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
    let lastActivityAt = 0;
    let lastProgress = '';
    let lastStatus = '';
    let lastWarmupAt = 0;
    let fallbackTried = false;      // one auto-retry on the backup engine per job

    // Leitner state lives in localStorage: { [questionHash]: { box, due, seen } }.
    // No backend, no Redis, no auth - it survives reloads on this device only.
    const LEITNER_KEY = 'leitner:v1';
    let leitnerState = loadLeitnerState();
    let reviewMode = false;

    function loadLeitnerState() {
        try {
            const parsed = JSON.parse(localStorage.getItem(LEITNER_KEY) || '{}');
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch (e) {
            return {};
        }
    }

    function saveLeitnerState() {
        try {
            localStorage.setItem(LEITNER_KEY, JSON.stringify(leitnerState));
        } catch (e) { /* private mode / quota: scheduling just won't persist */ }
    }

    function entryFor(card) {
        return leitnerState[Leitner.hashKey(card.question)];
    }

    function boxFor(card) {
        const entry = entryFor(card);
        return entry && entry.box ? entry.box : Leitner.initialBox(card.difficulty);
    }

    function dueCards(cards) {
        const now = Date.now();
        return cards.filter(function(card) { return Leitner.isDue(entryFor(card), now); });
    }

    function shuffled(list) {
        const out = list.slice();
        for (let i = out.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [out[i], out[j]] = [out[j], out[i]];
        }
        return out;
    }

    function gradeCard(card, gotIt) {
        const key = Leitner.hashKey(card.question);
        const seeded = { box: boxFor(card), seen: 0 };
        leitnerState[key] = Leitner.grade(leitnerState[key] || seeded, gotIt, Date.now());
        saveLeitnerState();
        renderCurrentView();
    }

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
        stopSpeaking();
        stopStudyPing();
        hideLoading();
        authToken = null;
        sessionStorage.removeItem('authToken');
        passwordOverlay.classList.remove('hidden');
        passwordInput.focus();
    }

    /* ---------------- pre-warm ---------------- */

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
        if (document.hidden) {
            stopSpeaking();
        } else {
            warmupNow();   // also fires on back-to-foreground tabs
        }
    });

    // Back-button / bfcache restore: the browser may have served a cached page
    // while the Modal container decayed to zero in the background. Re-warm
    // immediately so the next "Generate" click is instant.
    window.addEventListener('pageshow', function(e) {
        if (e.persisted) warmupNow();
    });

    // While the user is actively studying, ping the GPU every few minutes
    // so the scaledown_window never expires. Stops automatically when they
    // leave the results view. (Cooldown in warmupNow() prevents spam.)
    let studyTimer = null;
    function startStudyPing() {
        stopStudyPing();
        studyTimer = setInterval(warmupNow, 5 * 60 * 1000);
    }
    function stopStudyPing() {
        if (studyTimer) { clearInterval(studyTimer); studyTimer = null; }
    }

    /* ---------------- file selection ---------------- */

    function setFile(file) {
        if (file) {
            // Keep the hidden input in sync so the existing generate flow works.
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;

            generateBtn.disabled = false;
            generateBtn.textContent = `🧠 Generate from ${file.name}`;
            fileChipName.textContent = `✅ ${file.name}`;
            fileChip.classList.remove('hidden');
            document.body.classList.add('has-file');
            warmupNow();
        } else {
            fileInput.value = '';
            generateBtn.disabled = true;
            generateBtn.textContent = '🧠 Generate Flashcards';
            fileChip.classList.add('hidden');
            document.body.classList.remove('has-file');
        }
    }

    fileInput.addEventListener('change', function(e) {
        setFile(e.target.files[0] || null);
    });

    fileChipRemove.addEventListener('click', function() {
        setFile(null);
    });

    ['dragenter', 'dragover'].forEach(function(ev) {
        uploadLabel.addEventListener(ev, function(e) {
            e.preventDefault();
            uploadLabel.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(function(ev) {
        uploadLabel.addEventListener(ev, function(e) {
            e.preventDefault();
            uploadLabel.classList.remove('dragover');
        });
    });
    uploadLabel.addEventListener('drop', function(e) {
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) setFile(file);
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

        stopPolling();
        showLoading('Uploading and extracting text…');
        setProgress(-1);

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
            lastActivityAt = Date.now();
            lastProgress = '';
            lastStatus = 'queued';
            fallbackTried = false;
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
            showError('This job is taking unusually long' + jobRef() + '. It may still finish in the background.');
            return;
        }
        if (lastStatus === 'processing' && Date.now() - lastActivityAt > STALL_MS) {
            stopPolling();
            hideLoading();
            showError('Generation stalled for ' + Math.round(STALL_MS / 60000) + ' minutes' + jobRef() +
                ' — check the Modal logs.');
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
            lastStatus = data.status || '';

            // The server saw the job sit in "queued" past its patience. Retry it
            // once on the Gemini backup engine, which runs in Vercel and so still
            // works when Modal itself is the broken part.
            if (data.stalled && !fallbackTried) {
                fallbackTried = true;
                updateLoadingText('The AI worker is unavailable — switching to the backup engine…');
                setProgress(-1);
                try {
                    const fb = await fetch('/api/generate_flashcards', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({ job_id: currentJobId, provider: 'gemini' })
                    });
                    if (fb.status === 401) { expireAuth(); return; }
                    const fbData = await fb.json().catch(function() { return {}; });
                    if (fb.ok && fbData.success) {
                        // Cards are already in Redis; the next poll renders them.
                        lastActivityAt = Date.now();
                        schedulePoll(500);
                        return;
                    }
                    stopPolling();
                    hideLoading();
                    showError(fbData.error || 'The backup engine could not finish this document.');
                    return;
                } catch (err) {
                    stopPolling();
                    hideLoading();
                    showError('Could not reach the backup engine: ' + err.message);
                    return;
                }
            }

            if (data.status === 'queued') {
                const age = Number(data.age_seconds);
                if (Number.isFinite(age) && age > 300) {
                    updateLoadingText(`Still waiting for the AI worker to start… (${Math.floor(age / 60)} min). This usually means a cold start is unusually slow, or the worker is failing at boot.`);
                } else if (Number.isFinite(age) && age > 45) {
                    updateLoadingText(`Waiting for the AI worker to start… (cold start, ${age}s)`);
                } else {
                    updateLoadingText('Queued…');
                }
                const queuedSeconds = Number.isFinite(age) ? age : (Date.now() - jobStartedAt) / 1000;
                if (queuedSeconds * 1000 > QUEUED_GRACE_MS) {
                    stopPolling();
                    hideLoading();
                    showError('The AI worker never started' + jobRef() + ' (queued ' +
                        Math.round(queuedSeconds / 60) + ' min). Check the Modal logs.');
                    return;
                }
            } else if (data.status === 'processing') {
                const progress = data.progress || '';
                const match = progress.match(/^(\d+)\/(\d+)$/);
                if (progress !== lastProgress) {
                    lastProgress = progress;
                    lastActivityAt = Date.now();
                }
                if (progress && progress !== '0/0') {
                    updateLoadingText(`Generating flashcards… (chunk ${progress.replace('/', ' of ')})`);
                    if (match) {
                        const done = parseInt(match[1], 10);
                        const total = parseInt(match[2], 10);
                        if (total > 0) setProgress(Math.round((done / total) * 100));
                    }
                } else {
                    updateLoadingText('Generating flashcards…');
                }
            } else if (data.status === 'complete') {
                stopPolling();
                setProgress(100);
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

    /* ---------------- text to speech ---------------- */

    const TTS_MAX_CHARS = 600;
    const audioUrlCache = new Map();   // sentence -> object URL (session cache)
    let audioEl = null;
    let speakToken = 0;
    let serverTtsBroken = false;   // Piper lives in the Modal worker; see speakText()
    const ttsAborts = new Set();

    autoplayToggle.checked = localStorage.getItem('autoplay') === 'true';
    autoplayToggle.addEventListener('change', function() {
        localStorage.setItem('autoplay', autoplayToggle.checked ? 'true' : 'false');
        if (!autoplayToggle.checked) stopSpeaking();
    });

    // Split into sentences, then merge short ones so each request is a
    // sensible chunk (~1-2 sentences) and hard-split anything oversized.
    function splitSentences(text) {
        const cleaned = (text || '').replace(/\s+/g, ' ').trim();
        if (!cleaned) return [];
        const parts = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
        const merged = [];
        let buffer = '';
        parts.forEach(function(part) {
            const sentence = part.trim();
            if (!sentence) return;
            if (buffer && buffer.length + sentence.length > 220) {
                merged.push(buffer);
                buffer = sentence;
            } else {
                buffer = buffer ? buffer + ' ' + sentence : sentence;
            }
        });
        if (buffer) merged.push(buffer);

        const out = [];
        merged.forEach(function(chunk) {
            if (chunk.length <= 320) {
                out.push(chunk);
            } else {
                (chunk.match(/[\s\S]{1,320}(?:\s|$)/g) || []).forEach(function(piece) {
                    const trimmed = piece.trim();
                    if (trimmed) out.push(trimmed);
                });
            }
        });
        return out;
    }

    function stopSpeaking() {
        speakToken += 1;
        ttsAborts.forEach(function(controller) {
            try { controller.abort(); } catch (e) { /* already done */ }
        });
        ttsAborts.clear();
        if (audioEl) {
            try { audioEl.pause(); } catch (e) { /* not playing */ }
            audioEl.removeAttribute('src');
        }
        if ('speechSynthesis' in window) {
            try { window.speechSynthesis.cancel(); } catch (e) { /* not speaking */ }
        }
        document.querySelectorAll('.speak-btn.playing').forEach(function(btn) {
            btn.classList.remove('playing');
        });
    }

    function fetchAudioUrl(sentence) {
        if (audioUrlCache.has(sentence)) {
            return Promise.resolve(audioUrlCache.get(sentence));
        }
        const controller = new AbortController();
        ttsAborts.add(controller);
        return fetch('/api/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + authToken
            },
            body: JSON.stringify({ text: sentence.slice(0, TTS_MAX_CHARS) }),
            signal: controller.signal
        }).then(function(response) {
            if (response.status === 401) { expireAuth(); return null; }
            if (!response.ok) {
                // /api/tts is served by the GPU worker. If it is down (out of
                // Modal credit, worker failing to boot) stop asking for every
                // sentence and let the browser read aloud instead.
                serverTtsBroken = true;
                throw new Error('tts ' + response.status);
            }
            return response.blob();
        }).then(function(blob) {
            if (!blob) return null;
            const url = URL.createObjectURL(blob);
            if (audioUrlCache.size < 300) audioUrlCache.set(sentence, url);
            return url;
        }).finally(function() {
            ttsAborts.delete(controller);
        });
    }

    function playUrl(url) {
        return new Promise(function(resolve) {
            if (!audioEl) audioEl = new Audio();
            audioEl.onended = resolve;
            audioEl.onerror = resolve;
            audioEl.src = url;
            audioEl.play().catch(resolve);   // autoplay policy: gesture already happened
        });
    }

    // Browser-native speech: free, no backend, works when the TTS worker is gone.
    function speakWithBrowser(sentences, token) {
        return new Promise(function(resolve) {
            if (!('speechSynthesis' in window)) { resolve(); return; }
            let i = 0;
            function next() {
                if (token !== speakToken || i >= sentences.length) { resolve(); return; }
                const utterance = new SpeechSynthesisUtterance(sentences[i++]);
                utterance.onend = next;
                utterance.onerror = next;
                window.speechSynthesis.speak(utterance);
            }
            next();
        });
    }

    // Stream-by-pipelining: play sentence N while sentence N+1 is already downloading.
    async function speakText(text, button) {
        stopSpeaking();
        const token = ++speakToken;
        const sentences = splitSentences(text);
        if (!sentences.length) return;
        if (button) button.classList.add('playing');

        if (serverTtsBroken) {
            try {
                await speakWithBrowser(sentences, token);
            } finally {
                if (token === speakToken && button) button.classList.remove('playing');
            }
            return;
        }

        try {
            let pending = fetchAudioUrl(sentences[0]);
            for (let i = 0; i < sentences.length; i++) {
                const current = pending;
                pending = i + 1 < sentences.length ? fetchAudioUrl(sentences[i + 1]) : null;

                let url = null;
                try {
                    url = await current;
                } catch (err) {
                    if (token === speakToken) console.warn('TTS fetch failed', err);
                }
                if (token !== speakToken) return;
                if (!url) continue;

                await playUrl(url);
                if (token !== speakToken) return;
            }
        } finally {
            if (token === speakToken) {
                document.querySelectorAll('.speak-btn.playing').forEach(function(btn) {
                    btn.classList.remove('playing');
                });
            }
        }
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

    function setProgress(percent) {
        progressBar.style.width = (percent < 0 ? 8 : percent) + '%';
        progressBar.style.opacity = percent < 0 ? '0.45' : '1';
    }

    function jobRef() {
        return currentJobId ? ` (job ${currentJobId.slice(0, 8)})` : '';
    }

    function showLoading(text) {
        stopSpeaking();
        stopStudyPing();
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
        stopSpeaking();
        hideLoading();
        setProgress(-1);
        results.classList.remove('hidden');
        error.classList.add('hidden');

        resultsCount.textContent = `✅ Generated ${flashcards.length} flashcards!`;

        // Deck summary: difficulty mix + how many cards carry a source quote.
        const counts = { easy: 0, medium: 0, hard: 0 };
        flashcards.forEach(function(c) {
            const d = (c.difficulty || '').toLowerCase();
            if (counts[d] !== undefined) counts[d]++;
        });
        const sourced = flashcards.filter(function(c) {
            return (c.source_quote || '').trim();
        }).length;
        resultsStats.textContent =
            `${counts.easy} easy · ${counts.medium} medium · ${counts.hard} hard · ${sourced}/${flashcards.length} sourced`;

        reviewMode = false;
        renderCurrentView();

        startStudyPing();   // keep the GPU warm for the next generation
    }

    // Due count is always visible so the nudge exists before review mode is on.
    function updateReviewButton() {
        dueCount.textContent = String(dueCards(currentFlashcards).length);
        reviewToggle.classList.toggle('active', reviewMode);
        reviewToggle.setAttribute('aria-pressed', reviewMode ? 'true' : 'false');
    }

    function renderCurrentView() {
        stopSpeaking();   // re-rendering replaces the audio buttons underneath playback
        const cards = reviewMode ? shuffled(dueCards(currentFlashcards)) : currentFlashcards;
        updateReviewButton();
        flashcardsContainer.innerHTML = '';

        if (reviewMode && cards.length === 0) {
            const done = document.createElement('p');
            done.className = 'review-empty';
            done.textContent = '🎉 Nothing due right now — come back tomorrow, or turn Review due off to browse the deck.';
            flashcardsContainer.appendChild(done);
            return;
        }

        cards.forEach(function(card, index) {
            flashcardsContainer.appendChild(createFlashcardElement(card, index + 1));
        });
    }

    reviewToggle.addEventListener('click', function() {
        reviewMode = !reviewMode;
        renderCurrentView();
    });

    function showError(message) {
        stopSpeaking();
        hideLoading();
        setProgress(-1);
        results.classList.add('hidden');
        error.classList.remove('hidden');
        error.textContent = '❌ ' + message;
    }

    function createFlashcardElement(card, number) {
        const flashcard = document.createElement('div');
        flashcard.className = 'flashcard';
        flashcard.style.animationDelay = Math.min((number - 1) * 18, 500) + 'ms';

        const rawDifficulty = (card.difficulty || '').toLowerCase();
        const difficulty = ['easy', 'medium', 'hard'].includes(rawDifficulty) ? rawDifficulty : '';
        const quote = (card.source_quote || '').trim();
        const box = boxFor(card);
        const chip = difficulty
            ? `<span class="difficulty-chip ${difficulty}">${difficulty}</span>`
            : '';
        const boxBadge = `<span class="box-badge" title="Leitner box ${box} of ${Leitner.MAX_BOX}">B${box}</span>`;
        const gradeActions = reviewMode ? `
                    <div class="grade-actions">
                        <button class="grade-btn again" type="button">Again</button>
                        <button class="grade-btn got" type="button">Got it</button>
                    </div>` : '';

        flashcard.innerHTML = `
            <div class="flashcard-inner">
                <div class="flashcard-side flashcard-front">
                    <div class="card-top">
                        <button class="speak-btn" type="button" title="Read question aloud" aria-label="Read question aloud">🔊</button>
                        <div class="card-tags">${chip}${boxBadge}</div>
                        <span class="card-number">Q${number}</span>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${escapeHtml(card.question)}</p>
                    </div>
                    <div class="card-foot">tap to flip</div>
                </div>
                <div class="flashcard-side flashcard-back">
                    <div class="card-top">
                        <button class="speak-btn" type="button" title="Read answer aloud" aria-label="Read answer aloud">🔊</button>
                        <div class="card-tags"></div>
                        <span class="card-number">A${number}</span>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${escapeHtml(card.answer)}</p>
                        ${quote ? `
                        <button class="quote-toggle" type="button">📄 Show source</button>
                        <blockquote class="source-quote">${escapeHtml(quote)}</blockquote>` : ''}
                    </div>
                    ${reviewMode ? '' : '<div class="card-foot">tap to flip back</div>'}
                    ${gradeActions}
                </div>
            </div>
        `;

        const inner = flashcard.querySelector('.flashcard-inner');
        inner.addEventListener('click', function() {
            inner.classList.toggle('flip');
            if (autoplayToggle.checked) {
                const showingAnswer = inner.classList.contains('flip');
                speakText(showingAnswer ? card.answer : card.question, null);
            }
        });

        const quoteToggle = flashcard.querySelector('.quote-toggle');
        if (quoteToggle) {
            quoteToggle.addEventListener('click', function(e) {
                e.stopPropagation();   // revealing the source should not flip the card away
                const back = flashcard.querySelector('.flashcard-back');
                const open = back.classList.toggle('quote-open');
                quoteToggle.textContent = open ? '📄 Hide source' : '📄 Show source';
            });
        }

        if (reviewMode) {
            flashcard.querySelector('.grade-btn.again').addEventListener('click', function(e) {
                e.stopPropagation();
                gradeCard(card, false);
            });
            flashcard.querySelector('.grade-btn.got').addEventListener('click', function(e) {
                e.stopPropagation();
                gradeCard(card, true);
            });
        }

        flashcard.querySelector('.flashcard-front .speak-btn').addEventListener('click', function(e) {
            e.stopPropagation();
            speakText(card.question, e.currentTarget);
        });
        flashcard.querySelector('.flashcard-back .speak-btn').addEventListener('click', function(e) {
            e.stopPropagation();
            speakText(card.answer, e.currentTarget);
        });

        return flashcard;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML.replace(/\n/g, '<br>');
    }
});
