/* Study Decks — single-user, offline-first.
 *
 * Decks, cards, stars, podcast script and audio live in IndexedDB (db.js).
 * The network is touched only to create decks, generate podcasts, and read aloud.
 * No auth: this is one student on one device.
 */
(function () {
    'use strict';

    // ---- element refs ----
    const $ = function (id) { return document.getElementById(id); };
    const homeView = $('homeView');
    const deckView = $('deckView');
    const fileInput = $('fileInput');
    const fileChip = $('fileChip');
    const fileChipName = $('fileChipName');
    const fileChipRemove = $('fileChipRemove');
    const generateBtn = $('generateBtn');
    const loading = $('loading');
    const loadingText = $('loadingText');
    const progressBar = $('progressBar');
    const cancelBtn = $('cancelBtn');
    const deckList = $('deckList');
    const noDecks = $('noDecks');
    const deckTitle = $('deckTitle');
    const deckMeta = $('deckMeta');
    const backBtn = $('backBtn');
    const exportBtn = $('exportBtn');
    const searchInput = $('searchInput');
    const shuffleBtn = $('shuffleBtn');
    const allCount = $('allCount');
    const starredCount = $('starredCount');
    const emptyFilter = $('emptyFilter');
    const studyCard = $('studyCard');
    const prevBtn = $('prevBtn');
    const nextBtn = $('nextBtn');
    const studyProgress = $('studyProgress');
    const error = $('error');
    // podcast
    const podcastBtn = $('podcastBtn');
    const podcastPanel = $('podcastPanel');
    const podcastClose = $('podcastClose');
    const podcastTitle = $('podcastTitle');
    const podcastGenScript = $('podcastGenScript');
    const podcastGenAudio = $('podcastGenAudio');
    const podcastStatus = $('podcastStatus');
    const podcastPlayer = $('podcastPlayer');
    const podcastPlay = $('podcastPlay');
    const podcastProgress = $('podcastProgress');
    const podcastTime = $('podcastTime');
    const podcastTranscript = $('podcastTranscript');

    // ---- constants ----
    const MAX_UPLOAD_BYTES = 3 * 1024 * 1024;
    const POLL_INTERVAL_MS = 2000;
    const MAX_JOB_MS = 20 * 60 * 1000;

    // ---- state ----
    let currentDeckId = null;
    let deckCards = [];
    let viewIndexes = [];
    let cursor = 0;
    let filter = 'all';
    let fallbackTried = false;
    let currentJobId = null;
    let podcast = null;          // { title, turns }
    let podcastSegs = [];        // [{ turns:[idx], url, duration }]
    let podAudio = null;
    let podSegIndex = -1;
    let podPlaying = false;

    // ---- utils ----
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    }
    function shuffle(list) {
        const out = list.slice();
        for (let i = out.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = out[i]; out[i] = out[j]; out[j] = tmp;
        }
        return out;
    }
    function hashText(s) {
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
        return (h >>> 0).toString(36);
    }
    function showError(message) { error.textContent = '❌ ' + message; error.classList.remove('hidden'); }
    function clearError() { error.classList.add('hidden'); }
    function fileToBase64(file) {
        return new Promise(function (resolve, reject) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = function () { resolve(reader.result); };
            reader.onerror = function (e) { reject(e); };
        });
    }

    // =====================================================================
    // Generation
    // =====================================================================
    function setProgress(p) { progressBar.style.width = (p < 0 ? 8 : p) + '%'; progressBar.style.opacity = p < 0 ? '0.45' : '1'; }
    function updateLoadingText(t) { loadingText.textContent = t; }
    function showLoading(t) { clearError(); loading.classList.remove('hidden'); cancelBtn.classList.remove('hidden'); updateLoadingText(t || 'Working…'); setProgress(-1); generateBtn.disabled = true; }
    function hideLoading() { loading.classList.add('hidden'); cancelBtn.classList.add('hidden'); generateBtn.disabled = !fileInput.files[0]; }
    function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

    async function generateDeck() {
        const file = fileInput.files[0];
        if (!file) return;
        if (file.size > MAX_UPLOAD_BYTES) { showError('File too large (max 3 MB).'); return; }
        showLoading('Uploading and extracting text…');
        try {
            const dataUrl = await fileToBase64(file);
            const b64 = dataUrl.split(',')[1];
            const resp = await fetch('/api/generate_flashcards', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_content: b64, file_type: file.type })
            });
            const data = await resp.json();
            if (!data.success) { hideLoading(); showError(data.message || data.error || 'Failed to start'); return; }
            currentJobId = data.job_id;
            fallbackTried = false;
            await pollJob(currentJobId, file.name);
        } catch (err) {
            hideLoading();
            showError('Generation failed: ' + err.message);
        }
    }

    async function pollJob(jobId, sourceName) {
        const startedAt = Date.now();
        let lastProgress = '';
        while (true) {
            if (Date.now() - startedAt > MAX_JOB_MS) { hideLoading(); showError('Taking unusually long.'); return; }
            let data;
            try {
                const resp = await fetch('/api/status?job_id=' + encodeURIComponent(jobId));
                data = await resp.json();
            } catch (e) { await wait(POLL_INTERVAL_MS); continue; }

            if (data.status === 'complete') {
                try {
                    await persistDeck(data.cards, sourceName, data.provider || 'qwen', data.text || '');
                    hideLoading();
                    setFile(null);
                    await openDeck(currentDeckId);
                } catch (e) {
                    hideLoading();
                    showError('Saved the deck but failed to open it: ' + e.message);
                }
                return;
            } else if (data.status === 'error') {
                hideLoading();
                showError(data.error || 'Generation failed');
                return;
            } else if (data.status === 'processing') {
                const p = data.progress || '';
                if (p && p !== '0/0' && p !== lastProgress) {
                    lastProgress = p;
                    const m = p.match(/^(\d+)\/(\d+)$/);
                    if (m) setProgress(Math.round((parseInt(m[1], 10) / parseInt(m[2], 10)) * 100));
                    updateLoadingText('Generating… (chunk ' + p.replace('/', ' of ') + ')');
                }
            } else if (data.status === 'queued') {
                const age = Number(data.age_seconds);
                if (Number.isFinite(age) && age > 45) updateLoadingText('Waiting for the AI worker… (cold start, ' + age + 's)');
                if (data.stalled && !fallbackTried) {
                    fallbackTried = true;
                    updateLoadingText('Switching to the backup engine…');
                    setProgress(-1);
                    try {
                        const fb = await fetch('/api/generate_flashcards', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ job_id: jobId, provider: 'gemini' })
                        });
                        const fbData = await fb.json().catch(function () { return {}; });
                        if (fb.ok && fbData.success) continue;
                        hideLoading();
                        showError(fbData.error || 'Backup engine failed.');
                        return;
                    } catch (e) { hideLoading(); showError('Could not reach the backup engine.'); return; }
                } else if (data.stalled) {
                    hideLoading();
                    showError('The AI worker never started.');
                    return;
                }
            }
            await wait(POLL_INTERVAL_MS);
        }
    }

    async function persistDeck(cards, sourceName, provider, text) {
        const now = Date.now();
        const deckId = hashText(sourceName + '|' + now);
        cards.forEach(function (c) { c.starred = false; });
        await Store.saveDeck({
            id: deckId,
            title: sourceName.replace(/\.[^.]+$/, ''),
            sourceName: sourceName,
            createdAt: now,
            provider: provider,
            text: text
        }, cards);
        currentDeckId = deckId;
    }

    // =====================================================================
    // Home
    // =====================================================================
    async function renderHome() {
        stopPodcast();
        podcastPanel.classList.add('hidden');
        deckView.classList.add('hidden');
        homeView.classList.remove('hidden');
        let decks = [];
        try { decks = await Store.listDecks(); } catch (e) { decks = []; }
        deckList.innerHTML = '';
        noDecks.classList.toggle('hidden', decks.length > 0);
        decks.forEach(function (deck) {
            const el = document.createElement('button');
            el.className = 'deck-item';
            el.type = 'button';
            el.innerHTML =
                '<div class="deck-item-main">' +
                '  <span class="deck-item-title">' + escapeHtml(deck.title) + '</span>' +
                '  <span class="deck-item-meta">' + deck.cardCount + ' cards · ' +
                (deck.starredCount ? deck.starredCount + ' starred' : 'no stars') + '</span>' +
                '</div><span class="deck-item-open">Open →</span>';
            el.addEventListener('click', function () { openDeck(deck.id); });

            const del = document.createElement('button');
            del.className = 'deck-item-delete';
            del.type = 'button';
            del.setAttribute('aria-label', 'Delete deck');
            del.textContent = '✕';
            del.addEventListener('click', function (e) {
                e.stopPropagation();
                if (confirm('Delete "' + deck.title + '"? This cannot be undone.')) {
                    Store.deleteDeck(deck.id).then(renderHome);
                }
            });

            const wrap = document.createElement('div');
            wrap.className = 'deck-item-wrap';
            wrap.appendChild(el);
            wrap.appendChild(del);
            deckList.appendChild(wrap);
        });
    }

    // =====================================================================
    // Deck view
    // =====================================================================
    async function openDeck(id) {
        currentDeckId = id;
        deckCards = await Store.getCards(id);
        filter = 'all';
        searchInput.value = '';
        cursor = 0;
        recomputeView();
        homeView.classList.add('hidden');
        deckView.classList.remove('hidden');
        clearError();
        const deck = await Store.getDeck(id);
        deckTitle.textContent = deck ? deck.title : 'Deck';
        deckMeta.textContent = deckCards.length + ' cards';
        renderStudyCard();
    }

    function recomputeView() {
        const q = searchInput.value.trim().toLowerCase();
        viewIndexes = [];
        deckCards.forEach(function (card, i) {
            if (filter === 'starred' && !card.starred) return;
            if (q) {
                const hay = (card.question + ' ' + card.answer + ' ' + (card.source_quote || '')).toLowerCase();
                if (hay.indexOf(q) === -1) return;
            }
            viewIndexes.push(i);
        });
        cursor = Math.min(cursor, Math.max(0, viewIndexes.length - 1));
        allCount.textContent = deckCards.length;
        starredCount.textContent = deckCards.filter(function (c) { return c.starred; }).length;
        emptyFilter.classList.toggle('hidden', viewIndexes.length > 0);
        studyCard.classList.toggle('hidden', viewIndexes.length === 0);
        prevBtn.disabled = nextBtn.disabled = viewIndexes.length <= 1;
        updateProgress();
    }

    function updateProgress() {
        studyProgress.textContent = viewIndexes.length ? (cursor + 1) + ' / ' + viewIndexes.length : '';
    }

    function renderStudyCard() {
        const idx = viewIndexes[cursor];
        if (idx == null) { studyCard.innerHTML = ''; return; }
        const card = deckCards[idx];
        const chip = ['easy', 'medium', 'hard'].includes(card.difficulty)
            ? '<span class="difficulty-chip ' + card.difficulty + '">' + card.difficulty + '</span>' : '';
        const starClass = card.starred ? 'star-btn starred' : 'star-btn';
        const quote = (card.source_quote || '').trim();

        studyCard.innerHTML =
            '<div class="flashcard"><div class="flashcard-inner" id="fcInner">' +
            '  <div class="flashcard-side flashcard-front">' +
            '    <div class="card-top">' +
            '      <button class="speak-btn" id="speakQ" type="button" title="Read question aloud">🔊</button>' +
            '      <div class="card-tags">' + chip + '</div>' +
            '      <button class="' + starClass + '" id="starBtn" type="button" title="Star / unstar (S)">★</button>' +
            '    </div>' +
            '    <div class="card-body"><p class="card-text">' + escapeHtml(card.question) + '</p></div>' +
            '    <div class="card-foot">tap to flip</div>' +
            '  </div>' +
            '  <div class="flashcard-side flashcard-back">' +
            '    <div class="card-top">' +
            '      <button class="speak-btn" id="speakA" type="button" title="Read answer aloud">🔊</button>' +
            '      <div class="card-tags"></div>' +
            '      <button class="' + starClass + '" id="starBtnBack" type="button" title="Star / unstar (S)">★</button>' +
            '    </div>' +
            '    <div class="card-body">' +
            '      <p class="card-text">' + escapeHtml(card.answer) + '</p>' +
            (quote ? '<button class="quote-toggle" id="quoteToggle" type="button">📄 Show source</button>' +
                     '<blockquote class="source-quote">' + escapeHtml(quote) + '</blockquote>' : '') +
            '    </div>' +
            '    <div class="card-foot">tap to flip back</div>' +
            '  </div></div></div>';

        const inner = $('fcInner');
        inner.addEventListener('click', function () { toggleFlip(); });
        const q = $('quoteToggle');
        if (q) q.addEventListener('click', function (e) {
            e.stopPropagation();
            const back = inner.querySelector('.flashcard-back');
            const open = back.classList.toggle('quote-open');
            q.textContent = open ? '📄 Hide source' : '📄 Show source';
        });
        $('starBtn').addEventListener('click', function (e) { e.stopPropagation(); toggleStar(); });
        $('starBtnBack').addEventListener('click', function (e) { e.stopPropagation(); toggleStar(); });
        $('speakQ').addEventListener('click', function (e) { e.stopPropagation(); speakText(card.question, e.currentTarget); });
        $('speakA').addEventListener('click', function (e) { e.stopPropagation(); speakText(card.answer, e.currentTarget); });
    }

    function toggleFlip() {
        const inner = $('fcInner');
        if (inner) inner.classList.toggle('flip');
    }

    function toggleStar() {
        const idx = viewIndexes[cursor];
        if (idx == null) return;
        deckCards[idx].starred = !deckCards[idx].starred;
        Store.setStarred(currentDeckId, idx, deckCards[idx].starred);
        starredCount.textContent = deckCards.filter(function (c) { return c.starred; }).length;
        const inner = $('fcInner');
        const flipped = inner && inner.classList.contains('flip');
        renderStudyCard();
        if (flipped) $('fcInner').classList.add('flip');
    }

    function step(dir) {
        if (viewIndexes.length === 0) return;
        cursor = (cursor + dir + viewIndexes.length) % viewIndexes.length;
        renderStudyCard();
        updateProgress();
        const inner = $('fcInner');
        if (inner) inner.classList.remove('flip');
    }

    // =====================================================================
    // Export
    // =====================================================================
    function exportCsv() {
        if (deckCards.length === 0) return;
        const header = ['question', 'answer', 'source_quote', 'difficulty', 'starred'];
        const rows = deckCards.map(function (c) {
            return [c.question, c.answer, c.source_quote || '', c.difficulty || '', c.starred ? 'yes' : ''].map(csvCell).join(',');
        });
        const csv = '\ufeff' + [header.join(',')].concat(rows).join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = (deckTitle.textContent || 'deck').replace(/[^\w\-]+/g, '_') + '.csv';
        a.click();
        URL.revokeObjectURL(url);
    }
    function csvCell(v) {
        const s = String(v == null ? '' : v);
        return '"' + s.replace(/"/g, '""') + '"';
    }

    // =====================================================================
    // Podcast
    // =====================================================================
    async function openPodcast() {
        stopPodcast();
        podcastPanel.classList.remove('hidden');
        const saved = await Store.getPodcast(currentDeckId);
        if (saved && saved.turns && saved.turns.length) {
            podcast = { title: saved.title || deckTitle.textContent, turns: saved.turns };
            podcastTitle.textContent = podcast.title;
            renderTranscript();
            podcastGenScript.classList.add('hidden');
            podcastGenAudio.classList.remove('hidden');
            podcastStatus.textContent = 'Script ready (' + podcast.turns.length + ' turns).';
        } else {
            podcast = null;
            podcastTitle.textContent = 'Podcast';
            podcastTranscript.innerHTML = '';
            podcastGenScript.classList.remove('hidden');
            podcastGenAudio.classList.add('hidden');
            podcastStatus.textContent = 'No podcast yet for this deck.';
        }
        // reset player state
        podcastPlayer.classList.add('hidden');
        podcastSegs = [];
        podSegIndex = -1;
        podPlaying = false;
        podcastPlay.textContent = '▶';
        podcastProgress.style.width = '0%';
        podcastTime.textContent = '0:00';
    }

    function renderTranscript() {
        if (!podcast || !podcast.turns.length) { podcastTranscript.innerHTML = ''; return; }
        podcastTranscript.innerHTML = podcast.turns.map(function (t, i) {
            return '<div class="podcast-turn" data-turn="' + i + '">' +
                '<span class="podcast-speaker ' + t.speaker.toLowerCase() + '">' + escapeHtml(t.speaker) + '</span>' +
                '<span class="podcast-text">' + escapeHtml(t.text) + '</span></div>';
        }).join('');
    }

    async function generateScript() {
        const deck = await Store.getDeck(currentDeckId);
        const text = deck && deck.text;
        if (!text) { podcastStatus.textContent = 'This deck has no saved source text. Regenerate the deck to enable podcast.'; return; }
        podcastGenScript.disabled = true;
        podcastStatus.textContent = 'Writing the script…';
        try {
            const resp = await fetch('/api/podcast_script', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.error || 'script failed');
            podcast = { title: data.title, turns: data.turns };
            await Store.savePodcast(currentDeckId, podcast);
            podcastTitle.textContent = data.title;
            renderTranscript();
            podcastGenScript.classList.add('hidden');
            podcastGenAudio.classList.remove('hidden');
            podcastStatus.textContent = 'Script ready (' + data.turns.length + ' turns).';
        } catch (e) {
            podcastStatus.textContent = 'Script failed: ' + e.message;
        } finally {
            podcastGenScript.disabled = false;
        }
    }

    function buildSegments(turns, maxChars) {
        const segs = [];
        let cur = [];
        let len = 0;
        turns.forEach(function (t, i) {
            const l = t.text.length + 8;
            if (cur.length && len + l > maxChars) { segs.push(cur); cur = [i]; len = l; }
            else { cur.push(i); len += l; }
        });
        if (cur.length) segs.push(cur);
        return segs;
    }

    function dialogueFor(indices) {
        return indices.map(function (i) { return podcast.turns[i].speaker + ': ' + podcast.turns[i].text; }).join('\n');
    }

    async function generateAudio() {
        if (!podcast || !podcast.turns.length) return;
        const segments = buildSegments(podcast.turns, 900);
        podcastGenAudio.disabled = true;
        podcastStatus.textContent = 'Generating audio…';
        podcastSegs = new Array(segments.length);
        let done = 0;
        let next = 0;
        const CONCURRENCY = 3;

        async function worker() {
            while (next < segments.length) {
                const i = next++;
                const dialogue = dialogueFor(segments[i]);
                const key = hashText(dialogue);
                let blob, duration;
                try {
                    const cached = await Store.getAudio(key);
                    if (cached && cached.blob) { blob = cached.blob; duration = cached.duration; }
                } catch (e) { /* ignore */ }
                if (!blob) {
                    const resp = await fetch('/api/podcast_audio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dialogue: dialogue })
                    });
                    if (!resp.ok) throw new Error('segment ' + (i + 1) + ' failed (' + resp.status + ')');
                    blob = await resp.blob();
                    duration = parseFloat(resp.headers.get('X-Audio-Duration') || '0') || estimateDuration(segments[i]);
                    await Store.putAudio(key, blob, duration);
                }
                podcastSegs[i] = { turns: segments[i], url: URL.createObjectURL(blob), duration: duration };
                done++;
                podcastStatus.textContent = 'Audio ' + done + ' / ' + segments.length;
            }
        }

        try {
            await Promise.all([worker(), worker(), worker()]);
            podcastStatus.textContent = '';
            podcastGenAudio.classList.add('hidden');
            podcastPlayer.classList.remove('hidden');
        } catch (e) {
            podcastStatus.textContent = 'Audio failed: ' + e.message;
        } finally {
            podcastGenAudio.disabled = false;
        }
    }

    function estimateDuration(indices) {
        const chars = indices.reduce(function (s, i) { return s + podcast.turns[i].text.length; }, 0);
        return chars / 15;
    }

    function totalDuration() {
        return podcastSegs.reduce(function (s, seg) { return s + (seg.duration || 0); }, 0);
    }

    function togglePodcastPlay() {
        if (!podcastSegs.length) return;
        if (podPlaying) { podAudio.pause(); podPlaying = false; podcastPlay.textContent = '▶'; }
        else if (podSegIndex < 0) { playSegment(0); }
        else { podAudio.play(); podPlaying = true; podcastPlay.textContent = '⏸'; }
    }

    function playSegment(i) {
        podSegIndex = i;
        if (!podAudio) {
            podAudio = new Audio();
            podAudio.onended = onSegmentEnded;
            podAudio.ontimeupdate = onTimeUpdate;
        }
        podAudio.src = podcastSegs[i].url;
        podAudio.play().catch(function () {});
        podPlaying = true;
        podcastPlay.textContent = '⏸';
    }

    function onSegmentEnded() {
        if (podSegIndex + 1 < podcastSegs.length) playSegment(podSegIndex + 1);
        else { podPlaying = false; podcastPlay.textContent = '▶'; highlightTurn(-1); }
    }

    function onTimeUpdate() {
        const seg = podcastSegs[podSegIndex];
        if (!seg || !podAudio) return;
        let elapsed = 0;
        for (let i = 0; i < podSegIndex; i++) elapsed += podcastSegs[i].duration || 0;
        elapsed += podAudio.currentTime;
        const total = totalDuration();
        if (total > 0) podcastProgress.style.width = (100 * elapsed / total) + '%';
        podcastTime.textContent = fmtTime(elapsed);
        highlightTurn(podSegIndex, seg.duration ? podAudio.currentTime / seg.duration : 0);
    }

    function highlightTurn(segIdx, frac) {
        podcastTranscript.querySelectorAll('.podcast-turn').forEach(function (r) { r.classList.remove('active'); });
        if (segIdx < 0) return;
        const seg = podcastSegs[segIdx];
        const chars = seg.turns.map(function (i) { return Math.max(1, podcast.turns[i].text.length); });
        const total = chars.reduce(function (a, b) { return a + b; }, 0);
        let acc = 0;
        for (let k = 0; k < seg.turns.length; k++) {
            const start = acc / total;
            acc += chars[k];
            const end = acc / total;
            if (frac >= start && frac < end) {
                const row = podcastTranscript.querySelector('[data-turn="' + seg.turns[k] + '"]');
                if (row) {
                    row.classList.add('active');
                    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                }
                return;
            }
        }
    }

    function stopPodcast() {
        if (podAudio) { podAudio.pause(); }
        podPlaying = false;
        podcastPlay.textContent = '▶';
    }

    function fmtTime(s) {
        s = Math.max(0, Math.floor(s));
        return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
    }

    // =====================================================================
    // TTS (Piper on Modal; browser speech fallback)
    // =====================================================================
    const TTS_MAX_CHARS = 600;
    let audioEl = null;
    let speakToken = 0;
    let serverTtsBroken = false;
    const audioUrlCache = new Map();

    function splitSentences(text) {
        const cleaned = (text || '').replace(/\s+/g, ' ').trim();
        if (!cleaned) return [];
        const parts = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
        const merged = [];
        let buffer = '';
        parts.forEach(function (part) {
            const s = part.trim();
            if (!s) return;
            if (buffer && buffer.length + s.length > 220) { merged.push(buffer); buffer = s; }
            else { buffer = buffer ? buffer + ' ' + s : s; }
        });
        if (buffer) merged.push(buffer);
        return merged;
    }

    function stopSpeaking() {
        speakToken += 1;
        if (audioEl) { try { audioEl.pause(); } catch (e) {} audioEl.removeAttribute('src'); }
        if ('speechSynthesis' in window) { try { window.speechSynthesis.cancel(); } catch (e) {} }
        document.querySelectorAll('.speak-btn.playing').forEach(function (b) { b.classList.remove('playing'); });
    }

    function fetchAudioUrl(sentence) {
        if (audioUrlCache.has(sentence)) return Promise.resolve(audioUrlCache.get(sentence));
        return fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: sentence.slice(0, TTS_MAX_CHARS) })
        }).then(function (response) {
            if (!response.ok) { serverTtsBroken = true; throw new Error('tts ' + response.status); }
            return response.blob();
        }).then(function (blob) {
            if (!blob) return null;
            const url = URL.createObjectURL(blob);
            if (audioUrlCache.size < 300) audioUrlCache.set(sentence, url);
            return url;
        });
    }

    function playUrl(url) {
        return new Promise(function (resolve) {
            if (!audioEl) audioEl = new Audio();
            audioEl.onended = resolve;
            audioEl.onerror = resolve;
            audioEl.src = url;
            audioEl.play().catch(resolve);
        });
    }

    function speakWithBrowser(sentences, token) {
        return new Promise(function (resolve) {
            if (!('speechSynthesis' in window)) { resolve(); return; }
            let i = 0;
            function next() {
                if (token !== speakToken || i >= sentences.length) { resolve(); return; }
                const u = new SpeechSynthesisUtterance(sentences[i++]);
                u.onend = next; u.onerror = next;
                window.speechSynthesis.speak(u);
            }
            next();
        });
    }

    async function speakText(text, button) {
        stopSpeaking();
        const token = ++speakToken;
        const sentences = splitSentences(text);
        if (!sentences.length) return;
        if (button) button.classList.add('playing');
        if (serverTtsBroken) {
            try { await speakWithBrowser(sentences, token); }
            finally { if (button) button.classList.remove('playing'); }
            return;
        }
        try {
            for (let i = 0; i < sentences.length; i++) {
                let url = null;
                try { url = await fetchAudioUrl(sentences[i]); }
                catch (err) {
                    if (token === speakToken) { serverTtsBroken = true; await speakWithBrowser(sentences.slice(i), token); }
                    break;
                }
                if (token !== speakToken) break;
                if (!url) continue;
                await playUrl(url);
                if (token !== speakToken) break;
            }
        } finally {
            if (button) button.classList.remove('playing');
        }
    }

    // =====================================================================
    // Wire-up
    // =====================================================================
    generateBtn.addEventListener('click', generateDeck);
    exportBtn.addEventListener('click', exportCsv);
    backBtn.addEventListener('click', function () { stopSpeaking(); renderHome(); });
    shuffleBtn.addEventListener('click', function () { viewIndexes = shuffle(viewIndexes); cursor = 0; renderStudyCard(); updateProgress(); });
    searchInput.addEventListener('input', function () { cursor = 0; recomputeView(); renderStudyCard(); });
    document.querySelectorAll('.tab').forEach(function (tab) {
        tab.addEventListener('click', function () {
            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            tab.classList.add('active');
            filter = tab.getAttribute('data-filter');
            cursor = 0;
            recomputeView();
            renderStudyCard();
        });
    });
    prevBtn.addEventListener('click', function () { step(-1); });
    nextBtn.addEventListener('click', function () { step(1); });

    document.addEventListener('keydown', function (e) {
        if (deckView.classList.contains('hidden')) return;
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON')) {
            if (e.target === searchInput && e.key === 'Escape') searchInput.blur();
            return;
        }
        if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
        else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
        else if (e.key === ' ' || e.key === 'Spacebar') { toggleFlip(); e.preventDefault(); }
        else if (e.key === 's' || e.key === 'S') { toggleStar(); }
        else if (e.key === '/') { searchInput.focus(); e.preventDefault(); }
    });

    function setFile(file) {
        if (file) {
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            generateBtn.disabled = false;
            fileChipName.textContent = '✅ ' + file.name;
            fileChip.classList.remove('hidden');
        } else {
            fileInput.value = '';
            generateBtn.disabled = true;
            fileChip.classList.add('hidden');
        }
    }
    fileInput.addEventListener('change', function (e) { setFile(e.target.files[0] || null); });
    fileChipRemove.addEventListener('click', function () { setFile(null); });
    cancelBtn.addEventListener('click', function () {
        if (currentJobId) {
            fetch('/api/cancel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: currentJobId })
            }).catch(function () {});
        }
        hideLoading();
    });

    // podcast wiring
    podcastBtn.addEventListener('click', openPodcast);
    podcastClose.addEventListener('click', function () { stopPodcast(); podcastPanel.classList.add('hidden'); });
    podcastGenScript.addEventListener('click', generateScript);
    podcastGenAudio.addEventListener('click', generateAudio);
    podcastPlay.addEventListener('click', togglePodcastPlay);

    // service worker (offline app shell)
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function () {
            navigator.serviceWorker.register('/sw.js').catch(function () { /* offline-in-dev or unsupported */ });
        });
    }

    // init
    renderHome();
    if (typeof indexedDB === 'undefined') {
        showError('This browser cannot store decks (IndexedDB unavailable). Decks will not persist.');
    }
})();
