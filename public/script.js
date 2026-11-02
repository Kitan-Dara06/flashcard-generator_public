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
    const podcastGenAllAudio = $('podcastGenAllAudio');
    const podcastStatus = $('podcastStatus');
    const episodeList = $('episodeList');
    const podcastTranscript = $('podcastTranscript');
    const miniPlayer = $('miniPlayer');
    const miniPlay = $('miniPlay');
    const miniTitle = $('miniTitle');
    const miniTime = $('miniTime');
    const miniProgress = $('miniProgress');
    const miniOpen = $('miniOpen');
    const typeModeBtn = $('typeModeBtn');

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
    // Set by the Cancel button. The generation loops check it between requests so
    // cancelling actually stops the work instead of only hiding the spinner -
    // without it the chunk loop kept firing (and kept paying) for minutes, then
    // saved and opened the deck anyway.
    let cancelRequested = false;
    // Guards the one automatic recovery attempt for a job that went idle (closed
    // tab, or every engine failed). Already-finished groups are in Redis, so
    // resuming costs nothing for them.
    let resumeTried = false;
    let podcast = [];            // episodes: [{title,hook,turns,terms,coverage,segs,voice}]
    let activeEpisode = 0;
    let podAudio = null;
    let podSegIndex = -1;        // segment index within the active episode
    let podPlaying = false;
    // True while any episode is being written or synthesised. Without it,
    // clicking a second episode mid-generation started a second loop over the
    // same segment array - two writers racing on one episode's clips.
    let podcastBusy = false;
    // Typing mode: opt-in, self-assessed, no LLM calls. The grade is the star.
    let typingMode = false;

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
        // The API generates one section group per request, so nothing appears to
        // happen for the first few seconds while the text is extracted.
        let hintTimer = setTimeout(function () {
            updateLoadingText('Generating cards… this can take up to a minute.');
        }, 4000);
        try {
            const dataUrl = await fileToBase64(file);
            const b64 = dataUrl.split(',')[1];
            const resp = await fetch('/api/generate_flashcards', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_content: b64, file_type: file.type })
            });
            const data = await resp.json();
            clearTimeout(hintTimer);
            if (!data.success) { hideLoading(); showError(data.message || data.error || 'Failed to start'); return; }
            currentJobId = data.job_id;
            fallbackTried = false;
            cancelRequested = false;
            resumeTried = false;
            // The API answers with a chunk plan: one request per section group.
            if (data.chunks > 0) {
                await runChunks(currentJobId, data.chunks, file.name);
            } else {
                await pollJob(currentJobId, file.name);
            }
        } catch (err) {
            clearTimeout(hintTimer);
            hideLoading();
            showError('Generation failed: ' + err.message);
        }
    }

    // Drive Gemini one section group at a time. Each call is a few tens of
    // seconds - inside the serverless limit - and progress is real.
    async function runChunks(jobId, total, sourceName, startAt) {
        for (let i = (startAt || 0); i < total; i++) {
            if (cancelRequested) return;
            updateLoadingText('Generating cards… (section group ' + (i + 1) + ' of ' + total + ')');
            setProgress(Math.round((i / total) * 100));

            let data = {};
            let ok = false;
            try {
                const resp = await fetch('/api/generate_flashcards', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_id: jobId, chunk: i })
                });
                data = await resp.json().catch(function () { return {}; });
                ok = resp.ok && data.success;
            } catch (err) {
                hideLoading();
                showError('Lost connection while generating: ' + err.message);
                return;
            }

            if (data.quota || data.unavailable) {
                // Every API engine is out of credit or overloaded: hand the job to Modal.
                updateLoadingText((data.quota
                    ? 'Out of API credit — switching to the backup engine…'
                    : 'The API is busy — switching to the backup engine…'));
                setProgress(-1);
                try {
                    await fetch('/api/generate_flashcards', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_id: jobId, provider: 'qwen' })
                    });
                } catch (e) { /* pollJob reports the failure */ }
                await pollJob(jobId, sourceName);
                return;
            }

            if (data.cancelled) return;      // the server saw the cancel flag

            if (!ok) {
                hideLoading();
                showError(data.error || 'Generation failed at section group ' + (i + 1));
                return;
            }

            if (cancelRequested) return;
            setProgress(Math.round(((i + 1) / total) * 100));
            if (data.final) break;   // merged and deduped server-side
        }
        if (cancelRequested) return;
        await finishJob(jobId, sourceName);
    }

    async function finishJob(jobId, sourceName) {
        let data = {};
        try {
            const resp = await fetch('/api/status?job_id=' + encodeURIComponent(jobId));
            data = await resp.json();
        } catch (e) {
            hideLoading();
            showError('Could not read the finished job: ' + e.message);
            return;
        }
        if (data.status === 'complete') {
            hideLoading();
            setFile(null);
            try {
                await persistDeck(data.cards, sourceName, data.provider || 'api', data.text || '');
                await openDeck(currentDeckId);
                if (data.note) deckMeta.textContent += ' · ' + data.note;
            } catch (e) {
                showError('Saved the deck but failed to open it: ' + e.message);
            }
            return;
        }
        if (data.status === 'error') {
            hideLoading();
            showError(data.error || 'Generation failed');
            return;
        }
        if (data.status === 'cancelled') {
            hideLoading();
            return;
        }
        await pollJob(jobId, sourceName);
    }

    async function pollJob(jobId, sourceName) {
        const startedAt = Date.now();
        let lastProgress = '';
        while (true) {
            if (cancelRequested) return;
            if (Date.now() - startedAt > MAX_JOB_MS) { hideLoading(); showError('Taking unusually long.'); return; }
            let data;
            try {
                const resp = await fetch('/api/status?job_id=' + encodeURIComponent(jobId));
                data = await resp.json();
            } catch (e) { await wait(POLL_INTERVAL_MS); continue; }

            if (data.status === 'complete') {
                try {
                    await persistDeck(data.cards, sourceName, data.provider || 'api', data.text || '');
                    hideLoading();
                    setFile(null);
                    await openDeck(currentDeckId);
                    if (data.note) deckMeta.textContent += ' · ' + data.note;
                } catch (e) {
                    hideLoading();
                    showError('Saved the deck but failed to open it: ' + e.message);
                }
                return;
            } else if (data.status === 'error') {
                hideLoading();
                showError(data.error || 'Generation failed');
                return;
            } else if (data.status === 'cancelled') {
                hideLoading();
                return;
            } else if (data.status === 'processing') {
                const p = data.progress || '';
                if (p && p !== '0/0' && p !== lastProgress) {
                    lastProgress = p;
                    const m = p.match(/^(\d+)\/(\d+)$/);
                    if (m) setProgress(Math.round((parseInt(m[1], 10) / parseInt(m[2], 10)) * 100));
                    updateLoadingText('Generating… (chunk ' + p.replace('/', ' of ') + ')');
                }
                // Nothing has landed for a while: the tab was closed mid-job, or an
                // engine failed and the client gave up. Groups already generated are
                // in Redis, so continue from the first missing one instead of
                // reporting a failure and losing the work.
                if (data.stalled && data.resumable && !resumeTried) {
                    resumeTried = true;
                    const totalChunks = Number(data.total_chunks) || 0;
                    if (totalChunks > 0) {
                        updateLoadingText('Resuming where it stopped…');
                        setProgress(-1);
                        await runChunks(jobId, totalChunks, sourceName,
                                        Number(data.done_chunks) || 0);
                        return;
                    }
                    hideLoading();
                    showError(data.error || 'Generation stalled.');
                    return;
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
                        if (fb.ok && fbData.success) {
                            // The server handed back a chunk plan: drive it here
                            // rather than waiting for a worker that never started.
                            if (fbData.chunks > 0) {
                                await runChunks(jobId, fbData.chunks, sourceName);
                                return;
                            }
                            continue;
                        }
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

        if (typingMode) {
            studyCard.innerHTML =
                '<div class="flashcard"><div class="flashcard-inner" id="fcInner">' +
                '  <div class="flashcard-side flashcard-front">' +
                '    <div class="card-top">' +
                '      <div class="card-tags">' + chip + '</div>' +
                '      <button class="' + starClass + '" id="starBtn" type="button" title="Star / unstar (S)">★</button>' +
                '    </div>' +
                '    <div class="card-body"><p class="card-text">' + escapeHtml(card.question) + '</p>' +
                '      <textarea id="typeAnswer" class="type-answer" rows="3" placeholder="Type your answer, then Reveal…"></textarea>' +
                '      <button id="typeReveal" class="generate-btn" type="button">Reveal</button>' +
                '    </div>' +
                '  </div>' +
                '  <div class="flashcard-side flashcard-back">' +
                '    <div class="card-top">' +
                '      <button class="speak-btn" id="speakA" type="button" title="Read answer aloud">🔊</button>' +
                '      <div class="card-tags"></div>' +
                '      <button class="' + starClass + '" id="starBtnBack" type="button" title="Star / unstar (S)">★</button>' +
                '    </div>' +
                '    <div class="card-body">' +
                '      <div id="typeComparison"></div>' +
                '      <p class="card-text">' + escapeHtml(card.answer) + '</p>' +
                (quote ? '<button class="quote-toggle" id="quoteToggle" type="button">📄 Show source</button>' +
                         '<blockquote class="source-quote">' + escapeHtml(quote) + '</blockquote>' : '') +
                '    </div>' +
                '    <div class="type-grade">' +
                '      <button class="grade-btn again" data-grade="again" type="button">Again</button>' +
                '      <button class="grade-btn hard" data-grade="hard" type="button">Hard</button>' +
                '      <button class="grade-btn good" data-grade="good" type="button">Got it</button>' +
                '    </div>' +
                '  </div></div></div>';
            $('typeReveal').addEventListener('click', revealTypedAnswer);
            $('typeAnswer').addEventListener('keydown', function (e) {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); revealTypedAnswer(); }
            });
            $('starBtn').addEventListener('click', function (e) { e.stopPropagation(); toggleStar(); });
            $('starBtnBack').addEventListener('click', function (e) { e.stopPropagation(); toggleStar(); });
            $('speakA').addEventListener('click', function (e) { e.stopPropagation(); speakText(card.answer, e.currentTarget); });
            const q2 = $('quoteToggle');
            if (q2) q2.addEventListener('click', function (e) {
                e.stopPropagation();
                const back = $('fcInner').querySelector('.flashcard-back');
                const open = back.classList.toggle('quote-open');
                q2.textContent = open ? '📄 Hide source' : '📄 Show source';
            });
            document.querySelectorAll('.grade-btn').forEach(function (b) {
                b.addEventListener('click', function (e) { e.stopPropagation(); grade(b.dataset.grade); });
            });
            return;
        }

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

    // ---- typing mode: local, offline, self-assessed ----------------
    const TYPING_STOP = new Set(('a an the of to and or for with in on at by from as is are was were be been being has have had do does did not no yes it its this that these those i you he she they we our your his her their them me us then than so if but about above after again also among any because before below between both can could did down during each few more most much now off once only other over own same should some such there under until very what when where which while who whom why will would').split(' '));

    function keyWords(s) {
        return (s.toLowerCase().match(/[a-z0-9]+/g) || [])
            .filter(function (w) { return w.length >= 4 && !TYPING_STOP.has(w); });
    }

    // Words in the expected answer the student did not type. These are the signal:
    // what was forgotten, not what was remembered. No score is shown - a paraphrase
    // that is right in different words will still show misses, so the student is
    // the judge (via Again / Hard / Got it), never the string match.
    function missedTerms(expected, user) {
        const have = new Set(keyWords(user));
        return Array.from(new Set(keyWords(expected))).filter(function (w) { return !have.has(w); });
    }

    function revealTypedAnswer() {
        const idx = viewIndexes[cursor];
        if (idx == null) return;
        const card = deckCards[idx];
        const typed = $('typeAnswer') ? $('typeAnswer').value : '';
        const missed = missedTerms(card.answer, typed);
        const box = $('typeComparison');
        if (box) {
            const missedSet = new Set(missed);
            const highlighted = escapeHtml(card.answer).replace(/[A-Za-z0-9]+/g, function (word) {
                return missedSet.has(word.toLowerCase())
                    ? '<mark class="missed">' + word + '</mark>' : word;
            });
            box.innerHTML = '<div class="type-your"><span class="type-label">Your answer</span>' +
                '<div class="type-typed">' + (escapeHtml(typed) || '<em>nothing typed</em>') + '</div></div>' +
                '<div class="type-expected"><span class="type-label">Expected</span>' +
                '<div>' + highlighted + '</div></div>';
        }
        toggleFlip();
    }

    // Again/Hard -> star on (goes back into the cram pile); Got it -> star off.
    function grade(level) {
        const idx = viewIndexes[cursor];
        if (idx == null) return;
        const shouldStar = level !== 'good';
        if (deckCards[idx].starred !== shouldStar) {
            deckCards[idx].starred = shouldStar;
            Store.setStarred(currentDeckId, idx, shouldStar);
            starredCount.textContent = deckCards.filter(function (c) { return c.starred; }).length;
        }
        step(1);
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
    // Podcast (episodes, drawer + persistent mini-player)
    // =====================================================================
    const VOICE_PAIRS = [
        ['Kore', 'Puck'],
        ['Zephyr', 'Charon'],
        ['Aoede', 'Fenrir'],
        ['Leda', 'Orus'],
        ['Puck', 'Zephyr'],
        ['Charon', 'Kore'],
    ];
    let podElapsed = 0;

    function voiceFor(episodeIndex) {
        return VOICE_PAIRS[episodeIndex % VOICE_PAIRS.length];
    }

    async function openPodcast() {
        podcastPanel.classList.remove('hidden');
        const record = await Store.getPodcast(currentDeckId);
        podcast = Store.normalizeEpisodes(record);
        activeEpisode = 0;
        if (podcast.length) {
            podcastTitle.textContent = 'Podcast · ' + podcast.length + ' episode' + (podcast.length === 1 ? '' : 's');
            renderEpisodeList();
            renderTranscript();
            podcastGenScript.classList.add('hidden');
            podcastGenAllAudio.classList.remove('hidden');
            podcastStatus.textContent = podcast.length + ' script' + (podcast.length === 1 ? '' : 's') + ' ready.';
        } else {
            podcastTitle.textContent = 'Podcast';
            episodeList.innerHTML = '';
            podcastTranscript.innerHTML = '';
            podcastGenScript.classList.remove('hidden');
            podcastGenAllAudio.classList.add('hidden');
            podcastStatus.textContent = 'No episodes yet for this deck.';
        }
        syncMiniPlayer();
    }

    function renderEpisodeList() {
        if (!podcast.length) { episodeList.innerHTML = ''; return; }
        episodeList.innerHTML = podcast.map(function (ep, i) {
            const hasAudio = ep.segs && ep.segs.length;
            return '<button class="episode-item' + (i === activeEpisode ? ' active' : '') + '" data-ep="' + i + '">' +
                '<span class="episode-index">' + (i + 1) + '</span>' +
                '<span class="episode-body">' +
                '<span class="episode-title">' + escapeHtml(ep.title || ep.hook || ('Episode ' + (i + 1))) + '</span>' +
                '<span class="episode-meta">' + ep.turns.length + ' turns' +
                (hasAudio ? ' · ' + ep.segs.length + ' clips' : ' · no audio') + '</span></span>' +
                '<span class="episode-action">' + (hasAudio ? '▶' : '🎧') + '</span></button>';
        }).join('');
        episodeList.querySelectorAll('.episode-item').forEach(function (btn) {
            btn.addEventListener('click', function () { episodeAction(Number(btn.dataset.ep)); });
        });
    }

    async function episodeAction(i) {
        if (!podcast[i] || podcastBusy) return;
        activeEpisode = i;
        renderEpisodeList();
        renderTranscript();
        if (podcast[i].segs && podcast[i].segs.length) {
            selectEpisode(i);
        } else {
            podcastBusy = true;
            try { await generateAudio(i); }
            finally { podcastBusy = false; }
            if (podcast[i].segs && podcast[i].segs.length) selectEpisode(i);
            renderEpisodeList();
        }
    }

    function selectEpisode(i) {
        activeEpisode = i;
        podSegIndex = -1;
        podElapsed = 0;
        renderEpisodeList();
        renderTranscript();
        if (podcast[i].segs && podcast[i].segs.length) playSegment(0);
        syncMiniPlayer();
    }

    function renderTranscript() {
        const ep = podcast[activeEpisode];
        if (!ep || !ep.turns.length) { podcastTranscript.innerHTML = ''; return; }
        podcastTranscript.innerHTML = ep.turns.map(function (t, i) {
            return '<div class="podcast-turn" data-turn="' + i + '">' +
                '<span class="podcast-speaker ' + t.speaker.toLowerCase() + '">' + escapeHtml(t.speaker) + '</span>' +
                '<span class="podcast-text">' + escapeHtml(t.text) + '</span></div>';
        }).join('');
    }

    // Scripts are driven one episode per request, exactly like card chunks. Doing
    // all N in one call would be N LLM calls against a 60 s function limit, and a
    // 10-episode deck would always time out.
    async function generateScript() {
        if (podcastBusy) return;
        const deck = await Store.getDeck(currentDeckId);
        const text = deck && deck.text;
        if (!text) { podcastStatus.textContent = 'This deck has no saved source text. Regenerate the deck to enable podcast.'; return; }
        const cards = await Store.getCards(currentDeckId);
        // A cancelled card job leaves this set; without clearing it the loop
        // below would break out on its first iteration and produce no episodes.
        cancelRequested = false;
        podcastGenScript.disabled = true;
        podcastBusy = true;
        podcastStatus.textContent = 'Planning episodes…';
        const episodes = [];
        const failures = [];
        try {
            const plan = await fetch('/api/podcast_script', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, cards: cards })
            });
            const planData = await plan.json();
            if (!plan.ok || !planData.success) throw new Error(planData.error || 'planning failed');
            const total = planData.total || 1;
            for (let i = 0; i < total; i++) {
                if (cancelRequested) break;
                podcastStatus.textContent = 'Writing episode ' + (i + 1) + ' of ' + total + '…';
                let data = {};
                try {
                    const resp = await fetch('/api/podcast_script', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, cards: cards, episode: i })
                    });
                    data = await resp.json();
                    if (!resp.ok || !data.success) { failures.push(i + 1); continue; }
                } catch (e) { failures.push(i + 1); continue; }
                const ep = data.episode;
                ep.segs = [];
                episodes.push(ep);
                // Save as we go: a deck interrupted here keeps the episodes done.
                await Store.savePodcast(currentDeckId, episodes);
            }
            if (!episodes.length) throw new Error('no episode scripts generated');
            podcast = episodes;
            activeEpisode = 0;
            podcastTitle.textContent = 'Podcast · ' + episodes.length + ' episode' + (episodes.length === 1 ? '' : 's');
            renderEpisodeList();
            renderTranscript();
            podcastGenScript.classList.add('hidden');
            podcastGenAllAudio.classList.remove('hidden');
            podcastStatus.textContent = episodes.length + ' script' + (episodes.length === 1 ? '' : 's') + ' ready'
                + (failures.length ? '; episode' + (failures.length > 1 ? 's ' : ' ') + failures.join(', ') + ' failed — press again to retry them' : '') + '.';
        } catch (e) {
            podcastStatus.textContent = 'Script failed: ' + e.message;
        } finally {
            podcastGenScript.disabled = false;
            podcastBusy = false;
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

    function episodeDialogue(ep, seg) {
        return seg.map(function (i) { return ep.turns[i].speaker + ': ' + ep.turns[i].text; }).join('\n');
    }

    // One segment's audio, retried, with a voice fallback.
    // The provider intermittently refuses content that succeeds moments later
    // (finishReason=OTHER, no audio), and the TTS quota is per-minute, so a
    // failure here usually means "ask again shortly" rather than "this is broken".
    // An unknown voice is a 400; it is retried with the verified default pair so
    // voice variety never costs the episode. Returns {resp, pair}.
    async function synthesizeSegment(dialogue, voice) {
        const requested = (voice && voice.length === 2) ? voice : ['Kore', 'Puck'];
        const pairs = (requested[0] === 'Kore' && requested[1] === 'Puck')
            ? [requested] : [requested, ['Kore', 'Puck']];
        const attempts = 3;
        let lastError = 'no attempt made';
        for (let p = 0; p < pairs.length; p++) {
            for (let attempt = 1; attempt <= attempts; attempt++) {
                try {
                    const resp = await fetch('/api/podcast_audio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ dialogue: dialogue, voices: pairs[p] })
                    });
                    if (resp.ok) return { resp: resp, pair: pairs[p] };
                    let detail = '';
                    try { detail = (await resp.json()).error || ''; } catch (e) { /* not json */ }
                    lastError = detail || ('HTTP ' + resp.status);
                    if (resp.status === 400 && p < pairs.length - 1) break;  // bad voice, try default
                } catch (e) {
                    lastError = e.message;
                }
                if (attempt < attempts) await wait(1500 * attempt * attempt);
            }
        }
        throw new Error(lastError);
    }

    // Synthesize one episode's audio, pacing itself against the per-minute quota
    // and collecting failures rather than aborting. Successful clips are cached in
    // IndexedDB, so retrying only pays for the gaps.
    async function generateAudio(epIndex) {
        const ep = podcast[epIndex];
        if (!ep || !ep.turns.length) return;
        const segments = buildSegments(ep.turns, 900);
        const voice = voiceFor(epIndex);
        podcastStatus.textContent = 'Audio for "' + (ep.title || ep.hook) + '"…';
        ep.segs = new Array(segments.length);
        const CONCURRENCY = 1;      // the quota is per-minute; sequential is safer
        let done = 0;
        const failures = [];

        for (let i = 0; i < segments.length; i++) {
            const dialogue = episodeDialogue(ep, segments[i]);
            let blob = null, duration = 0, pair = voice;
            const checkKey = function (p) {
                return (currentDeckId || 'no-deck') + '|' + p.join(',') + '|' + hashText(dialogue);
            };
            try {
                const cached = await Store.getAudio(checkKey(voice));
                if (cached && cached.blob) { blob = cached.blob; duration = cached.duration; }
                else {
                    // A previous episode may have had to fall back to the default
                    // pair; reusing that avoids re-synthesising with a rejected voice.
                    const alt = await Store.getAudio(checkKey(['Kore', 'Puck']));
                    if (alt && alt.blob) { blob = alt.blob; duration = alt.duration; }
                }
            } catch (e) { /* ignore */ }
            if (!blob) {
                try {
                    const result = await synthesizeSegment(dialogue, voice);
                    pair = result.pair;
                    blob = await result.resp.blob();
                    duration = parseFloat(result.resp.headers.get('X-Audio-Duration') || '0') || estimateDuration(ep, segments[i]);
                    await Store.putAudio(checkKey(pair), blob, duration);
                } catch (e) {
                    failures.push(i + 1);
                    podcastStatus.textContent = 'Audio ' + done + ' / ' + segments.length
                        + ' (segment ' + (i + 1) + ' failed: ' + e.message + ')';
                    continue;
                }
            }
            ep.segs[i] = { turns: segments[i], url: URL.createObjectURL(blob), duration: duration };
            done++;
            podcastStatus.textContent = 'Audio ' + done + ' / ' + segments.length;
            renderEpisodeList();
            await wait(1200);       // pace against the per-minute TTS quota
        }
        if (failures.length) {
            podcastStatus.textContent = 'Generated ' + done + ' of ' + segments.length
                + ' clips; ' + failures.join(', ') + ' failed. Click the episode again — finished ones are reused.';
        } else {
            podcastStatus.textContent = '';
        }
    }

    async function generateAllAudio() {
        if (podcastBusy) return;
        podcastBusy = true;
        podcastGenAllAudio.disabled = true;
        try {
            for (let i = 0; i < podcast.length; i++) {
                await generateAudio(i);
                await wait(2500);
            }
        } finally {
            podcastBusy = false;
            podcastGenAllAudio.disabled = false;
            syncMiniPlayer();     // clips now exist for episodes the user is not on
        }
    }

    function estimateDuration(ep, indices) {
        const chars = indices.reduce(function (s, i) { return s + ep.turns[i].text.length; }, 0);
        return chars / 15;
    }

    function episodeDuration(ep) {
        return (ep.segs || []).reduce(function (s, seg) { return s + (seg.duration || 0); }, 0);
    }

    function togglePodcastPlay() {
        const ep = podcast[activeEpisode];
        if (!ep || !(ep.segs || []).length) return;
        if (podPlaying) { podAudio.pause(); podPlaying = false; syncMiniPlayer(); return; }
        if (podSegIndex < 0) playSegment(0);
        else { podAudio.play(); podPlaying = true; syncMiniPlayer(); }
    }

    function playSegment(i) {
        const ep = podcast[activeEpisode];
        if (!ep || !(ep.segs || [])[i]) return;
        stopSpeaking();              // one voice at a time: card TTS vs podcast
        podSegIndex = i;
        if (!podAudio) {
            podAudio = new Audio();
            podAudio.onended = onSegmentEnded;
            podAudio.ontimeupdate = onTimeUpdate;
        }
        podAudio.src = ep.segs[i].url;
        podAudio.play().catch(function () {});
        podPlaying = true;
        syncMiniPlayer();
    }

    function onSegmentEnded() {
        const ep = podcast[activeEpisode];
        if (!ep) return;
        const segs = ep.segs || [];
        if (podSegIndex + 1 < segs.length) { playSegment(podSegIndex + 1); return; }
        if (activeEpisode + 1 < podcast.length) { selectEpisode(activeEpisode + 1); return; }
        podPlaying = false;
        syncMiniPlayer();
        highlightTurn(-1);
    }

    function onTimeUpdate() {
        const ep = podcast[activeEpisode];
        if (!ep || !podAudio) return;
        const segs = ep.segs || [];
        const seg = segs[podSegIndex];
        if (!seg) return;
        let elapsed = 0;
        for (let i = 0; i < podSegIndex; i++) elapsed += segs[i].duration || 0;
        elapsed += podAudio.currentTime;
        podElapsed = elapsed;
        const total = episodeDuration(ep);
        if (total > 0) miniProgress.style.width = (100 * elapsed / total) + '%';
        miniTime.textContent = fmtTime(elapsed);
        highlightTurn(podSegIndex, seg.duration ? podAudio.currentTime / seg.duration : 0);
    }

    function highlightTurn(segIdx, frac) {
        podcastTranscript.querySelectorAll('.podcast-turn').forEach(function (r) { r.classList.remove('active'); });
        const ep = podcast[activeEpisode];
        if (!ep || segIdx < 0) return;
        const seg = (ep.segs || [])[segIdx];
        if (!seg) return;
        const chars = seg.turns.map(function (i) { return Math.max(1, ep.turns[i].text.length); });
        const total = chars.reduce(function (a, b) { return a + b; }, 0);
        let acc = 0;
        for (let k = 0; k < seg.turns.length; k++) {
            const start = acc / total;
            acc += chars[k];
            const end = acc / total;
            if (frac >= start && frac < end) {
                const row = podcastTranscript.querySelector('[data-turn="' + seg.turns[k] + '"]');
                if (row) { row.classList.add('active'); row.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
                return;
            }
        }
    }

    function stopPodcast() {
        if (podAudio) { podAudio.pause(); }
        podPlaying = false;
        syncMiniPlayer();
    }

    function syncMiniPlayer() {
        const ep = podcast[activeEpisode];
        const hasAudio = !!(ep && (ep.segs || []).length);
        miniPlayer.classList.toggle('hidden', !hasAudio);
        // The bar is fixed over the deck; without clearance it would cover the
        // nav buttons and the keyboard hint on the last thing you try to tap.
        document.body.classList.toggle('pod-open', hasAudio);
        miniPlay.textContent = podPlaying ? '⏸' : '▶';
        miniTitle.textContent = ep ? (ep.title || ep.hook || 'Podcast') : 'Podcast';
        miniTime.textContent = fmtTime(podElapsed);
        if (hasAudio) {
            const total = episodeDuration(ep);
            if (total > 0) miniProgress.style.width = (100 * podElapsed / total) + '%';
        }
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
        stopPodcast();               // one voice at a time: card TTS vs podcast
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
    backBtn.addEventListener('click', function () { stopSpeaking(); stopPodcast(); renderHome(); });
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

    // typing mode: opt-in, persisted, no LLM calls
    typingMode = localStorage.getItem('typingMode') === '1';
    function setTypingMode(on) {
        typingMode = on;
        localStorage.setItem('typingMode', on ? '1' : '0');
        typeModeBtn.classList.toggle('active', on);
        renderStudyCard();
    }
    typeModeBtn.classList.toggle('active', typingMode);
    typeModeBtn.addEventListener('click', function () { setTypingMode(!typingMode); });

    document.addEventListener('keydown', function (e) {
        if (deckView.classList.contains('hidden')) return;
        // Typing mode puts a TEXTAREA on the card. Without it here, a space
        // in the student's answer would flip the card and swallow the space.
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
                      || e.target.tagName === 'BUTTON')) {
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
        // Stop the loops first, then tell the server: the flag is what makes an
        // in-flight chunk sequence actually stop instead of finishing in the
        // background and opening the deck anyway.
        cancelRequested = true;
        if (currentJobId) {
            fetch('/api/cancel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: currentJobId })
            }).catch(function () {});
        }
        currentJobId = null;
        hideLoading();
    });

    // podcast wiring
    podcastBtn.addEventListener('click', openPodcast);
    // Closing the drawer hides the transcript but keeps audio playing, so the
    // user can keep flipping cards while the episode runs (see the mini-player).
    podcastClose.addEventListener('click', function () { podcastPanel.classList.add('hidden'); });
    podcastGenScript.addEventListener('click', generateScript);
    podcastGenAllAudio.addEventListener('click', generateAllAudio);
    miniPlay.addEventListener('click', togglePodcastPlay);
    miniOpen.addEventListener('click', openPodcast);

    // service worker (offline app shell). updateViaCache:'none' stops the
    // browser from serving an old sw.js out of its HTTP cache, so deploys land.
    if ('serviceWorker' in navigator) {
        let hadController = !!navigator.serviceWorker.controller;
        navigator.serviceWorker.addEventListener('controllerchange', function () {
            if (!hadController) { hadController = true; return; }  // first install
            window.location.reload();                             // new SW took over
        });
        window.addEventListener('load', function () {
            navigator.serviceWorker.register('/sw.js', { updateViaCache: 'none' })
                .catch(function () { /* unsupported / file:// */ });
        });
    }

    // init
    renderHome();
    if (typeof indexedDB === 'undefined') {
        showError('This browser cannot store decks (IndexedDB unavailable). Decks will not persist.');
    }
})();
