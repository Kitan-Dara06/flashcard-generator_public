/* IndexedDB store — the source of truth for decks, cards, star marks, podcast
 * script and cached audio. Works entirely offline.
 *
 * Exposed as window.Store.
 *
 * Audio keys are "<deckId>|<content hash>" so a deck's clips can be cleared when
 * the deck is deleted; the content hash alone would orphan them permanently.
 */
(function () {
    'use strict';

    const DB_NAME = 'study-decks';
    // v2: audio keys gained a "<deckId>|" prefix so deleting a deck can clear its
    // clips. The old bare-hash keys cannot be attributed to any deck, so they are
    // dropped on upgrade rather than left orphaned forever.
    const DB_VERSION = 2;

    const STORES = {
        decks: 'decks',      // keyPath 'id'
        cards: 'cards',      // keyPath 'deckId', value = card array
        podcasts: 'podcasts',// keyPath 'deckId'
        audio: 'audio'       // keyPath 'key', value = Blob
    };

    let dbPromise = null;
    let available = true;

    function open() {
        if (dbPromise) return dbPromise;
        dbPromise = new Promise(function (resolve, reject) {
            let req;
            try {
                req = indexedDB.open(DB_NAME, DB_VERSION);
            } catch (err) {
                available = false;
                reject(err);
                return;
            }
            req.onupgradeneeded = function (e) {
                const db = e.target.result;
                ['decks', 'cards', 'podcasts', 'audio'].forEach(function (name) {
                    if (!db.objectStoreNames.contains(name)) {
                        db.createObjectStore(name, { keyPath: name === 'decks' ? 'id' : (name === 'cards' || name === 'podcasts' ? 'deckId' : 'key') });
                    }
                });
                if (e.oldVersion < 2) {
                    e.target.transaction.objectStore('audio').clear();
                }
            };
            req.onsuccess = function () { resolve(req.result); };
            req.onerror = function () { available = false; reject(req.error); };
        });
        return dbPromise;
    }

    function isAvailable() { return available; }

    function request(db, store, mode, fn) {
        return new Promise(function (resolve, reject) {
            let tx;
            try {
                tx = db.transaction(store, mode);
            } catch (err) {
                reject(err);
                return;
            }
            const objectStore = tx.objectStore(store);
            const req = fn(objectStore);
            if (req && 'onsuccess' in req) {
                req.onsuccess = function () { resolve(req.result); };
            } else {
                resolve();
            }
            tx.onerror = function () { reject(tx.error); };
        });
    }

    function get(db, store, key) {
        return request(db, store, 'readonly', function (s) { return s.get(key); });
    }
    function getAll(db, store) {
        return request(db, store, 'readonly', function (s) { return s.getAll(); });
    }
    function put(db, store, value) {
        return request(db, store, 'readwrite', function (s) { return s.put(value); });
    }
    function del(db, store, key) {
        return request(db, store, 'readwrite', function (s) { return s.delete(key); });
    }

    // ----- public API -----

    async function listDecks() {
        const db = await open();
        const decks = await getAll(db, STORES.decks);
        // newest first; enrich with card + starred counts
        const enriched = [];
        for (const deck of decks) {
            const record = await get(db, STORES.cards, deck.id);
            const cards = (record && record.cards) || [];
            enriched.push({
                id: deck.id,
                title: deck.title,
                sourceName: deck.sourceName,
                createdAt: deck.createdAt,
                provider: deck.provider,
                cardCount: cards.length,
                starredCount: cards.filter(function (c) { return c.starred; }).length
            });
        }
        enriched.sort(function (a, b) { return (b.createdAt || 0) - (a.createdAt || 0); });
        return enriched;
    }

    async function getDeck(id) {
        const db = await open();
        return get(db, STORES.decks, id);
    }

    async function getCards(id) {
        const db = await open();
        const record = await get(db, STORES.cards, id);
        return (record && record.cards) || [];
    }

    async function saveDeck(deck, cards) {
        const db = await open();
        await put(db, STORES.decks, deck);
        await put(db, STORES.cards, { deckId: deck.id, cards: cards });
        return deck.id;
    }

    async function setStarred(deckId, index, starred) {
        const db = await open();
        const record = (await get(db, STORES.cards, deckId)) || { deckId: deckId, cards: [] };
        const cards = record.cards;
        if (cards[index]) {
            cards[index].starred = starred;
            await put(db, STORES.cards, { deckId: deckId, cards: cards });
        }
    }

    async function deleteDeck(id) {
        const db = await open();
        await del(db, STORES.decks, id);
        await del(db, STORES.cards, id);
        await del(db, STORES.podcasts, id);
        // Audio is the biggest store by far (a single segment is ~180 KB) and is
        // keyed by "<deckId>|<hash>", so a deck's clips are reachable by prefix.
        // Without this, deleting a deck leaves its audio orphaned forever.
        await deleteAudioForDeck(db, id);
    }

    async function deleteAudioForDeck(db, deckId) {
        const prefix = deckId + "|";
        const all = await getAll(db, STORES.audio);
        for (const entry of all) {
            if (typeof entry.key === "string" && entry.key.indexOf(prefix) === 0) {
                await del(db, STORES.audio, entry.key);
            }
        }
    }

    // ----- podcast (episodes) -----

    async function getPodcast(deckId) {
        const db = await open();
        return get(db, STORES.podcasts, deckId);
    }

    // Legacy decks stored one script as {title, turns}; the episode model stores
    // {episodes: [...]}. This returns an episodes array either way so callers
    // never have to know which shape they are reading, and marks the record so
    // the UI can offer to convert it rather than hiding that option.
    function normalizeEpisodes(record) {
        if (!record) return [];
        if (Array.isArray(record.episodes)) return record.episodes;
        if (Array.isArray(record.turns) && record.turns.length) {
            return [{ title: record.title || 'Podcast', hook: record.title,
                       turns: record.turns, legacy: true }];
        }
        return [];
    }

    async function savePodcast(deckId, episodes) {
        const db = await open();
        // segs hold in-memory object URLs for audio; they cannot survive a reload
        // and are rebuilt from the audio cache on demand. Store only the script.
        const toStore = episodes.map(function (ep) {
            return {
                title: ep.title,
                hook: ep.hook,
                turns: ep.turns,
                terms: ep.terms || [],
                coverage: ep.coverage
            };
        });
        return put(db, STORES.podcasts, {
            deckId: deckId,
            episodes: toStore,
            generatedAt: Date.now()
        });
    }

    async function getAudio(key) {
        const db = await open();
        return get(db, STORES.audio, key);
    }

    async function putAudio(key, blob, duration) {
        const db = await open();
        return put(db, STORES.audio, { key: key, blob: blob, duration: duration });
    }

    window.Store = {
        open: open,
        isAvailable: isAvailable,
        listDecks: listDecks,
        getDeck: getDeck,
        getCards: getCards,
        saveDeck: saveDeck,
        setStarred: setStarred,
        deleteDeck: deleteDeck,
        getPodcast: getPodcast,
        savePodcast: savePodcast,
        normalizeEpisodes: normalizeEpisodes,
        getAudio: getAudio,
        putAudio: putAudio
    };
})();
