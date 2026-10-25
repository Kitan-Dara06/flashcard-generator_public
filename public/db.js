/* IndexedDB store — the source of truth for decks, cards, star marks, podcast
 * script and cached audio. Works entirely offline.
 *
 * Exposed as window.Store.
 */
(function () {
    'use strict';

    const DB_NAME = 'study-decks';
    const DB_VERSION = 1;

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
    }

    // ----- podcast (phase 3) -----

    async function getPodcast(deckId) {
        const db = await open();
        return get(db, STORES.podcasts, deckId);
    }

    async function savePodcast(deckId, podcast) {
        const db = await open();
        return put(db, STORES.podcasts, {
            deckId: deckId,
            title: podcast.title,
            turns: podcast.turns,
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
        getAudio: getAudio,
        putAudio: putAudio
    };
})();
