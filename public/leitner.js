/* Leitner box arithmetic — pure logic, no DOM, no storage.
 *
 * Box k cards are reviewed every 2^(k-1) days: 1, 2, 4, 8. Four boxes means a
 * card needs four "Got it" answers spread over 15 days to graduate.
 *
 * Exported for node tests (test_leitner.js) and exposed as window.Leitner.
 */
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    } else {
        root.Leitner = api;
    }
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const BOX_INTERVALS_DAYS = [1, 2, 4, 8];
    const MAX_BOX = BOX_INTERVALS_DAYS.length;
    const DAY_MS = 24 * 60 * 60 * 1000;

    // The model's difficulty rating seeds a brand-new card's box.
    function initialBox(difficulty) {
        const d = String(difficulty || '').toLowerCase();
        if (d === 'hard') return 1;
        if (d === 'easy') return 3;
        return 2;
    }

    function nextBox(box, gotIt) {
        if (!gotIt) return 1;
        return Math.min(box + 1, MAX_BOX);
    }

    function intervalMs(box) {
        const clamped = Math.min(Math.max(box, 1), MAX_BOX);
        return BOX_INTERVALS_DAYS[clamped - 1] * DAY_MS;
    }

    function dueAt(box, now) {
        return now + intervalMs(box);
    }

    // A card with no state at all is new, and therefore due.
    function isDue(entry, now) {
        if (!entry || typeof entry.due !== 'number') return true;
        return entry.due <= now;
    }

    function grade(entry, gotIt, now) {
        const base = entry && typeof entry.box === 'number'
            ? entry
            : { box: initialBox(entry && entry.difficulty), seen: 0 };
        const box = nextBox(base.box, gotIt);
        return {
            box: box,
            due: dueAt(box, now),
            seen: (base.seen || 0) + 1
        };
    }

    // Stable identity for a card across reloads: its normalised question text.
    function hashKey(text) {
        const s = String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) {
            h ^= s.charCodeAt(i);
            h = Math.imul(h, 16777619);
        }
        return (h >>> 0).toString(36);
    }

    return {
        BOX_INTERVALS_DAYS: BOX_INTERVALS_DAYS,
        MAX_BOX: MAX_BOX,
        DAY_MS: DAY_MS,
        initialBox: initialBox,
        nextBox: nextBox,
        intervalMs: intervalMs,
        dueAt: dueAt,
        isDue: isDue,
        grade: grade,
        hashKey: hashKey
    };
});
