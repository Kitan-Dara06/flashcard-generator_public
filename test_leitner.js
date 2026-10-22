// Box arithmetic tests for the Leitner scheduler.
//   node test_leitner.js
const assert = require('assert');
const Leitner = require('./public/leitner.js');

let failures = 0;
function check(name, fn) {
    try {
        fn();
        console.log('  ok   ' + name);
    } catch (err) {
        failures += 1;
        console.log('  FAIL ' + name + ' -> ' + err.message);
    }
}

const NOW = Date.UTC(2026, 0, 1, 0, 0, 0);
const days = (n) => n * Leitner.DAY_MS;

check('difficulty seeds the starting box', () => {
    assert.strictEqual(Leitner.initialBox('hard'), 1);
    assert.strictEqual(Leitner.initialBox('medium'), 2);
    assert.strictEqual(Leitner.initialBox('easy'), 3);
    assert.strictEqual(Leitner.initialBox(''), 2);       // unknown -> middle
    assert.strictEqual(Leitner.initialBox('EASY'), 3);   // case insensitive
});

check('Got it promotes, capped at box 4', () => {
    assert.strictEqual(Leitner.nextBox(1, true), 2);
    assert.strictEqual(Leitner.nextBox(3, true), 4);
    assert.strictEqual(Leitner.nextBox(4, true), 4);
});

check('Again resets to box 1 from anywhere', () => {
    assert.strictEqual(Leitner.nextBox(1, false), 1);
    assert.strictEqual(Leitner.nextBox(3, false), 1);
    assert.strictEqual(Leitner.nextBox(4, false), 1);
});

check('intervals double per box', () => {
    assert.strictEqual(Leitner.intervalMs(1), days(1));
    assert.strictEqual(Leitner.intervalMs(2), days(2));
    assert.strictEqual(Leitner.intervalMs(3), days(4));
    assert.strictEqual(Leitner.intervalMs(4), days(8));
});

check('grade schedules the next review', () => {
    const first = Leitner.grade({ box: 2, seen: 0 }, true, NOW);
    assert.strictEqual(first.box, 3);
    assert.strictEqual(first.due, NOW + days(4));
    assert.strictEqual(first.seen, 1);

    const again = Leitner.grade(first, false, NOW);
    assert.strictEqual(again.box, 1);
    assert.strictEqual(again.due, NOW + days(1));
    assert.strictEqual(again.seen, 2);
});

check('a card is due when its date has passed', () => {
    const entry = { box: 1, due: NOW + days(1), seen: 1 };
    assert.strictEqual(Leitner.isDue(entry, NOW), false);
    assert.strictEqual(Leitner.isDue(entry, NOW + days(1)), true);   // boundary
    assert.strictEqual(Leitner.isDue(entry, NOW + days(2)), true);
});

check('cards with no state are due immediately', () => {
    assert.strictEqual(Leitner.isDue(undefined, NOW), true);
    assert.strictEqual(Leitner.isDue(null, NOW), true);
    assert.strictEqual(Leitner.isDue({}, NOW), true);
});

check('hashKey is stable and distinguishes questions', () => {
    const a = Leitner.hashKey('What year was SIWES established?');
    const b = Leitner.hashKey('what year was siwes established');
    const c = Leitner.hashKey('Who funded SIWES?');
    assert.strictEqual(a, b);          // punctuation/case insensitive
    assert.notStrictEqual(a, c);
    assert.ok(typeof Leitner.hashKey('') === 'string' && Leitner.hashKey('').length > 0);
});

check('fifteen-day graduation path', () => {
    // hard card starts at box 1; four Got-its over 1+2+4+8 days graduates it
    let entry = { box: Leitner.initialBox('hard'), seen: 0 };
    let when = NOW;
    const gaps = [];
    for (let i = 0; i < 4; i++) {
        entry = Leitner.grade(entry, true, when);
        gaps.push(Math.round((entry.due - when) / Leitner.DAY_MS));
        when = entry.due;
    }
    assert.deepStrictEqual(gaps, [2, 4, 8, 8]);
    assert.strictEqual(entry.box, 4);
    assert.strictEqual(entry.seen, 4);
});

console.log(failures ? `\nFAILED (${failures})` : '\nPASS: Leitner scheduling behaves');
process.exit(failures ? 1 : 0);
