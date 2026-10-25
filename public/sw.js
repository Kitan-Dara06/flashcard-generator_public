/* Service worker — caches the app shell so the app opens and studies fully
 * offline. Never caches /api/* responses (those are POSTs to Vercel anyway).
 * Bump CACHE_VERSION on any change to the shell files to bust the cache.
 */
const CACHE_VERSION = 'study-decks-v1';
const SHELL = ['/', '/index.html', '/style.css', '/script.js', '/db.js'];

self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open(CACHE_VERSION)
            .then(function (cache) { return cache.addAll(SHELL); })
            .then(function () { return self.skipWaiting(); })
    );
});

self.addEventListener('activate', function (event) {
    event.waitUntil(
        caches.keys()
            .then(function (keys) {
                return Promise.all(keys.filter(function (k) { return k !== CACHE_VERSION; })
                    .map(function (k) { return caches.delete(k); }));
            })
            .then(function () { return self.clients.claim(); })
    );
});

self.addEventListener('fetch', function (event) {
    if (event.request.method !== 'GET') return;
    const url = new URL(event.request.url);
    if (url.origin !== location.origin) return;       // leave cross-origin alone
    if (url.pathname.startsWith('/api/')) return;      // never cache API

    event.respondWith(
        caches.match(event.request).then(function (cached) {
            if (cached) return cached;
            return fetch(event.request).then(function (response) {
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_VERSION).then(function (cache) { cache.put(event.request, clone); });
                }
                return response;
            });
        })
    );
});
