/* Service worker — offline support that doesn't serve stale code.
 *
 * Strategy: NETWORK-FIRST for the app shell, falling back to cache only when
 * the network fails. A cache-first shell is faster offline but means every
 * deploy is invisible until the cache is manually busted — which is how a fixed
 * db.js can sit behind a stale cached copy long after a redeploy.
 *
 * /api/* is never touched: those are POSTs to Vercel and must always hit the
 * network.
 */
const CACHE_VERSION = 'study-decks-v2';
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
    if (url.origin !== location.origin) return;   // leave cross-origin alone
    if (url.pathname.startsWith('/api/')) return; // API must always be fresh

    event.respondWith(
        fetch(event.request)
            .then(function (response) {
                if (response && response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_VERSION).then(function (cache) {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            })
            .catch(function () {
                // Offline (or server unreachable): serve the last good copy.
                return caches.match(event.request).then(function (cached) {
                    if (cached) return cached;
                    // A navigation to an unknown path should still open the app.
                    if (event.request.mode === 'navigate') return caches.match('/index.html');
                    return new Response('', { status: 504, statusText: 'Offline' });
                });
            })
    );
});
