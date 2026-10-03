// Tiny IndexedDB key/value store. Everything stays in this browser.

const DB_NAME = 'signlens';
const STORE = 'kv';
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(req && req.result);
    t.onerror = () => reject(t.error);
  });
}

export async function get(key) {
  try { return await tx('readonly', (s) => s.get(key)); } catch { return undefined; }
}
export async function set(key, value) {
  try { await tx('readwrite', (s) => s.put(value, key)); return true; } catch { return false; }
}
export async function del(key) {
  try { await tx('readwrite', (s) => s.delete(key)); return true; } catch { return false; }
}

// Small UI preferences go in localStorage, guarded because it can throw.
export function loadPrefs(defaults) {
  try { return { ...defaults, ...JSON.parse(localStorage.getItem('signlens-prefs') || '{}') }; }
  catch { return { ...defaults }; }
}
export function savePrefs(prefs) {
  try { localStorage.setItem('signlens-prefs', JSON.stringify(prefs)); } catch { /* ignore */ }
}
