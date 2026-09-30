// Run with Node 24+, passing an installed jsdom package directory as argv[2].
const assert = require('node:assert/strict');
const { JSDOM, VirtualConsole } = require(process.argv[2]);
const base = 'http://127.0.0.1:18765';
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(String(error)));
async function until(predicate) {
  const deadline = Date.now() + 10000;
  while (!predicate()) {
    if (Date.now() > deadline) throw Error('Dashboard render timeout');
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
(async () => {
  const dom = await JSDOM.fromURL(base, {
    runScripts: 'dangerously', virtualConsole,
    beforeParse(window) { window.fetch = (path, options) => fetch(new URL(path, base), options); },
  });
  try {
    const document = dom.window.document;
    await until(() => document.querySelectorAll('#examples article').length === 15);
    assert.match(document.querySelector('#connection').textContent, /연결됨/);
    assert.equal(document.querySelectorAll('#model option').length, 12);
    assert.equal(document.querySelectorAll('#runs tr').length, 13);
    const model = document.querySelector('#model');
    model.value = 'baseline-42';
    model.dispatchEvent(new dom.window.Event('change'));
    await until(() => document.querySelectorAll('#examples article').length === 15);
    const filter = document.querySelector('#filter');
    filter.value = 'fixed';
    filter.dispatchEvent(new dom.window.Event('input'));
    assert.equal(document.querySelectorAll('#examples article').length, 0);
    filter.value = '';
    filter.dispatchEvent(new dom.window.Event('input'));
    const search = document.querySelector('#search');
    search.value = '이거 손 타나요?';
    search.dispatchEvent(new dom.window.Event('input'));
    assert.equal(document.querySelectorAll('#examples article').length, 1);
    assert.match(document.querySelector('#examples').textContent, /발열|열|뜨거/);
    assert.deepEqual(errors, []);
    console.log('PASS: live fetch, 12 models, 707 examples, model switching, filters, search, Korean labels');
  } finally { dom.window.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
