const test = require('node:test');
const assert = require('node:assert');
const request = require('supertest');
const app = require('../app');

test('reads an item', async () => {
  const res = await request(app).get('/items/1');
  assert.deepStrictEqual(res.body, { id: '1', name: 'one' });
});

test('deletes an item', async () => {
  assert.strictEqual((await request(app).delete('/items/2')).status, 204);
  assert.strictEqual((await request(app).get('/items/2')).status, 404);
});

test('serves nested file paths', async () => {
  const res = await request(app).get('/files/a/b.txt');
  assert.strictEqual(res.body.path, 'a/b.txt');
});
