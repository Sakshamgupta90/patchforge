const express = require('express');

const app = express();
const items = new Map([['1', 'one'], ['2', 'two']]);

app.get('/items/:id', (req, res) => {
  const name = items.get(req.param('id'));
  if (!name) return res.sendStatus(404);
  res.json({ id: req.params.id, name });
});

app.del('/items/:id', (req, res) => {
  items.delete(req.params.id);
  res.send(204);
});

app.get('/files/*', (req, res) => {
  res.json({ path: req.params[0] });
});

module.exports = app;
