const express = require('express');
const bodyParser = require('body-parser');

const app = express();

var cors = require('cors');

// CORS
app.use(cors());
app.use(bodyParser.json());

// Rutas
app.use(require('./src/service/predict/routes.js'));

module.exports = app;
