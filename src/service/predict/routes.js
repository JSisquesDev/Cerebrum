const express = require('express');
const controller = require('./controller');

const router = express.Router();

router.get('/birds', controller.classifyBird);

module.exports = router;
