const model = require('./model');
const logger = require('../../util/logger');
const reqDetective = require('../../util/requestDetective');

module.exports = {
  async classifyBird(req, res) {
    logger.info('controller.js - Entering classifyBird()');

    const requestData = reqDetective.analize(req);

    logger.debug('controller.js - Request data: ' + JSON.stringify(requestData));

    var result = await model.classifyBird();

    if (!result) {
      logger.error('controller.js - Bird not obtained: ');
      return res.status(400).send('');
    }

    logger.debug('controller.js - Response: ' + JSON.stringify(result));

    return res.status(200).send(result);
  },
};
