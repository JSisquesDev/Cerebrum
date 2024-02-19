const model = require('./model');
const logger = require('../../util/logger');
const reqDetective = require('../../util/requestDetective');
const errors = require('../../util/errors');
const httpCodes = require('http-status-codes');

module.exports = {
  async predict(req, res) {
    logger.info('controller.js - Entering predict()');

    const requestData = reqDetective.analize(req);
    logger.debug('controller.js - Request data: ' + JSON.stringify(requestData));

    const body = JSON.stringify(req.body);

    if (body) {
      const error = await errors.createError(
        httpCodes.StatusCodes.BAD_REQUEST,
        httpCodes.ReasonPhrases.BAD_REQUEST,
        'Body not provided',
        'The body bla blabla',
        'a',
      );
      logger.info(`controller.js - ${error}`);
      return res.status(httpCodes.StatusCodes.BAD_REQUEST).send(error);
    }

    logger.info(`controller.js - Request body: ${body}`);

    const detectionPrediction = await model.predictDetection();

    const classificationPrediction = await model.predictClassification();

    const segmentationPrediction = await model.predictSegmentation();

    return res.status(httpCodes.StatusCodes.OK).send();
  },
};
