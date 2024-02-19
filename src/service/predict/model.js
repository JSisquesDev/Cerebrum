const dao = require('./dao');
const logger = require('../../util/logger');
const tf = require('@tensorflow');

//const { CurriculumVitae } = require('../../entities/CurriculumVitae');

module.exports = {
  async classifyBird(language) {
    logger.info('model.js - Entering classifyBird()');

    const model = '';

    if (!model) {
      logger.error('model.js - Model not founded');
      return null;
    }

    const response = '';

    //const cu = new CurriculumVitae(personalData);
    return response;
  },
};
