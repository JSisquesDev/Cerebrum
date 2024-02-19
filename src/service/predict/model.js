const dao = require('./dao');
const logger = require('../../util/logger');

//const { CurriculumVitae } = require('../../entities/CurriculumVitae');

module.exports = {
  async predictDetection() {
    logger.info('model.js - Entering predictDetection()');

    // Cargamos el modelo de detección
    const model = '';

    if (!model) {
      logger.error('model.js - Detection model not founded');
      return null;
    }

    const response = 'OK';

    return response;
  },

  async predictClassification() {
    logger.info('model.js - Entering predictClassification()');

    // Cargamos el modelo de classificación
    const model = '';

    if (!model) {
      logger.error('model.js - Classification model not founded');
      return null;
    }

    const response = 'OK';

    return response;
  },

  async predictSegmentation() {
    logger.info('model.js - Entering predictSegmentation()');

    // Cargamos el modelo de segmentación
    const model = '';

    if (!model) {
      logger.error('model.js - Segmentation model not founded');
      return null;
    }

    const response = 'OK';

    return response;
  },
};
