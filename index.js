require('dotenv').config();

const app = require('./app');
const logger = require('./src/util/logger.js');

const port = process.env.BACKEND_PORT;

app.listen(port, () => {
  logger.info(`Server initialized, runnig at port ${port}`);
});
