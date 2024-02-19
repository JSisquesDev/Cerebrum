module.exports = {
  async createError(code, name, message = '', explanation = '', response = '') {
    return JSON.stringify({
      code: code,
      name: name,
      message: message,
      explanation: explanation,
      response: response,
    });
  },
};
