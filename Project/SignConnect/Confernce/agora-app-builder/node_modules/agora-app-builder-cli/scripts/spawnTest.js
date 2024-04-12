const {spawn} = require('./spawn');

module.exports.spawn = (cb) => {
    spawn("ls", ['-a']);
    cb();
}