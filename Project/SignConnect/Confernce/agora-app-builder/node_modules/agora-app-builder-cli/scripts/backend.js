const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName} = require('import-cwd')('./config.json');
const fs = require('fs');

function backend(cb) {
  var process = spawn("git",["clone","https://github.com/samyak-jain/AgoraBackend.git",`${projectName}Backend`]);
  spinners.add('backend',{text:"Downloading backend"});
  process.on('exit', () => {
    fs.promises.access(`${projectName}Backend`)
      .then(() => {
        spinners.succeed('backend');
      })
      .catch(e => {
        spinners.fail('backend', { text: 'Backend download was unsuccesful'});
      })
      .finally(() => {
        cb();
      })
  });
}

module.exports.backend = backend;
