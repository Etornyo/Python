const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName} = require('import-cwd')('./config.json');
const fs = require('fs');
var commandExists = require('command-exists');

function syncSource(cb) {
  spinners.add('updateFrontend',{text:"Fetching the latest front-end source code"});
  commandExists('git', function(err, commandExists) {
    if(commandExists) {
      fs.promises.access(`${projectName}`)
        .then(()=>{
          var process = spawn("cd",[projectName,"&&","git","fetch","agora","--force","&&","git","rebase","agora/dist"]);
          process.on('exit', () => {
            spinners.succeed('updateFrontend');
            cb();            
          });
          process.on('error',(err)=>{
            spinners.succeed('updateFrontend');
            cb();
          });
        })
        .catch(e => {
            spinners.fail('updateFrontend', { text: 'Couldn\'t find the frontend boilerplate'});
            cb();
        })
    }
    else {
      spinners.fail('updateFrontend', { text: 'Fetching frontend failed since we couldn\'t detect git'});
      cb();
    }
  });
//   process.stdout.on('data', (data) => {
//     console.log(`stdout: ${data}`);
//   });
}
module.exports.syncSource = syncSource;
