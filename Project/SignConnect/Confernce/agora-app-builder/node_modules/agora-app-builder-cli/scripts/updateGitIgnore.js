const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName} = require('import-cwd')('./config.json');
const fs = require('fs');
var commandExists = require('command-exists');

function updateGitIgnore(cb) {
  spinners.add('update',{text:"Setting up gitignore"});
  commandExists('git', function(err, commandExists) {
    if(commandExists) {
        fs.promises.access(`${projectName}/.gitignore`)
            .then(()=>{
              var process = spawn("cd",[projectName,"&&","git","checkout","--ours",".gitignore","&&","git","add",".gitignore","&&","git","rebase","--continue"]);
              process.on('exit', () => {
                fs.promises.access(`${projectName}/src`)
                  .then(() => {
                    spinners.succeed('update');
                  })
                  .catch(e => {
                    spinners.fail('update', { text: 'Front-end download was unsuccesful'});
                  })
                  .finally(() => {
                    cb();
                  })
              });
              process.on('data',(err)=>{
                  console.log(err);
              });
            })
            .catch(e => {
                spinners.fail('update', { text: 'Couldn\'t find the .gitignore file to update'});
                cb();
            })
    }
    else {
        spinners.fail('backend', { text: 'Initialization failed since we couldn\'t detect git'});
        cb();
    }
  });
//   process.stdout.on('data', (data) => {
//     console.log(`stdout: ${data}`);
//   });
}

// create(()=>console.log("finished"),'proj');

module.exports.updateGitIgnore = updateGitIgnore;
