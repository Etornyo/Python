const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName} = require('import-cwd')('./config.json');
const fs = require('fs');
var commandExists = require('command-exists');

function uikitInit(cb) {
    spinners.add('uikit',{text:"Downloading & configuring uikit"});
    commandExists('git', function(err, commandExists) {
        if(commandExists) {
            fs.promises.access(`${projectName}`)
                .then(()=>{
                    var process = spawn("cd",[projectName,"&&","git","clone","https://github.com/AgoraIO-Community/ReactNative-UIKit.git","agora-rn-uikit","&&","cd","agora-rn-uikit","&&","git","checkout","app-builder"]);
                    process.on('exit', () => {
                        fs.promises.access(`${projectName}/agora-rn-uikit`)
                            .then(() => {
                                spinners.succeed('uikit');
                            })
                            .catch(e => {
                                spinners.fail('uikit', { text: 'UIKit download was unsuccesful'});
                            })
                            .finally(() => {
                                cb();
                            })
                    })
                })
                .catch(e => {
                    spinners.fail('uikit', { text: 'Couldn\'t find the frontend boilerplate'});
                    cb();
                })
        }
        else {
          spinners.fail('uikit', { text: 'Fetching failed since we couldn\'t detect git'});
          cb();
        }
    });
    // process.stdout.on('data', (data) => {
    //   console.log(`stdout: ${data}`);
    // });
}

// create(()=>console.log("finished"),'proj');

module.exports.initUIKit = uikitInit;