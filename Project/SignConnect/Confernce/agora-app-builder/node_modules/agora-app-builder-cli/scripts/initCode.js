const {spawn} = require('./spawn');
const {projectName} = require('import-cwd')('./config.json')
const {spinners} =require('./cli');
const fs = require('fs');
var commandExists = require('command-exists');

const init = function(cb){
    spinners.add('initBoiler', {text:'Initializing frontend'});
    commandExists('git', function(err, commandExists) {
        if(commandExists) {
            fs.promises.access(`${projectName}`)
                .then(()=>{
                    var process = spawn("cd",[projectName,"&&","git","init","&&","git","add",".","&&","git","commit","-m","init","&&","git","remote","add","agora","https://github.com/AgoraIO-Community/app-builder-core.git"]);
                    process.on('exit', () => {
                        spinners.succeed('initBoiler');
                        cb();
                    });
                    // process.on('data',(err)=>{
                    //     console.log(err);
                    // });
                })
                .catch(e => {
                    spinners.fail('initBoiler', { text: 'Couldn\'t find the frontend to initialize'});
                    cb();
                })
        }
        else {
            spinners.fail('backend', { text: 'Initialzation failed since we couldn\'t detect git'});
            cb();
        }
    });
}

module.exports.initCode = init;