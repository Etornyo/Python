const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName, logoSquare} = require('import-cwd')('./config.json');

function androidIcons(cb) {
    if(logoSquare === ''){
        cb();
    }
    else{
        var process = spawn("cd",[projectName,"&&","npm","run","icons:android"]);
        spinners.add('androidIcon',{text:"Configuring icons for Android"});
        process.on('exit', () => {
            spinners.succeed('androidIcon');
            cb();
        })
    }
    // process.stdout.on('data', (data) => {
    //   console.log(`stdout: ${data}`);
    // });
}

// create(()=>console.log("finished"),'proj');

module.exports.androidIcons = androidIcons;