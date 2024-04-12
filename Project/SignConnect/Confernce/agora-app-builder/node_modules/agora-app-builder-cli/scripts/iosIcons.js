const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName, logoSquare} = require('import-cwd')('./config.json');

function iosIcons(cb) {
    if(logoSquare === ''){
        cb();
    }
    else{
        var process = spawn("cd",[projectName,"&&","npm","run","icons:ios"]);
        spinners.add('iosIcon',{text:"Configuring icons for IOS"});
        process.on('exit', () => {
            spinners.succeed('iosIcon');
            cb();
        })
    }
    // process.stdout.on('data', (data) => {
    //   console.log(`stdout: ${data}`);
    // });
}

// create(()=>console.log("finished"),'proj');

module.exports.iosIcons = iosIcons;