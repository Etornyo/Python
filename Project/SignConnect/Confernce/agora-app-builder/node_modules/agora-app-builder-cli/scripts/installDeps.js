const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName} = require('import-cwd')('./config.json');
const fs = require('fs');
const os = require('os');
const del = require('del');

function installDeps(cb) {
  if(os.platform() === 'win32'){
    spinners.add('npmrc',{text:"Adding npmrc for 32-bit config"});
    fs.writeFile(`${projectName}/.npmrc`, 'arch=ia32', {encoding:'utf8'},(err)=>{
      if(err){
        spinners.fail('npmrc', { text: 'npmrc could not be added'});
      }
      else{
        spinners.succeed('npmrc');
        var process = spawn("cd",[projectName,"&&","npm","install"]);
        spinners.add('installDeps',{text:"Installing dependencies"});
        process.on('exit', () => {
          fs.promises.access(`${projectName}/node_modules`)
          .then(()=>{
            spinners.succeed('installDeps');
            spinners.add('npmrc2',{text:"Removing npmrc config"});
            del([`${projectName}/.npmrc`]).then(()=>{
              spinners.succeed('npmrc2');
              spinners.add('image',{text:"Installing 64bit image processing library"});
              var process2 = spawn(`cd ${projectName} && npm install @bam.tech/react-native-make@3.0.0`,{shell: true});
              process2.on('exit', () => {
                fs.promises.access(`${projectName}/node_modules/@bam.tech/react-native-make`)
                  .then(()=>{
                    spinners.succeed('image');
                  })
                  .catch(e=>{
                    spinners.fail('image', { text: 'Image processing lib was not installed succesfully'});
                  })
                  .finally(()=>{
                    cb();
                  })
              });
            }).catch(()=>{
              spinners.fail('npmrc2', { text: 'Could not remove npmrc'});
              cb();
            });
          })
          .catch(e=>{
            console.error(e);
            spinners.fail('installDeps', { text: 'Dependencies were not installed succesfully'});
            cb();
          })
        });
      }
    })
  }
  else{
    var process = spawn(`cd ${projectName} && npm install`,{shell: true});
    spinners.add('installDeps',{text:"Installing dependencies"});
    process.on('exit', () => {
      fs.promises.access(`${projectName}/node_modules`)
      .then(()=>{
        spinners.succeed('installDeps');
      })
      .catch(e=>{
        spinners.fail('installDeps', { text: 'Dependencies were not installed succesfully'});
      })
      .finally(()=>{
        cb();
      })
    });
  }
  // process.stdout.on('data', (data) => {
  //   console.log(`stdout: ${data}`);
  // });
}

// create(()=>console.log("finished"),'proj');

module.exports.installDeps = installDeps;
