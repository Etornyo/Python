const fs = require('fs').promises
const {spinners} =require('./cli');
const {projectName, logoRect, logoSquare} = require('import-cwd')('./config.json');
const path = require('path');
const opts = require('yargs').argv;


async function copy() {
  spinners.add('copy',{text:"Copying icon"});
  try {
    await fs.mkdir(`${projectName}/build`, {recursive:true});
    // if(logoRect!== ''){
    //   var rectCopy = fs.copyFile(logoRect, `${projectName}/build/${logoRect}`);
    // }
    if(logoSquare!== ''){
      var squareCopy = fs.copyFile(
        path.join(process.cwd(),logoSquare), 
        path.join(process.cwd(),`${projectName}/build/icon.png`)
      );
      await squareCopy;
      spinners.succeed('copy');
    }
    else{
      spinners.fail('copy', { text: 'No icon was specified in the config file'});
    }
  }
  catch(e){
    if(opts.info){
      console.error(e);
    }
    spinners.fail('copy', { text: 'Couldn\'t copy the icon'});
  }
  return;
}

module.exports.copyAssets = copy;
