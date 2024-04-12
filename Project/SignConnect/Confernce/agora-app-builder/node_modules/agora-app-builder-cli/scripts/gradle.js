const fs =require('fs').promises;
const {projectName} = require('import-cwd')('./config.json');
const path = require('path');
const {spinners} =require('./cli');

async function modifyGradle(){
  spinners.add('processGradle',{text:"Configuring gradle file"});
  try{
    const gradle = await fs.readFile(`${projectName}/android/build.gradle`,{encoding: 'utf8'});
    const newGradle = gradle.replace("minSdkVersion = 16",`minSdkVersion = 21`);
    await fs.writeFile(path.join(`${projectName}/android/build.gradle`),newGradle , {encoding: "utf8"});
    spinners.succeed('processGradle');
  }
  catch (e){
    spinners.fail('processGradle', { text: `Failed process ${projectName}/android/build.gradle`});
  }
}

module.exports.modifyGradle = modifyGradle;