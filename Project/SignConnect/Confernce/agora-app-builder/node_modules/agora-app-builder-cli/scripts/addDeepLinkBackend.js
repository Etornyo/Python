const fs =require('fs').promises;
const {projectName} = require('import-cwd')('./config.json');
const path = require('path');
const {spinners} =require('./cli');
const opts = require('yargs').argv;

async function process(){
  spinners.add('processManifest',{text:"configuring deep links on the backend"});
  try{
    const html = await fs.readFile(`${projectName}Backend/web/mobile.html`,{encoding: 'utf8'});
    const newHtml = html.replace("my-scheme://my-host/",`${projectName}://my-host/`);
    await fs.writeFile(path.join(`${projectName}Backend/web/mobile.html`),newHtml , {encoding: "utf8"});
    spinners.succeed('processManifest');
  }
  catch (e){
    spinners.fail('processManifest', { text: 'Failed process backend templates to add deeplinks'});
    if(opts.info){
      console.error(e);
    }
  }
}

module.exports.backendDeep = process;
