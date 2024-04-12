const {spawn} = require('./spawn');
const {spinners} =require('./cli');
const {projectName, displayName} = require('import-cwd')('./config.json');
const fs = require('fs');

function create(cb) {
  var process = spawn("npx",["react-native","init",projectName,"--title",`\"${displayName}\"`,"--skip-install","--template","react-native-template-typescript@6.5.7"]);
  spinners.add('create',{text:"Creating front-end boilerplate"});
  process.on('exit', () => {
    fs.promises.access(`${projectName}/package.json`)
    .then(()=>{
      spinners.succeed('create');
    })
    .catch(e=>{
      spinners.fail('create', { text: 'Boilerplate creation was unsuccesful'});
    })
    .finally(()=>{
      cb();
    })
  })
  // process.stdout.on('data', (data) => {
  //   console.log(`stdout: ${data}`);
  // });

}

// create(()=>console.log("finished"),'proj');

module.exports.create = create;
