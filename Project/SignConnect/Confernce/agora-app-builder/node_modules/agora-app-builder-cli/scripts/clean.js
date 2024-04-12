const del = require('del');
const {projectName} = require('import-cwd')('./config.json')
const {spinners} =require('./cli');

function clean() {
  spinners.add('clean', {text:'Cleaning the project directory'});
  return del([`${projectName}*`]).then(()=>{spinners.succeed('clean')});
}

module.exports.clean = clean;
