const spawn = require('cross-spawn');
const opts = require('yargs').argv;

module.exports.spawn = (cmd, args) => spawn(cmd, args, { 
  stdio: opts.info ? 'inherit' : 'ignore', 
  shell:true,
  env:{
    ...process.env,
    GIT_EDITOR:true
  }
});