const Spinners = require('spinnies')
const spinners = new Spinners();
const chalk = require('chalk');
const opts = require('yargs').argv;

class logger{
    constructor(){
        this.tasks={};
    }
    add(name,opts){
        this.tasks[name] = opts;
        console.log(chalk.blue('[started] : ' + (opts && opts.text ? opts.text: name)));
    }
    succeed(name,opts){
        console.log(chalk.green('[success] : ' + (opts && opts.text ? opts.text: this.tasks[name].text)));
    }
    fail(name,opts){
        console.log(chalk.red('[failed] : ' + (opts && opts.text ? opts.text: this.tasks[name].text)));
    }
}

const log = new logger();
module.exports.spinners = opts.info ? log : spinners;

// log.add("first",{text:"first item"});
// log.succeed("first");
// log.fail("first",{text:"couldn't find git"});
 