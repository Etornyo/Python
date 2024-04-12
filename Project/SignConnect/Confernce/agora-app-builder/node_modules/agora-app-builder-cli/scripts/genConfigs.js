// generates front-end and backend config.json
const config = require('import-cwd')('./config.json');
const datauri = require('datauri');
const fs = require('fs').promises;
const {spinners} =require('./cli');
const url = require('url');
const {logo, background} = require('./images')
const opts = require('yargs').argv;

async function generate (){

    const frontVars= [
        "projectName",
        "displayName",
        "AppID",
        "primaryColor",
        "pstn",
        "precall",
        "watermark",
        "chat",
        "cloudRecording",
        "screenSharing",
        "platformIos",
        "platformAndroid",
        "platformWeb",
        "platformWindows",
        "platformMac",
        "platformLinux",
        "CLIENT_ID",
        "ENABLE_OAUTH",
        "encryption",
    ];
    const backVars= [
        "APP_CERTIFICATE",
        "CUSTOMER_ID",
        "CUSTOMER_CERTIFICATE",
        "BUCKET_NAME",
        "BUCKET_ACCESS_KEY",
        "BUCKET_ACCESS_SECRET",
        "CLIENT_ID",
        "CLIENT_SECRET",
        "PSTN_USERNAME",
        "PSTN_PASSWORD",
        "ENABLE_OAUTH"
    ];

    const frontend = {}
    const backend = {}
    spinners.add('config',{text:"Configuring front-end and backend"});
    let configure = true;
    try{
        await fs.access(`${config.projectName}`)
    }
    catch(e){
        spinners.fail('config', { text: 'couldn\'t find frontend to configure'});
        if(opts.info){
            console.error(e);
        }
        configure = false;
    }
    if(configure){
        try{
            await fs.access(`${config.projectName}Backend`);
        }
        catch(e){
            spinners.fail('config', { text: 'couldn\'t find backend to configure'});
            configure = false;
        }
    }
    
    if(configure){
        try{
            frontVars.map(key => frontend[key] = config[key]);
            backVars.map(key => backend[key] = config[key]);
    
            backend['APP_ID'] = config['AppID'];
            backend['REDIRECT_URL'] = url.resolve(config['backEndURL'], 'oauth');
            backend['SCHEME'] = config['projectName'].toLowerCase();
            frontend['logo'] = (config['logoRect']!=='')? await datauri(config['logoRect']) : logo;
            frontend['illustration'] = (config['illustration']!=='')? await datauri(config['illustration']) : '';
            frontend['bg'] = (config['bg']!=='')? await datauri(config['bg']) : background;
            frontend['frontEndURL'] = (config['frontEndURL'].slice(-1)==='/') ? config['frontEndURL'].slice(0, config['frontEndURL'].length-1):config['frontEndURL'];
            frontend['backEndURL'] = (config['backEndURL'].slice(-1)==='/') ? config['backEndURL'].slice(0, config['backEndURL'].length-1):config['backEndURL']
            
            frontend['landingHeading'] = config['HEADING'];
            frontend['landingSubHeading'] = config['SUBHEADING'];
            const frontendConfigPromise = fs.writeFile(
                `${config['projectName']}/config.json`,
                JSON.stringify(frontend,null,2)
            );
            const backendConfigPromise = fs.writeFile(
                `${config['projectName']}Backend/config.json`,
                JSON.stringify(backend,null,2)
            );
            await Promise.all([frontendConfigPromise, backendConfigPromise]);
            spinners.succeed('config');
        }
        catch(e){
            spinners.fail('config', { text: 'couldn\'t complete configuring'});
            if(opts.info){
                console.error(e);
            }
        }
    }
    return;
}

module.exports.generateConfig = generate;
