const {series, parallel} = require('gulp')
const {create} = require('./scripts/create');
const {clean} = require('./scripts/clean');
const {package} =require('./scripts/deps');
const {installDeps} = require('./scripts/installDeps');
const {copyAssets} = require('./scripts/copy');
const {backend} = require('./scripts/backend');
const {generateConfig} = require('./scripts/genConfigs');
const {cleanBoilerplate} = require('./scripts/cleanBoilerplate');
const {initCode} = require('./scripts/initCode');
const {updateGitIgnore} = require('./scripts/updateGitIgnore');
const {syncSource} = require('./scripts/syncSource');
const {initUIKit} = require('./scripts/initUIKit');
const {androidIcons} = require('./scripts/androidIcons');
const {iosIcons} = require('./scripts/iosIcons');
const {processXml} = require('./scripts/processXml');
const {modifyGradle} = require('./scripts/gradle');

module.exports.default = series(
    clean,
    parallel(
        create, // create a boilerplate
        backend
    ),
    generateConfig,
    package,
    installDeps,
    copyAssets,
    cleanBoilerplate,
    initCode,
    syncSource,
    updateGitIgnore,
    initUIKit,
    androidIcons,
    iosIcons,
    processXml,
    modifyGradle
);
// module.exports.default = initCode;
