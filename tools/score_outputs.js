// Scores the saved solver outputs (outputs/ilp and outputs/rilp) with the validator's own
// scoring function (performValidation from validator.js), so the numbers match the web tool.
//
// Usage (from the repository root):
//   node tools/score_outputs.js [path/to/validator.js] [instance ...]
//
// Defaults:
//   validator.js : ../tv-programs-validator/validator.js  (the two repositories side by side)
//   instances    : croatia kosovo netherlands uk_synthetic (the four reported in the seminar paper)
const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const validatorPath = path.resolve(args[0] && args[0].endsWith('.js') ? args.shift() : path.join(repo, '..', 'tv-programs-validator', 'validator.js'));
const instances = args.length ? args : ['croatia', 'kosovo', 'netherlands', 'uk_synthetic'];

if (!fs.existsSync(validatorPath)) {
  console.error('validator.js not found at ' + validatorPath + '\nClone tv-programs-validator next to this repository or pass the path as the first argument.');
  process.exit(1);
}

// Load only validateInputData + performValidation + formatTime from validator.js
// (the rest of that file uses the DOM and cannot run outside a browser).
const src = fs.readFileSync(validatorPath, 'utf8').split('\n');
const start = src.findIndex(l => l.startsWith('function validateInputData'));
const end = src.findIndex(l => l.startsWith('function displayResults'));
const ft = src.findIndex(l => l.startsWith('function formatTime'));
if (start < 0 || end < 0 || ft < 0) { console.error('Unexpected validator.js layout.'); process.exit(1); }
const code = src.slice(ft, ft + 6).join('\n') + '\n' + src.slice(start, end).join('\n') + '\nreturn { performValidation };';
const { performValidation } = new Function(code)();

for (const name of instances) {
  const input = JSON.parse(fs.readFileSync(path.join(repo, 'inputs', `${name}.json`)));
  for (const kind of ['ilp', 'rilp']) {
    const output = JSON.parse(fs.readFileSync(path.join(repo, 'outputs', kind, `${name}.json`)));
    const r = performValidation(input, output);
    const s = r.score;
    console.log(
      `${name.padEnd(13)} ${kind.padEnd(5)} valid=${String(r.valid).padEnd(5)} total=${String(s.total).padStart(6)} ` +
      `base=${s.base} bonus=${s.bonus} switches=${s.channelSwitches} early=${s.earlyTerminations} late=${s.lateStarts} violations=${r.violations.length}`
    );
  }
}
