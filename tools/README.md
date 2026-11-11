# Analysis scripts

Scripts used to evaluate the saved solver outputs for the seminar paper
*"Smart TV Scheduling in Public Venues: validation of solutions and modelling with constraint programming"*.
Both read `inputs/<instance>.json` and the saved outputs in `outputs/`.

| Script | What it does |
|---|---|
| `score_outputs.js` | Scores `outputs/ilp` and `outputs/rilp` with the validator's own `performValidation` function (loaded from `validator.js`), so the numbers equal what the web tool shows. |
| `paper_model_scorer.py` | An independent scorer written directly from the problem definition in the PATAT 2026 paper. It does not use `validator.js` and is used to cross-check it. |

## Run (from the repository root)

```bash
# 1. Validator-based scoring (needs Node.js and tv-programs-validator cloned next to this repository)
node tools/score_outputs.js
node tools/score_outputs.js ../tv-programs-validator/validator.js croatia kosovo   # explicit path / instances

# 2. Independent scoring (needs Python 3, no extra packages)
python3 tools/paper_model_scorer.py
python3 tools/paper_model_scorer.py croatia
```

Default instances are the four reported in the paper: `croatia`, `kosovo`, `netherlands`, `uk_synthetic`.
Other instances with saved outputs can be passed by name.

## Expected output (paper, Tables 7 and 8)

| Instance | ILP total | RILP valid | RILP total |
|---|---|---|---|
| croatia | 2220 | no | 0 |
| kosovo | 2591 | no | 0 |
| netherlands | 2636 | no | 0 |
| uk_synthetic | 1844 | yes | -1072 |

## Solver

`pip install ortools`, then for example `python ilp.py inputs/croatia.json out.json -t 300 -c 8`
(`-t` time limit in seconds, `-c` number of workers).
