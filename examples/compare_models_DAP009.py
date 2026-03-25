"""Fit and compare all built-in psytrax models on mouse DAP009.

Fits four models on the first N_TRIALS trials:
  1. Logistic regression  (choice only, K=2)
  2. DDM                  (choice + RT, K=4)
  3. Race model           (choice + RT, K=6)
  4. MLP                  (choice only, K=13)

Results are saved to fits/ and compared by log evidence.
Run the Streamlit app afterwards to visualise:
  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from itertools import groupby

import psytrax
import psytrax.models.logistic  as logistic_model
import psytrax.models.ddm       as ddm_model
import psytrax.models.race      as race_model
import psytrax.models.mlp       as mlp_model

CSV      = os.path.join(os.path.dirname(__file__),
           '../../learning_race_model/mouse_data/empirical_behav_normalised.csv')
MOUSE    = 'DAP009'
N_TRIALS = 500
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'fits')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mouse(csv_path, mouse_name):
    df    = pd.read_csv(csv_path)
    names = np.array([df['expRef'][i][-6:] for i in range(len(df))])
    idx   = np.where(names == mouse_name)[0]
    sub   = df.iloc[idx].copy()

    mask = (sub['repeatNumber'] > 1) | (sub['trialNumber'] <= 5)
    sub  = sub[~mask]
    sub  = sub[sub['choice'] != 'NoGo']

    c = (sub['contrastRight'] - sub['contrastLeft']).to_numpy()
    T = (sub['choiceCompleteTime'] - sub['stimulusOnsetTime']).to_numpy()
    r = sub['choice'].map({'Right': 1.0, 'Left': 0.0}).to_numpy()

    sess             = np.array([s[:12] for s in sub['expRef'].to_numpy()])
    session_lengths  = np.array([sum(1 for _ in g) for _, g in groupby(sess)])
    boundaries       = np.cumsum(session_lengths, dtype=int)

    first_contrasts  = np.unique(c[:boundaries[0]])
    max_sessions     = 1
    for i in range(len(boundaries) - 1):
        if not np.array_equal(np.unique(c[boundaries[i]:boundaries[i+1]]), first_contrasts):
            break
        max_sessions += 1

    end = boundaries[max_sessions - 1]
    return {
        'inputs':          {'c': c[:end]},
        'responses':        r[:end],
        'times':            T[:end],
        'session_lengths':  session_lengths[:max_sessions],
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    'logistic': dict(
        module      = logistic_model,
        needs_rt    = False,
    ),
    'ddm': dict(
        module      = ddm_model,
        needs_rt    = True,
    ),
    'race': dict(
        module      = race_model,
        needs_rt    = True,
    ),
    'mlp': dict(
        module      = mlp_model,
        needs_rt    = False,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Loading {MOUSE}...')
    data = load_mouse(CSV, MOUSE)
    N    = min(N_TRIALS, len(data['responses']))
    print(f'Using first {N} trials.\n')

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for name, cfg in MODELS.items():
        mod = cfg['module']
        E0  = mod.default_E0(N)

        print(f'{"─"*50}')
        print(f'Fitting: {name.upper()}  (K={mod.N_PARAMS})')
        print(f'{"─"*50}')

        path = psytrax.fit(
            data               = data,
            log_lik_trial      = mod.log_lik_trial,
            n_params           = mod.N_PARAMS,
            param_names        = mod.PARAM_NAMES,
            hyper              = mod.default_hyper(),
            E0                 = E0,
            n_trials           = N,
            session_boundaries = True,
            device             = 'auto',
            subject_name       = f'{MOUSE}_{name}',
            save               = True,
        )
        results[name] = np.load(path, allow_pickle=True).item()
        print(f'  → log evidence: {results[name]["log_evidence"]:.2f}')
        print(f'  → duration:     {results[name]["duration"]}\n')

    # --- Summary table ---
    print(f'\n{"═"*50}')
    print(f'{"Model":<12}  {"K":>4}  {"Log evidence":>14}')
    print(f'{"─"*50}')
    for name, res in results.items():
        K = res['params'].shape[0]
        print(f'{name:<12}  {K:>4}  {res["log_evidence"]:>14.2f}')
    print(f'{"═"*50}')
    print(f'\nSaved to {OUT_DIR}/')
    print('Visualise with:  streamlit run app.py')
