# Optuna Tuning

Script principale:

```bash
python scripts/optuna_script.py
```

## Come Funziona

Per ogni trial Optuna:

- sceglie gli iperparametri dallo search space;
- scrive una config in `results/optuna/<model>/<study>/trial_XXXX/config.yaml`;
- crea `common_resolved.yaml` con path separati per risultati e TensorBoard;
- lancia `train_dqn.py` o `train_ppo.py`;
- legge `mean_reward` dal JSON finale e lo usa come metrica da massimizzare.

Lo study viene salvato in SQLite:

```text
results/optuna/<model>/<study>/study.db
```

## Default Consigliato

Lo script usa gia:

```text
n_trials = 20
timesteps = 500000
```

Quindi il comando base consigliato per DQN e':

```bash
python scripts/optuna_script.py --model dqn --device cpu --study-name dqn_paper_500k
```

Equivalente esplicito:

```bash
python scripts/optuna_script.py --model dqn --n-trials 20 --timesteps 500000 --device cpu --study-name dqn_paper_500k
```

## Altri Comandi

Smoke test DQN piu rapido:

```bash
python scripts/optuna_script.py --model dqn --n-trials 10 --timesteps 250000 --device cpu --study-name dqn_paper_smoke
```

Run DQN piu lunga:

```bash
python scripts/optuna_script.py --model dqn --n-trials 30 --timesteps 500000 --device cpu --study-name dqn_paper_v1
```

PPO con lo stesso budget consigliato:

```bash
python scripts/optuna_script.py --model ppo --n-trials 20 --timesteps 500000 --device cpu --study-name ppo_paper_500k
```

## Riprendere Una Run

Se lo script si interrompe, riprendi lo stesso study con `--resume`:

```bash
python scripts/optuna_script.py --model dqn --study-name dqn_paper_500k --n-trials 20 --timesteps 500000 --device cpu --resume
```

I trial gia completati non vengono rifatti. Un trial interrotto a meta non riparte dal checkpoint: Optuna prosegue con i trial successivi.

## Stime Runtime

Dati storici del progetto:

```text
DQN 1M step: circa 1h56m
PPO 1M step: circa 21m
```

Stime prudenziali:

```text
DQN 20 trial a 500k: circa 20-28 ore
PPO 20 trial a 500k: circa 6-16 ore
```

Nota: DQN puo variare molto per `gradient_steps` e `train_freq`; PPO per `n_steps`, `n_epochs` e `batch_size`.
