# Space Invaders Deep RL

Progetto di Deep Reinforcement Learning su Atari `SpaceInvaders` con due pipeline di training reali:

- `DQN` value-based con replay buffer e target network
- `PPO` actor-critic on-policy con rollout buffer e GAE

La codebase attuale e' centrata su queste due pipeline. Gli agenti `Dummy` non fanno piu' parte del progetto attivo.

## Stato del progetto

- Pipeline attiva: `DQN` e `PPO`
- Ambiente target: `ALE/SpaceInvaders-v5`
- Pipeline visiva unificata tra `DQN` e `PPO`
- Input alla CNN: `(3 * frame_stack) x 84 x 84`
- Logging: TensorBoard
- Analisi risultati: script in `src/analysis/` e `scripts/compare_results.py`

## Struttura del progetto

```text
space_invaders_rl/
  configs/
    config_common.yaml        # seed, env, eval, logging
    config_dqn.yaml           # iperparametri DQN
    config_ppo.yaml           # iperparametri PPO
    default.yaml              # configurazione generale / sperimentale
  scripts/
    train_dqn.py              # training DQN
    train_ppo.py              # training PPO
    compare_results.py        # analisi statistica dei risultati
    quick_test.sh             # smoke test rapido
  src/
    agents/
      base_agent.py
      dqn_agent.py
      ppo_agent.py
    envs/
      make_env.py
      atari_wrappers.py
      vec_env.py
    models/
      cnn_backbone.py
      actor_critic.py
    preprocessing/
      image_preprocessor.py   # pipeline visiva unificata
      masking.py
    training/
      replay_buffer.py
      rollout_buffer.py
      tb_logger.py
    analysis/
      compare_runs.py
      plots.py
    utils/
      config.py
      seeding.py
  tests/
  logs/
  results/
```

## Installazione

```bash
pip install -r requirements.txt
```

Per gli ambienti Atari:

```bash
autorom --accept-license
```

## Pipeline osservazioni

La pipeline di training per `DQN` e `PPO` usa lo stesso preprocessing visivo:

```text
frame RGB grezzo Atari
  -> resize 84x84
  -> rgb oppure grayscale
  -> se grayscale: replica su 3 canali
  -> frame stack di N frame
  -> masking opzionale
  -> reward clipping
  -> tensore finale (3 * N, 84, 84)
```

La logica `frame grezzo -> input CNN` e' centralizzata in [src/preprocessing/image_preprocessor.py](/Users/mattiasegreto/Desktop/space_invaders_rl/src/preprocessing/image_preprocessor.py) ed e' condivisa da entrambi gli algoritmi.

## Architettura

### DQN

Pipeline principale:

```text
(3 * frame_stack, 84, 84)
  -> CNNBackbone(in_channels=3 * frame_stack)
  -> Linear head
  -> Q-values per azione
```

Caratteristiche:

- replay buffer uniforme
- target network con hard update
- epsilon-greedy
- loss Huber
- gradient clipping

### PPO

Pipeline principale:

```text
(3 * frame_stack, 84, 84)
  -> CNNBackbone(in_channels=3 * frame_stack)
  -> policy head
  -> value head
```

Caratteristiche:

- actor-critic con backbone condiviso
- rollout buffer on-policy
- GAE
- PPO-Clip
- env vettorizzati paralleli

## Esecuzione training

### DQN

```bash
python scripts/train_dqn.py
python scripts/train_dqn.py --timesteps 1000 --device cpu
python scripts/train_dqn.py --config configs/config_dqn.yaml --common configs/config_common.yaml
```

### PPO

```bash
python scripts/train_ppo.py
python scripts/train_ppo.py --timesteps 10000 --device cpu
python scripts/train_ppo.py --config configs/config_ppo.yaml --common configs/config_common.yaml
```

## Configurazione

Le configurazioni usate dal training attuale sono:

- `configs/config_common.yaml`
- `configs/config_dqn.yaml`
- `configs/config_ppo.yaml`

`configs/default.yaml` resta disponibile come file di configurazione generale per moduli standalone e sperimentazione, ma non e' il file principale usato dagli script di training correnti.

### Config comune

In `config_common.yaml` trovi:

- `seed`
- `env.id`
- `preprocessing.mode`
- `preprocessing.image_size`
- `preprocessing.frame_stack`
- `preprocessing.normalize`
- `masking.enabled`
- `masking.num_masks`
- `masking.mask_size`
- `masking.mask_duration`
- `masking.apply_probability`
- `masking.mask_value`
- `eval.episodes`
- `eval.results_dir`
- `logging.tensorboard_dir`

### Config DQN

In `config_dqn.yaml` trovi i principali iperparametri:

- `learning_rate`
- `gamma`
- `batch_size`
- `replay_buffer_size`
- `target_update_freq`
- `epsilon_start`
- `epsilon_end`
- `epsilon_decay_steps`
- `learning_starts`
- `train_freq`
- `total_timesteps`

### Config PPO

In `config_ppo.yaml` trovi i principali iperparametri:

- `n_envs`
- `n_steps`
- `n_epochs`
- `batch_size`
- `learning_rate`
- `lr_schedule`
- `clip_range`
- `gamma`
- `gae_lambda`
- `ent_coef`
- `vf_coef`
- `total_timesteps`

## TensorBoard

Per monitorare il training:

```bash
tensorboard --logdir logs/ --port 6006
```

Poi apri `http://localhost:6006`.

## Risultati e analisi

I trainer salvano:

- log TensorBoard in `logs/dqn` e `logs/ppo`
- modelli in `results/`
- file risultati in `results/`

Per l'analisi:

```bash
python scripts/compare_results.py
python scripts/compare_results.py --results-dir results/ --plot-format pdf
python scripts/compare_results.py --no-plots
```

Nota: la parte di analisi e' nata inizialmente sui risultati episodici in CSV. Se usi output riassuntivi diversi, puo' servire uniformare il formato prima del confronto.

## Test

Smoke test senza training:

```bash
pytest -q tests/test_preprocessing.py tests/test_visual_pipeline.py
```

Questi test verificano:

- preprocessing `rgb` e `grayscale`
- `frame_stack` configurabile
- masking opzionale
- forward di `DQNAgent` e `PPOAgent` senza addestramento

Per i test che richiedono Atari/Gymnasium completo, installa tutte le dipendenze e le ROM prima di eseguirli.

## Roadmap tecnica

- riallineare gli import di package e i test automatici
- uniformare meglio il formato dei risultati per il confronto tra run
