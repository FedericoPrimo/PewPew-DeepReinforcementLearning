# Space Invaders Deep RL — Milestone 1

Pipeline modulare e riproducibile per esperimenti di Deep Reinforcement Learning su Atari Space Invaders con Gymnasium.

## Struttura del progetto

```
space_invaders_rl/
  configs/
    default.yaml          # Configurazione principale
  src/
    envs/make_env.py      # Creazione ambiente Gymnasium
    preprocessing/
      image_preprocessor.py  # Preprocessing RGB e Grayscale
      masking.py             # Random screen masking
    agents/
      base_agent.py          # Interfaccia base agenti
      dummy_agent.py         # Agente con azioni casuali
    loops/
      run_episode.py         # Esecuzione singolo episodio
      evaluate.py            # Valutazione multi-episodio e multi-seed
    logging/
      metrics_logger.py      # Salvataggio metriche CSV/JSON
    analysis/
      compare_runs.py        # Confronto statistico tra run
      plots.py               # Grafici delle performance
    utils/
      seeding.py             # Gestione seed riproducibili
      config.py              # Caricamento configurazione YAML
  scripts/
    run_dummy.py             # Script principale eseguibile
    compare_results.py       # Script analisi risultati
  tests/
    test_env.py
    test_preprocessing.py
    test_dummy_agent.py
  results/                   # Output CSV/JSON salvati qui
```

## Installazione

```bash
pip install -r requirements.txt
```

> **Nota:** Per usare gli ambienti Atari è necessario installare le ROM. Con `ale-py` (incluso in `gymnasium[atari]`) eseguire:
> ```bash
> ale-import-roms /path/to/roms
> ```
> Oppure installare `autorom`:
> ```bash
> pip install autorom
> autorom --accept-license
> ```

## Utilizzo rapido

```bash
# Run con DummyAgent (configurazione default)
python scripts/run_dummy.py

# Run con override da riga di comando
python scripts/run_dummy.py --config configs/default.yaml --episodes 5 --seeds 1 2 3

# Analisi e confronto risultati
python scripts/compare_results.py --results_dir results/
```

## Configurazione

Tutti i parametri sono in `configs/default.yaml`. I principali:

| Parametro | Descrizione |
|---|---|
| `env.id` | ID ambiente Gymnasium |
| `env.render_mode` | `null` o `human` |
| `preprocessing.mode` | `rgb` o `grayscale` |
| `preprocessing.image_size` | Dimensione resize (es. 84) |
| `preprocessing.frame_stack` | Numero frame da stackare |
| `masking.enabled` | Abilita random masking |
| `evaluation.num_episodes` | Episodi per seed |
| `evaluation.seeds` | Lista seed |

## Eseguire i test

```bash
python -m pytest tests/ -v
```

## Milestone 1 — Acceptance Criteria

- [x] Creazione ambiente Space Invaders con Gymnasium
- [x] Run completa con DummyAgent
- [x] Ciclo observation → preprocessing → action → env.step
- [x] RGB e grayscale producono osservazioni con la stessa shape finale
- [x] Random masking configurabile e opzionale
- [x] Risultati episodi salvati in CSV e JSON
- [x] Script eseguibile da terminale
- [x] Test minimi
- [x] Codice modulare ed estendibile

## Roadmap

- **Milestone 2:** Agente CNN con DQN su input RGB
- **Milestone 3:** Confronto RGB vs Grayscale con analisi statistica
- **Milestone 4:** Esperimenti con random screen masking
