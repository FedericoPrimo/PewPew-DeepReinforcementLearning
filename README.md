# Space Invaders Deep RL

Pipeline modulare e riproducibile per esperimenti di Deep Reinforcement Learning su Atari Space Invaders con Gymnasium.

---

## Struttura del progetto

```
space_invaders_rl/
  configs/
    default.yaml              # Configurazione principale
  src/
    envs/make_env.py          # Creazione ambiente Gymnasium
    models/
      cnn_backbone.py         # CNN estrattore di features (Nature DQN)
    preprocessing/
      image_preprocessor.py  # Preprocessing RGB e Grayscale
      masking.py              # Random screen masking
    agents/
      base_agent.py           # Interfaccia base agenti
      dummy_agent.py          # Agente con azioni casuali (senza CNN)
      cnn_dummy_agent.py      # Agente con forward pass CNN + azioni casuali
    loops/
      run_episode.py          # Esecuzione singolo episodio
      evaluate.py             # Valutazione multi-episodio e multi-seed
    logging/
      metrics_logger.py       # Salvataggio metriche CSV/JSON
    analysis/
      compare_runs.py         # Confronto statistico tra run
      plots.py                # Grafici delle performance
    utils/
      seeding.py              # Gestione seed riproducibili
      config.py               # Caricamento configurazione YAML
  scripts/
    run_dummy.py              # Script principale eseguibile
    compare_results.py        # Script analisi risultati
  tests/
    test_env.py               # Test ambiente Gymnasium
    test_preprocessing.py     # Test preprocessing RGB/Grayscale + masking
    test_dummy_agent.py       # Test DummyAgent e pipeline E2E
    test_cnn.py               # Test CNN backbone e CNNDummyAgent
  results/                    # Output CSV/JSON (generato a runtime)
```

---

## Installazione

```bash
pip install -r requirements.txt
```

Per gli ambienti Atari, installa le ROM:

```bash
autorom --accept-license
```

---

## Utilizzo

### Eseguire una run

**Con CNN (default — Milestone 2):**
```bash
python scripts/run_dummy.py
python scripts/run_dummy.py --agent cnn-dummy
```

**Senza CNN — solo azioni casuali (Milestone 1):**
```bash
python scripts/run_dummy.py --agent dummy
```

**Vedere il gioco a schermo:**
```bash
python scripts/run_dummy.py --render
```

**Opzioni principali:**
```bash
python scripts/run_dummy.py --agent cnn-dummy --episodes 5 --seeds 42 123 456
python scripts/run_dummy.py --agent cnn-dummy --mode grayscale
python scripts/run_dummy.py --agent cnn-dummy --masking
python scripts/run_dummy.py --agent cnn-dummy --render --episodes 3 --seeds 1
python scripts/run_dummy.py --agent dummy --output-format json
```

### Analisi e grafici dei risultati

```bash
python scripts/compare_results.py
python scripts/compare_results.py --results-dir results/ --plot-format pdf
python scripts/compare_results.py --no-plots
```

---

## Configurazione

Tutti i parametri sono in `configs/default.yaml`.

### Ambiente
| Parametro | Default | Descrizione |
|---|---|---|
| `env.id` | `ALE/SpaceInvaders-v5` | ID ambiente Gymnasium |
| `env.render_mode` | `null` | `null` o `human` |

### Preprocessing
| Parametro | Default | Descrizione |
|---|---|---|
| `preprocessing.mode` | `rgb` | `rgb` o `grayscale` |
| `preprocessing.image_size` | `84` | Dimensione resize |
| `preprocessing.frame_stack` | `1` | Frame da stackare (1 = nessuno) |
| `preprocessing.normalize` | `true` | Normalizza pixel in [0, 1] |

### CNN
| Parametro | Default | Descrizione |
|---|---|---|
| `model.feature_dim` | `512` | Dimensione vettore di features in uscita |
| `model.device` | `cpu` | `cpu`, `cuda` o `mps` |
| `model.conv_layers` | Nature DQN | Lista layer convoluzionali (modificabile) |

Architettura CNN default (Nature DQN, Mnih et al. 2015):
```yaml
model:
  feature_dim: 512
  conv_layers:
    - out_channels: 32
      kernel_size: 8
      stride: 4
    - out_channels: 64
      kernel_size: 4
      stride: 2
    - out_channels: 64
      kernel_size: 3
      stride: 1
```

### Masking
| Parametro | Default | Descrizione |
|---|---|---|
| `masking.enabled` | `false` | Abilita random screen masking |
| `masking.num_masks` | `2` | Numero di maschere per frame |
| `masking.mask_size` | `[20, 20]` | Dimensione di ogni maschera |
| `masking.mask_duration` | `5` | Durata in frame |
| `masking.apply_probability` | `0.3` | Probabilità di applicazione |
| `masking.mask_value` | `zero` | `zero` o `mean` |

### Valutazione
| Parametro | Default | Descrizione |
|---|---|---|
| `evaluation.num_episodes` | `10` | Episodi per seed |
| `evaluation.seeds` | `[42, 123, 456, 789, 1000]` | Lista seed |
| `evaluation.results_format` | `csv` | `csv` o `json` |

---

## Architettura CNN

La CNN è un modulo indipendente dall'agente (`src/models/cnn_backbone.py`). Riceve l'osservazione preprocessata e produce un vettore di features:

```
Input: (B, 3, 84, 84)  float32  valori in [0, 1]
  → Conv2d(3→32,  kernel=8, stride=4) + ReLU
  → Conv2d(32→64, kernel=4, stride=2) + ReLU
  → Conv2d(64→64, kernel=3, stride=1) + ReLU
  → Flatten
  → Linear → (B, 512)
Output: (B, 512)  float32
```

Il `CNNDummyAgent` esegue il forward pass e salva le features in `agent.last_features`, ma sceglie ancora azioni casuali. Gli agenti delle milestone successive (DQN, PPO) useranno queste features per imparare una policy.

---

## Eseguire i test

```bash
# Tutti i test
python -m pytest tests/ -v

# Solo CNN (Milestone 2)
python -m pytest tests/test_cnn.py -v

# Solo preprocessing
python -m pytest tests/test_preprocessing.py -v

# Solo DummyAgent e pipeline
python -m pytest tests/test_dummy_agent.py -v

# Solo ambiente (richiede ALE installato)
python -m pytest tests/test_env.py -v
```

### Copertura test

| File | Test | Cosa verifica |
|---|---|---|
| `test_cnn.py` | 17 | CNN forward pass, shape, dtype, NaN, CNNDummyAgent, pipeline E2E |
| `test_preprocessing.py` | 14 | RGB/Grayscale shape, normalizzazione, frame stacking, masking |
| `test_dummy_agent.py` | 12 | DummyAgent, riproducibilità, pipeline E2E RGB e Grayscale |
| `test_env.py` | 4 | Creazione ambiente, reset, step, action space |

---

## Roadmap

| Milestone | Stato | Descrizione |
|---|---|---|
| 1 | ✅ | Pipeline + DummyAgent + preprocessing + masking |
| 2 | ✅ | CNN backbone (Nature DQN) + CNNDummyAgent |
| 3 | 🔜 | Agente DQN che impara a giocare |
| 4 | 🔜 | Confronto RGB vs Grayscale con analisi statistica |
| 5 | 🔜 | Esperimenti con random screen masking |