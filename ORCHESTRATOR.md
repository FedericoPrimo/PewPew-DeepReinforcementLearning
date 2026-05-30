# Orchestrator

`scripts/orchestrator.py` serve per lanciare una serie di run di training a partire da una cartella di configurazioni YAML.

L'idea e' semplice:
- scegli il modello: `dqn` oppure `ppo`
- indichi una cartella dentro `configs/` che contiene piu' file `.yaml`
- l'orchestrator esegue tutti i file trovati, uno dopo l'altro

Questa guida spiega:
1. come funziona
2. come si usa
3. quali parametri accetta

## 1. Come Funziona

L'orchestrator:
- legge tutti i file `.yaml` presenti nella cartella indicata con `--grid-dir`
- ordina i file in modo stabile
- per ogni config costruisce una run separata
- lancia lo script corretto:
  - `scripts/train_dqn.py` se `--model dqn`
  - `scripts/train_ppo.py` se `--model ppo`

Per ogni config, l'orchestrator crea una sottocartella dedicata sia per i risultati sia per i log TensorBoard.

Struttura tipica:

```text
results/sweeps/<model>/<grid_name>/<indice_nome_config>/
logs/sweeps/<model>/<grid_name>/<indice_nome_config>/
```

Esempio per DQN:

```text
results/sweeps/dqn/grid_dqn/01_dqn_baseline/
logs/sweeps/dqn/grid_dqn/01_dqn_baseline/
```

Dentro la cartella `results` di ogni run trovi normalmente:
- `common_resolved.yaml`
- `dqn_model.pth` oppure `ppo_model.pth`
- `dqn_results.json` oppure `ppo_results.json`

## Isolamento Degli Artefatti

L'orchestrator evita conflitti tra esperimenti in questo modo:
- ogni config ha una cartella dedicata
- checkpoint, JSON finali e log TensorBoard non vengono scritti in una cartella condivisa
- se rilanci la stessa config, la cartella della run viene rimossa e ricreata

Questo significa che:
- rilanciare la stessa sweep va bene
- rilanciare la stessa config va bene
- gli artefatti vecchi di quella config vengono sovrascritti in modo pulito

Questo e' importante soprattutto per TensorBoard, perche' evita di mischiare event file di run diverse nella stessa directory.

## common_resolved.yaml

Per ogni run, l'orchestrator genera un file `common_resolved.yaml`.

Questo file:
- parte da `configs/config_common.yaml` oppure dal file passato con `--common`
- sostituisce solo i path di output:
  - `eval.results_dir`
  - `logging.tensorboard_dir`

In questo modo:
- non devi modificare `config_common.yaml`
- ogni run usa path dedicati
- il training riceve una config comune coerente con la cartella corrente

## 2. Come Usarlo

## Dry Run

Il dry run serve per controllare:
- quali config verranno eseguite
- in che ordine
- quali path useranno
- quale comando verra' lanciato

Esempio DQN:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --dry-run
```

Esempio PPO:

```bash
python scripts/orchestrator.py --model ppo --grid-dir configs/ppo --dry-run
```

## Esecuzione Reale

Per lanciare davvero tutte le run:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn
```

```bash
python scripts/orchestrator.py --model ppo --grid-dir configs/ppo
```

## Override Del Device

Puoi forzare il device per tutte le run della sweep:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --device cpu
```

Oppure:

```bash
python scripts/orchestrator.py --model ppo --grid-dir configs/ppo --device cuda
```

## Override Dei Timesteps

Puoi forzare gli step totali per tutte le config della sweep:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --timesteps 10000
```

Questo valore ha priorita' rispetto a quello definito nei singoli file YAML.

## Nome Personalizzato Della Sweep

Di default il nome della sweep coincide con il nome della cartella passata in `--grid-dir`.

Esempio:
- `configs/grid_dqn` -> sweep name `grid_dqn`
- `configs/ppo` -> sweep name `ppo`

Se vuoi usare un nome diverso:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --sweep-name test_lr
```

In questo caso i path diventano:

```text
results/sweeps/dqn/test_lr/
logs/sweeps/dqn/test_lr/
```

## Config Comune Personalizzata

Se non vuoi usare `configs/config_common.yaml`, puoi passare un altro file:

```bash
python scripts/orchestrator.py --model ppo --grid-dir configs/ppo --common configs/config_common.yaml
```

Questo e' utile se in futuro vuoi mantenere piu' versioni della config comune.

## 3. Parametri Disponibili

Lo script accetta questi parametri:

## `--model`

Valori possibili:
- `dqn`
- `ppo`

Serve per scegliere quale trainer usare.

Esempio:

```bash
--model dqn
```

## `--grid-dir`

Percorso della cartella che contiene i file `.yaml` della sweep.

Esempi:

```bash
--grid-dir configs/grid_dqn
--grid-dir configs/ppo
```

## `--common`

Percorso del file di configurazione comune.

Default:

```bash
configs/config_common.yaml
```

## `--device`

Valori possibili:
- `cpu`
- `cuda`
- `mps`

Se impostato, sovrascrive il device definito nella config specifica del modello.

Esempio:

```bash
--device cpu
```

## `--timesteps`

Intero opzionale.

Se impostato, sovrascrive `total_timesteps` di tutte le config della sweep.

Esempio:

```bash
--timesteps 50000
```

## `--sweep-name`

Nome opzionale della sweep.

Se non lo specifichi, viene usato il nome della cartella `--grid-dir`.

Esempio:

```bash
--sweep-name ablation_v1
```

## `--dry-run`

Flag opzionale.

Se presente:
- non esegue il training
- non crea artefatti reali della run
- stampa solo i comandi e i path risolti

Esempio:

```bash
--dry-run
```

## Esempi Completi

Dry run DQN:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --dry-run
```

Sweep reale PPO:

```bash
python scripts/orchestrator.py --model ppo --grid-dir configs/ppo
```

Sweep DQN su CPU con 10k step:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --device cpu --timesteps 10000
```

Sweep DQN con nome personalizzato:

```bash
python scripts/orchestrator.py --model dqn --grid-dir configs/grid_dqn --sweep-name debug_small
```

## Note Pratiche

- L'orchestrator esegue le run in sequenza, non in parallelo.
- I file YAML nella cartella grid devono essere compatibili con il modello scelto.
- Se usi `--model dqn`, nella cartella devono esserci config DQN.
- Se usi `--model ppo`, nella cartella devono esserci config PPO.
- `config_common.yaml` in questa fase puo' restare invariato: l'orchestrator modifica solo i path di output nella copia `common_resolved.yaml`.
