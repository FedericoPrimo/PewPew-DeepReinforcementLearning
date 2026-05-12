#!/usr/bin/env bash
# Quick-test orchestrator: training DQN + PPO con barre di progresso visibili,
# poi confronto e TensorBoard.
#
# Default: sequenziale (DQN poi PPO) — barre di caricamento visibili in tempo reale.
# Modalità --parallel: i due training in background con log redirect (più veloce, no progress bar).
#
# Uso:
#   ./scripts/quick_test.sh                  # default: 1000 step, cpu, sequenziale
#   ./scripts/quick_test.sh 5000 cuda        # 5000 step, GPU
#   ./scripts/quick_test.sh 50000 cpu --no-tb       # skip tensorboard
#   ./scripts/quick_test.sh 1000 cpu --parallel     # parallelo (no progress bar)

TIMESTEPS="${1:-1000}"
DEVICE="${2:-cpu}"
SKIP_TB="false"
PARALLEL="false"
for arg in "$@"; do
    case "$arg" in
        --no-tb) SKIP_TB="true" ;;
        --parallel) PARALLEL="true" ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="results/quick_test_logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "  QUICK TEST — DQN vs PPO"
echo "  timesteps=$TIMESTEPS | device=$DEVICE | mode=$([[ $PARALLEL == true ]] && echo parallel || echo sequential)"
echo "========================================================"

if [[ "$PARALLEL" == "true" ]]; then
    # Modalità parallela: background + log redirect (no progress bar in terminale)
    echo "[1/3] Training DQN + PPO in parallelo (log: $LOG_DIR/)..."
    python scripts/train_dqn.py --timesteps "$TIMESTEPS" --device "$DEVICE" \
        > "$LOG_DIR/dqn.log" 2>&1 &
    DQN_PID=$!
    python scripts/train_ppo.py --timesteps "$TIMESTEPS" --device "$DEVICE" \
        > "$LOG_DIR/ppo.log" 2>&1 &
    PPO_PID=$!
    echo "  DQN PID=$DQN_PID  |  PPO PID=$PPO_PID"
    trap "echo '  Interrotto.'; kill $DQN_PID $PPO_PID 2>/dev/null; exit 1" INT TERM

    wait $DQN_PID; DQN_EXIT=$?
    wait $PPO_PID; PPO_EXIT=$?

    FAIL=0
    if [[ $DQN_EXIT -ne 0 ]]; then
        echo ""; echo "  DQN FAILED (exit $DQN_EXIT). Tail log:"; echo "  ---"
        tail -30 "$LOG_DIR/dqn.log" | sed 's/^/    /'; echo "  ---"
        FAIL=1
    fi
    if [[ $PPO_EXIT -ne 0 ]]; then
        echo ""; echo "  PPO FAILED (exit $PPO_EXIT). Tail log:"; echo "  ---"
        tail -30 "$LOG_DIR/ppo.log" | sed 's/^/    /'; echo "  ---"
        FAIL=1
    fi
    if [[ $FAIL -ne 0 ]]; then
        echo ""; echo "Training falliti. Abort."
        echo "Suggerimento: 'pip install -r requirements.txt' se mancano deps."
        exit 1
    fi
else
    # Modalità sequenziale: foreground con barre di progresso visibili
    echo ""
    echo "[1a/3] Training DQN (barra di progresso live)..."
    python scripts/train_dqn.py --timesteps "$TIMESTEPS" --device "$DEVICE"
    DQN_EXIT=$?
    if [[ $DQN_EXIT -ne 0 ]]; then
        echo ""; echo "DQN FAILED (exit $DQN_EXIT). Abort."
        exit 1
    fi

    echo ""
    echo "[1b/3] Training PPO (barra di progresso live)..."
    python scripts/train_ppo.py --timesteps "$TIMESTEPS" --device "$DEVICE"
    PPO_EXIT=$?
    if [[ $PPO_EXIT -ne 0 ]]; then
        echo ""; echo "PPO FAILED (exit $PPO_EXIT). Abort."
        exit 1
    fi
fi

echo ""
echo "[2/3] Confronto risultati..."
python scripts/compare.py

if [[ "$SKIP_TB" == "true" ]]; then
    echo ""
    echo "[3/3] TensorBoard skipped (--no-tb)."
    echo "Avvialo manualmente con: tensorboard --logdir logs/"
else
    echo ""
    echo "[3/3] Avvio TensorBoard su http://localhost:6006  (Ctrl+C per chiudere)"
    tensorboard --logdir logs/ --port 6006
fi
