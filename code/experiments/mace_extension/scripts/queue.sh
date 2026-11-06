#!/bin/zsh
# Full extension queue, cheapest-to-validate first. Each step is resumable; a stop ends the queue.
cd "$(dirname "$0")/../../.."
PY=../sage-seqground/.venv/bin/python
for args in "E2" "E4other" "E4family" "E3cal" "E1_haiku --model claude-haiku-5-5"; do
  echo "START $args $(date)"
  eval $PY -m experiments.mace_extension.scripts.run $args || exit 1
done
echo QUEUE_DONE
