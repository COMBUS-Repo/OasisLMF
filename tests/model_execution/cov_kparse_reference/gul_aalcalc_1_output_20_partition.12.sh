#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

find fifo/ \( -name '*P13[^0-9]*' -o -name '*P13' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryaalcalc

mkfifo fifo/gul_P13

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summary_P13.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P13 work/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P13.idx work/gul_S1_summaryaalcalc/P13.idx > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 &

( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P13  ) &  pid3=$!

wait $pid1 $pid2 $pid3

