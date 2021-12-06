#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo fifo/gul_P16

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_summarycalc_P16



# --- Do ground up loss computes ---
summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 & pid1=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_summarycalc_P16 > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 &

eve 16 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P16  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P16 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1
