#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/il_S1_summaryleccalc

mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &

eve -R 6 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P6  &

wait $pid1 $pid2


# --- Do insured loss kats ---
