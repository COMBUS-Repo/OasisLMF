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
mkdir -p output/full_correlation/

find fifo/ \( -name '*P2[^0-9]*' -o -name '*P2' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/il_S2_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S2_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P2

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_summary_P2.idx
mkfifo fifo/il_S2_eltcalc_P2
mkfifo fifo/il_S2_summarycalc_P2
mkfifo fifo/il_S2_pltcalc_P2

mkfifo fifo/full_correlation/il_P2

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_summary_P2.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P2
mkfifo fifo/full_correlation/il_S1_summarycalc_P2
mkfifo fifo/full_correlation/il_S1_pltcalc_P2
mkfifo fifo/full_correlation/il_S2_summary_P2
mkfifo fifo/full_correlation/il_S2_summary_P2.idx
mkfifo fifo/full_correlation/il_S2_eltcalc_P2
mkfifo fifo/full_correlation/il_S2_summarycalc_P2
mkfifo fifo/full_correlation/il_S2_pltcalc_P2



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid3=$!
eltcalc -s < fifo/il_S2_eltcalc_P2 > work/kat/il_S2_eltcalc_P2 & pid4=$!
summarycalctocsv -s < fifo/il_S2_summarycalc_P2 > work/kat/il_S2_summarycalc_P2 & pid5=$!
pltcalc -H < fifo/il_S2_pltcalc_P2 > work/kat/il_S2_pltcalc_P2 & pid6=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx > /dev/null & pid8=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_eltcalc_P2 fifo/il_S2_summarycalc_P2 fifo/il_S2_pltcalc_P2 work/il_S2_summaryaalcalc/P2.bin > /dev/null & pid9=$!
tee < fifo/il_S2_summary_P2.idx work/il_S2_summaryaalcalc/P2.idx > /dev/null & pid10=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 < fifo/il_P2 &

# --- Do insured loss computes ---
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid11=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid12=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid13=$!
eltcalc -s < fifo/full_correlation/il_S2_eltcalc_P2 > work/full_correlation/kat/il_S2_eltcalc_P2 & pid14=$!
summarycalctocsv -s < fifo/full_correlation/il_S2_summarycalc_P2 > work/full_correlation/kat/il_S2_summarycalc_P2 & pid15=$!
pltcalc -H < fifo/full_correlation/il_S2_pltcalc_P2 > work/full_correlation/kat/il_S2_pltcalc_P2 & pid16=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_eltcalc_P2 fifo/full_correlation/il_S1_summarycalc_P2 fifo/full_correlation/il_S1_pltcalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin > /dev/null & pid17=$!
tee < fifo/full_correlation/il_S1_summary_P2.idx work/full_correlation/il_S1_summaryaalcalc/P2.idx > /dev/null & pid18=$!
tee < fifo/full_correlation/il_S2_summary_P2 fifo/full_correlation/il_S2_eltcalc_P2 fifo/full_correlation/il_S2_summarycalc_P2 fifo/full_correlation/il_S2_pltcalc_P2 work/full_correlation/il_S2_summaryaalcalc/P2.bin > /dev/null & pid19=$!
tee < fifo/full_correlation/il_S2_summary_P2.idx work/full_correlation/il_S2_summaryaalcalc/P2.idx > /dev/null & pid20=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P2 -2 fifo/full_correlation/il_S2_summary_P2 < fifo/full_correlation/il_P2 &

( fmcalc -a2 < fifo/full_correlation/gul_fc_P2 > fifo/full_correlation/il_P2 ) & pid21=$!
( eve 2 2 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_fc_P2 -a1 -i - | fmcalc -a2 > fifo/il_P2  ) & pid22=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22

