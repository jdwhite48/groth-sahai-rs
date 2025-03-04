#!/usr/bin/env python3

"""
    Generate nice tables and graphs using the compiled and logged benchmarks.
    Requires running `cargo bench` with criterion's "html_reports" feature enabled.
    Also requires bechmark logs from extract_benches.py with `normalize_time_to_ms = True`.
"""

import os
import numpy as np
import pandas as pd

rootdir = "./logs"
bench_lognames = []

EMPTY_ENTRY = " "

# TODO: Extract microbenchmarks' mean_est, sd_est times as reference

# TODO: Microbenchmark random number generation for Fr

# TODO: Infer and report estimated / ideal times for our own reference, computed based on known dominant underlying computations
# TODO: Gather good sources attesting to this ^ (e.g. from DJB, Nigel Smart's Pairings for Cryptographers, OG papers for the fields & GS...)

# TODO: Proof generation time graph as **Groth-Sahai Overhead** for each field impl., with commit and commit-and-prove stack-line graph

# TODO: Render confidence interval of points as either error bars or (better) shaded confidence interval using pantas.melt + pyplot.fill_between / seaborn.lineplot
# See also: https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
# https://seaborn.pydata.org/generated/seaborn.lineplot.html

# TODO: Verification time graphs for each field impl., with separate line for plain Groth16?

# TODO: Figure out how to obtain and report (serialized) proof size / communication complexity




#bench_log.write("%s (ms),mean_low,mean_est,mean_hi,sd_low,sd_est,sd_hi,med_low,med_est,med_hi,MAD_low,MAD_est,MAD_hi,R\xb2_low,R\xb2_est,R\xb2_hi,Slope_low,Slope_est,Slope_hi\n" % bench_label)

def main():
    # Ensure pandas formats floats with 5 significant figures
    pd.set_option('display.float_format', lambda x: "%.5g" % x)


def time_convert_ms(str_time):
    # Format from criterion.rs report: "<float> <unit>" if time, or "<float>" if not
    unit_val, unit = (str_time + " ").split(" ", 1)
    if "ps" in unit:
        # 10^-12 -> 10^-3
        conv_val = float(unit_val) * pow(10.0, -9)
    elif "ns" in unit:
        # 10^-9 -> 10^-3
        conv_val = float(unit_val) * pow(10.0, -6)
    elif "µs" in unit:
        # 10^-6 --> 10^-3
        conv_val = float(unit_val) * pow(10.0, -3)
    elif "ms" in unit:
        # 10^-3 --> 10^-3
        conv_val = float(unit_val) * pow(10.0, 0)
    elif "s" in unit:
        # 10^0 --> 10^-3
        conv_val = float(unit_val) * pow(10.0, 3)
    elif unit.strip() == "" or unit.strip() == EMPTY_ENTRY:
        # Unitless value (e.g. R^2) will be left alone
        # EMPTY_ENTRY (e.g. missing Slope) returns EMPTY_ENTRY
        return unit_val

    # print("%s <-> %.5g ms" % (str_time, conv_val))

    # Preserve floating point precision when converting back into string
    return "%.5g" % conv_val


if __name__ == "__main__":
    main()
