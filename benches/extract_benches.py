#!/usr/bin/env python3

"""
    Extract criterion benchmarks from target/Criterion folder.
    Requires running `cargo bench` first with criterion's "html_reports" feature enabled.
"""

import os
import re
import sys
import numpy as np
import pandas as pd


target_dir = "../target/criterion/"
bench_lognames = []

# Use time_convert_ms to convert all of ps, ns, µs, ms, s to ms?
normalize_time_to_ms = True

EMPTY_ENTRY = " "

# Alternatively, filled curves for commit and commit-and-prove via Criterion benchmarks

def main():
    # Ensure pandas formats floats with 5 significant figures
    pd.set_option('display.float_format', lambda x: "%.5g" % x)
    verbose = False
    if sys.argv[1:] and sys.argv[1] == "verbose":
        verbose = True
    extract_benchmarks(verbose)


def extract_benchmarks(verbose=False):
    """
        Generate csv files containing the benchmarks, assuming the following:
            - `cargo bench` has already been run, and is in target/criterion
            - "html_reports" feature of Criterion-rs is enabled
            - Benchmarks are appropriately binned into groups (for now? untested)
            - Criterion reports all floats with 5 significant figures of precision
            - ./logs/ directory is writeable
    """

    for bench_label in os.listdir(target_dir):
        if bench_label == "report":
            # Ignore list page of all benchmarks
            continue
        print("Extracting benchmark group", bench_label, "...")
        bench_group = os.path.join(target_dir, bench_label)

        # TODO: Directory check for group vs. standalone benchmark
        bench_logname = os.path.join("logs", "%s_time.csv" % bench_label)
        bench_lognames.append(bench_logname)

        bench_log = open(bench_logname, "wt")
        bench_log.write("%s (ms),iterations,mean_low,mean_est,mean_hi,sd_low,sd_est,sd_hi,med_low,med_est,med_hi,MAD_low,MAD_est,MAD_hi,R\xb2_low,R\xb2_est,R\xb2_hi,Slope_low,Slope_est,Slope_hi\n" % bench_label)
        group_benches = os.listdir(bench_group)
        group_benches.sort()
        for bench in group_benches:
            if bench == "report":
                # Ignore raw json and etc. within each benchmark
                continue
            if verbose:
                print("Extracting benchmark", bench, "...")
            bench_report = os.path.join(bench_group, bench, "report", "index.html")
            record_benchmark(bench, bench_report, bench_log)
        bench_log.close()
        print("Generated benchmark group logfile at %s." % bench_logname)
        df = pd.read_csv(bench_logname, index_col=0)
        if verbose:
            print(df)


def record_benchmark(name, report, log):
    """
        Assuming `log` file is open with (ci_low, estimate, ci_high) columns for each data point (see `extract_benchmark`):
            - Read the HTML `report` for the given Criterion-rs benchmark
            - Normalize all reported times to ms (optional)
            - Write the given benchmark with label `name` as row in csv
    """
    # Parse HTML report, with first row names as its index
    # 'R^2' and 'Slope' are unlikely to be useful, since they measure time vs. #iterations for 100 sample-counts
    try:
        df = pd.read_html(report, index_col=0)[1].loc[['Mean', 'Std. Dev.', 'Median', 'MAD', 'R\xb2', 'Slope'], :]
    except KeyError:
        # Some benchmarks don't report a slope for some reason
        df = pd.read_html(report, index_col=0)[1].loc[['Mean', 'Std. Dev.', 'Median', 'MAD', 'R\xb2'], :]
        df.loc['Slope'] = EMPTY_ENTRY
    # Assume log file is open and csv has column labels
    # Flatten ('Lower Bound', 'Estimate', 'Upper Bound') columns into single row for csv
    row = df.to_numpy().flatten()
    # If enabled, convert all unit times reported by Criterion to floats in ms (without loss of precision)
    if normalize_time_to_ms:
        row = np.vectorize(time_convert_ms)(row)
    # Write benchmark as single row to logfile
    num_iter = re.search(r'\s+(?P<iter>\d+)\s+', name)
    if num_iter is None:
        num_iter = "NaN"
    else:
        num_iter = num_iter["iter"]
    log.write("%s,%s,%s\n" % (name, num_iter, ','.join(row)))


def time_convert_ms(str_time):
    # Format from criterion.rs report: "<float> <unit>" if time, or "<float>" if not
    # can't use time.strptime because no picosecond, nanosecond, or millisecond unit parsing...
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

    # Preserve sig. fig. precision when converting back into string
    return "%.5g" % conv_val


if __name__ == "__main__":
    main()
