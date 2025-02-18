#!/usr/bin/env python3

"""
    Extract criterion benchmarks from target folder. Requires running `cargo bench` first with criterion's "html_report" feature enabled.
"""

import os
import pandas as pd

rootdir = "../target/criterion/"
blstimedir = "logs/bls12_381_time.csv"
bntimedir = "logs/bn254_time.csv"

def main():

    for f in os.listdir(rootdir):
        if f == "report":
            # Ignore list page of all benchmarks
            continue
        print("Processing benchmark group...", f)
        bench_group = os.path.join(rootdir, f)

        if "BLS12-381" in bench_group:
            bench_logname = blstimedir
        elif "BN254" in bench_group:
            bench_logname = bntimedir
        # TODO: Directory check for group vs. standalone benchmark
        if bench_logname == "":
            continue
        bench_log = open(bench_logname, "wt")
        bench_log.write("microbenchmark,mean_low,mean_est,mean_hi,sd_low,sd_est,sd_hi,med_low,med_est,med_hi,MAD_low,MAD_est,MAD_hi\n")
        group_benches = os.listdir(bench_group)
        group_benches.sort()
        for d in group_benches:
            if d == "report":
                # Ignore raw json and etc. within each benchmark
                continue

            print("Processing benchmark...", d)
            bench_report = os.path.join(bench_group, d, "report/index.html")
            record_benchmark(d, bench_report, bench_log)
        bench_log.close()
        print(bench_logname + "\n")
        df = pd.read_csv(bench_logname, index_col=0)
        print(df)

def record_benchmark(name, report, log):
        # Set first row names as row index
        # Ignore 'Slope' and 'R^2' (string is squared superscript) rows
        df = pd.read_html(report, index_col=0)[1].loc[['Mean', 'Std. Dev.', 'Median', 'MAD'], :]
        # Flatten ('Lower Bound', 'Estimate', 'Upper Bound') columns into single row for csv
        log.write(name + "," + ",".join(df.to_numpy().flatten()) + '\n')

if __name__ == "__main__":
    main()
