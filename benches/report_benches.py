#!/usr/bin/env python3

"""
    Generate nice tables and graphs using the compiled and logged benchmarks.
    Requires running `cargo bench` with criterion's "html_reports" feature enabled.
    Also requires bechmark logs from extract_benches.py with `normalize_time_to_ms = True`.
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

logs_dir = os.path.join('.', 'logs')

# TODO: Microbenchmark random number generation for Fr
# TODO: Infer and report estimated / ideal times for our own reference, computed based on known dominant underlying computations
# TODO: Gather good sources attesting to this ^ (e.g. from DJB, Nigel Smart's Pairings for Cryptographers, OG papers for the fields & GS...)

# See also: https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
# https://seaborn.pydata.org/generated/seaborn.lineplot.html

# Column labels: %s (ms),mean_low,mean_est,mean_hi,sd_low,sd_est,sd_hi,med_low,med_est,med_hi,MAD_low,MAD_est,MAD_hi,R\xb2_low,R\xb2_est,R\xb2_hi,Slope_low,Slope_est,Slope_hi

def main():
    # Ensure pandas formats floats with 5 significant figures
    pd.set_option('display.float_format', lambda x: "%.5g" % x)
    verbose = False
    if sys.argv[1:] and sys.argv[1] == "verbose":
        verbose = True
    report_benchmarks(verbose)


def report_benchmarks(verbose=False):
    sns.set_theme(style="darkgrid")

    # BLS12-381 benchmark graphs
    bls12_g16_time_bench_path = os.path.join(logs_dir, 'BLS12-381_Groth16_time.csv')
    report_groth16_prove_benchmarks(bls12_g16_time_bench_path, verbose)
    report_groth16_verify_benchmarks(bls12_g16_time_bench_path, verbose)
    report_groth16_size_benchmarks(os.path.join(logs_dir, 'sizes.csv'), "BLS12-381", verbose)
    report_groth16_memory_benchmarks(os.path.join(logs_dir, 'memory.csv'), verbose)


    # BN254 benchmark graphs
    bn254_g16_time_bench_path = os.path.join(logs_dir, 'BN254_Groth16_time.csv')
    report_groth16_prove_benchmarks(bn254_g16_time_bench_path, verbose)
    report_groth16_verify_benchmarks(bn254_g16_time_bench_path, verbose)
    report_groth16_size_benchmarks(os.path.join(logs_dir, 'sizes.csv'), "BN254", verbose)

def report_groth16_prove_benchmarks(bench_path, verbose=False):

    print("Generating proof generation time graph for benchmark group", bench_path, "...")
    field_str =os.path.basename(bench_path).split('_')[0]

    df = pd.read_csv(bench_path, index_col=0)

    #clean up commit-and-prove to make a stack lineplot
    gs_commit_df = df.filter(regex='commit \d+ proofs', axis=0).assign(Operation = 'Commit').sort_values(by='iterations')#.set_index(['iterations'])
    if verbose:
        print(gs_commit_df)
    gs_prove_df = df.filter(regex='commit and prove \d+ equations satisfied', axis=0).assign(Operation = 'Commit and Prove').sort_values(by='iterations')#.set_index(['iterations'])
    if verbose:
        print(gs_prove_df)
    gs_cprove_df = pd.concat([gs_commit_df, gs_prove_df], ignore_index=True)
    print(gs_cprove_df[['iterations', 'Operation', 'mean_low', 'mean_est', 'mean_hi']])

    fig = sns.lineplot(data=gs_cprove_df, x=gs_cprove_df.iterations, y='mean_est', hue='Operation', hue_order=['Commit', 'Commit and Prove'], style='Operation', markers=True, dashes=[(3,3),(1,5)], legend='full')
    fig.fill_between(gs_commit_df.iterations, gs_commit_df['mean_low'], gs_commit_df['mean_hi'], alpha=0.3)
    fig.fill_between(gs_prove_df.iterations, gs_prove_df['mean_low'], gs_prove_df['mean_hi'], alpha=0.3, color=sns.color_palette()[1])
    fig.set(xlabel='# Groth16 Equations', ylabel='Time (ms)', title=f'Groth-Sahai over Groth16 Proof Generation Time ({field_str})')
    plt.legend(loc="upper left")
    fig.set_xlim(xmin=0, xmax=55)
    fig.set_ylim(ymin=-40, ymax=1240)


    plt.rcParams['legend.title_fontsize'] = 12
    plt.tight_layout()
    plt.savefig(bench_path.replace('time.csv', 'proof_time.png'))

    fig.clear()

def report_groth16_verify_benchmarks(bench_path, verbose=False):

    print("Generating verification time graph for benchmark group", bench_path, "...")
    field_str = os.path.basename(bench_path).split('_')[0]

    df = pd.read_csv(bench_path, index_col=0)

    g16_ver_df = df.filter(regex='verify \d+ plain Groth16 equations', axis=0).assign(Operation = 'Groth16 Verify').sort_values(by='iterations')#.set_index(['iterations'])
    if verbose:
        print(g16_ver_df)
    gs_ver_df = df.filter(regex='verify \d+ equations satisfied', axis=0).assign(Operation = 'GS over Groth16 Verify').sort_values(by='iterations')#.set_index(['iterations'])
    if verbose:
        print(gs_ver_df)
    ver_df = pd.concat([g16_ver_df, gs_ver_df], ignore_index=True)

    print(ver_df[['iterations', 'Operation', 'mean_low', 'mean_est', 'mean_hi']])

    fig = sns.lineplot(data=ver_df, x=ver_df.iterations, y='mean_est', hue='Operation', hue_order=['Groth16 Verify', 'GS over Groth16 Verify'], style='Operation', markers=True, dashes=[(3,3),(1,5)], legend='full')
    fig.fill_between(g16_ver_df.iterations, g16_ver_df['mean_low'], g16_ver_df['mean_hi'], alpha=0.3)
    fig.fill_between(gs_ver_df.iterations, gs_ver_df['mean_low'], gs_ver_df['mean_hi'], alpha=0.3) #, color=sns.color_palette()[1])
    fig.set(xlabel='# Groth16 Equations', ylabel='Time (ms)', title=f'Groth-Sahai over Groth16 Verification Time ({field_str})')
    plt.legend(loc="upper left")
    fig.set_xlim(xmin=0, xmax=55)
    fig.set_ylim(ymin=-10, ymax=310)

    plt.rcParams['legend.title_fontsize'] = 12
    plt.tight_layout()
    plt.savefig(bench_path.replace('time.csv', 'verify_time.png'))

    fig.clear()

def report_groth16_size_benchmarks(bench_path, field_str, verbose=False):

    print("Generating proof size graph for benchmark group", bench_path, "...")

    df = pd.read_csv(bench_path, index_col=0)

    gs_com1_df = df.filter(regex=f'{field_str} GS-over-Groth16 \d+ equation Com1 size', axis=0).assign(Value = 'Com1').drop_duplicates()
    gs_com1_df['iterations'] = pd.to_numeric(gs_com1_df.index.str.extract(f'{field_str} GS-over-Groth16 (\d+) equation Com1 size')[0]).to_numpy()
    gs_com1_df.sort_values(by='iterations')
    if verbose:
        print(gs_com1_df)

    gs_com2_df = df.filter(regex=f'{field_str} GS-over-Groth16 \d+ equation Com2 size', axis=0).assign(Value = 'Com2').drop_duplicates()
    gs_com2_df['iterations'] = pd.to_numeric(gs_com2_df.index.str.extract(f'{field_str} GS-over-Groth16 (\d+) equation Com2 size')[0]).to_numpy()
    gs_com2_df.sort_values(by='iterations')
    if verbose:
        print(gs_com2_df)

    gs_proof_df = df.filter(regex=f'{field_str} GS-over-Groth16 \d+ equation proof size', axis=0).assign(Value = 'Proof').drop_duplicates()
    gs_proof_df['iterations'] = pd.to_numeric(gs_proof_df.index.str.extract(f'{field_str} GS-over-Groth16 (\d+) equation proof size')[0]).to_numpy()
    gs_proof_df.sort_values(by='iterations')
    if verbose:
        print(gs_proof_df)

    size_df = pd.concat([gs_com1_df, gs_com2_df, gs_proof_df], ignore_index=True)

    print(size_df[['iterations', 'Value', 'size (B)', 'compressed size (B)']])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    sns.lineplot(data=size_df, x=size_df.iterations, y='size (B)', hue='Value', hue_order=['Com1', 'Com2', 'Proof'], style='Value', markers=True, dashes=True, legend='full', ax=ax1)
    sns.lineplot(data=size_df, x=size_df.iterations, y='compressed size (B)', hue='Value', hue_order=['Com1', 'Com2', 'Proof'], style='Value', markers=True, dashes=True, legend='full', ax=ax2, palette=sns.color_palette('pastel')[:3])
    fig.suptitle(f'Groth-Sahai over Groth16 Proof Size ({field_str})')
    ax1.set_xlabel('# Groth16 Equations')
    ax1.set_ylabel('Size (KB)')
    ax1.get_legend().set_title("Uncompressed")
    sns.move_legend(ax1, 'upper left')
    #ax2.set_ylabel('Compressed Size (KB)')
    ax2.set_ylabel('')
    ax2.get_legend().set_title("Compressed")
    sns.move_legend(ax2, 'center left')
    ax1.set_xlim(xmin=0, xmax=55)
    ax2.set_xlim(xmin=0, xmax=55)
    ax1.yaxis.set_major_formatter(lambda y, pos: (f'%.0f' % (y/1000)))
    #ax2.yaxis.set_major_formatter(lambda y, pos: (f'%.0f' % (y/1000)))
    ax2.set_yticks([])
    ax2.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax1.set_ylim(ymin=0, ymax=70000)
    ax2.set_ylim(ymin=-2400, ymax=70000)

    plt.rcParams['legend.title_fontsize'] = 12
    plt.tight_layout()
    plt.savefig(bench_path.replace('sizes.csv', f'{field_str}_Groth16_proof_size.png'))

    fig.clear()

def report_groth16_memory_benchmarks(bench_path, verbose=False):
    mem_df = pd.read_csv(bench_path)
    #mem_df["Max RSS (B)"] = mem_df["Max RSS (KB)"] / 1000
    mem_df["Page Size (KB)"] = mem_df["Page Size (B)"] / 1000
    mem_df["Heap + Stack Size (KB)"] = mem_df["Heap + Stack Size (B)"] / 1000
    mem_df = mem_df[["description", "Heap + Stack Size (KB)", "Page Size (KB)", "Max RSS (KB)"]]
    print(mem_df)

if __name__ == "__main__":
    main()
