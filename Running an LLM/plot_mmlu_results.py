import os
import glob
import json
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def load_json_files(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "*.json")))
    results = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["__file__"] = os.path.basename(f)
            results.append(data)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    return results


def build_summary(results):
    rows = []
    for r in results:
        model = r.get("model")
        device = r.get("device")
        q = r.get("quantization_bits")
        quant = str(q) if q is not None else "full"
        # prefer duration_seconds, fall back to real_time_seconds
        wall = r.get("duration_seconds", r.get("real_time_seconds"))
        cpu = r.get("cpu_time_seconds")
        gpu = r.get("gpu_time_seconds")
        rows.append({
            "model": model,
            "device": device,
            "quantization": quant,
            "wall_clock_seconds": wall,
            "cpu_time_seconds": cpu,
            "gpu_time_seconds": gpu,
            "file": r.get("__file__"),
        })
    df = pd.DataFrame(rows)
    return df


def build_subject_matrix(results):
    # gather all subjects and models
    subjects = set()
    model_names = []
    data = defaultdict(dict)
    for r in results:
        model = r.get("model") or r.get("__file__")
        model_names.append(model)
        for s in r.get("subject_results", []):
            subj = s.get("subject")
            acc = s.get("accuracy")
            if subj is None or acc is None:
                continue
            subjects.add(subj)
            data[model][subj] = acc

    subjects = sorted(subjects)
    model_names = [m for m in model_names]
    df = pd.DataFrame(index=subjects, columns=model_names)
    for m in model_names:
        for subj in subjects:
            df.at[subj, m] = data.get(m, {}).get(subj)
    # convert to numeric
    df = df.astype(float)
    return df


def plot_subject_accuracies(df, outpath):
    if df.empty:
        print("No subject-wise accuracies to plot.")
        return
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')
    ax = df.plot(kind='bar', figsize=(max(8, len(df.index) * 0.6), 6))
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Subject')
    ax.set_title('MMLU Subject Accuracies by Model')
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    # rotate x labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved bar chart to {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='.', help='Directory containing result JSON files')
    parser.add_argument('--out-prefix', default='mmlu_results', help='Output filename prefix')
    args = parser.parse_args()

    results = load_json_files(args.dir)
    if not results:
        print(f"No JSON files found in {args.dir}")
        return

    summary_df = build_summary(results)
    # print table to stdout
    print("\nSummary table:")
    print(summary_df.to_string(index=False))
    csv_out = f"{args.out_prefix}_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV to {csv_out}")

    subj_df = build_subject_matrix(results)
    png_out = f"{args.out_prefix}_subject_accuracies.png"
    plot_subject_accuracies(subj_df, png_out)


if __name__ == '__main__':
    main()
