import json
import shutil
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import os

from numpy import mean, isnan
from tqdm import tqdm
from textwrap import wrap

from dashboard import smooth

# SERVER_LOGGING_ROOT = "/mnt/slab/krypton/"
# SERVER_LOGGING_ROOT = "/Volumes/slab/krypton/e_maml_paper/"
LOGGING_ROOT = os.path.expanduser("~/data/rcall/gce/")
files = glob(LOGGING_ROOT + "maze-parameter-sweep/**/*-e_maml-*/**/progress.json", recursive=True)

# keys = ['grad_0_step', 'grad_1_step', 'grad_2_step', 'grad_3_step', 'pre-update', 'post-update']
KEYS = ['grad_0_step', 'grad_1_step']

REW_PLOT = -0.05
REW_SHOW = -0.036
SELECTION_FOLDER = './e-maml-maze-good-runs'


def pretty_please(path):
    log_dir = os.path.dirname(path)

    with open(path, "r") as f:
        lines = f.readlines()
        data = [json.loads(l) for l in lines]
        ledger = defaultdict(list)
        for d in data:
            for k in KEYS:
                try:
                    v = d[k]
                except Exception:
                    v = float('nan')
                ledger[k].append(v)

        tail = mean(ledger['grad_1_step'][-50:])
        if tail < REW_PLOT or isnan(tail):
            return  # print('bad experiment, do not plot')

        plt.figure(figsize=(6, 4), dpi=120)
        *meta_data, param_string = log_dir.split("/")[6:]
        param_text = "\n".join(wrap(param_string.replace('-', ' '), width=50))

        plt.title("\n".join(meta_data), fontsize=8)
        for k, v in ledger.items():
            plt.plot(smooth(v, 5), 'o-', markersize=0.2, label=k.replace("_", " "))

        plt.ylim(-0.08, 0)
        plt.text(0.0, -0.013, param_text, fontsize=8)
        plt.legend(loc='upper right', framealpha=0)
        # plt.ylim(0, 1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, param_string + ".progress.png"), dpi=100)
        if tail < REW_SHOW:
            plt.close()
        else:
            print(tail, param_string, sep='\t')
            os.makedirs(SELECTION_FOLDER, exist_ok=True)
            shutil.copytree(log_dir, os.path.join(SELECTION_FOLDER, param_string))
            plt.show(block=False)


if __name__ == "__main__":
    for f in sorted(files)[:]:
        pretty_please(f)
