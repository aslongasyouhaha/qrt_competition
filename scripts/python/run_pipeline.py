"""Run the archived QRT training and allocation blend from any directory."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'catboost', 'gru', 'blend'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data, out = args.data_dir.resolve(), args.output_dir.resolve()
    common = ['--x-train', str(data / 'X_train.csv'), '--y-train', str(data / 'y_train.csv'), '--x-test', str(data / 'X_test.csv')]
    stages = ['catboost', 'gru', 'blend'] if args.stage == 'all' else [args.stage]
    if not args.dry_run:
        required = [data / name for name in ['X_train.csv', 'y_train.csv', 'X_test.csv']] if args.stage != 'blend' else [out / model / name for model in ['catboost', 'gru'] for name in ['oof_predictions.csv', 'test_predictions_with_scores.csv']]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            parser.error('Missing input files: ' + ', '.join(missing))
        out.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if stage == 'blend':
            script = 'catboost_gru_segmented_linear_blend.py'
            options = ['--catboost-model-dir', str(out / 'catboost'), '--gru-model-dir', str(out / 'gru'), '--output-dir', str(out / 'blend')]
        else:
            script = {'catboost': 'catboost_cluster_allocation_residual_training.py', 'gru': 'gru_global_cluster_residual_training.py'}[stage]
            options = common + ['--output-dir', str(out / stage), '--preds-output', str(out / (stage + '_submission.csv'))]
            if stage == 'gru' and args.device:
                options += ['--device', args.device]
        command = [sys.executable, str(ROOT / 'qrt' / script), *options]
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
