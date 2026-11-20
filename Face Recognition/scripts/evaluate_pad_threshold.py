from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def rates(y_true, y_pred):
    # y=1 bona fide, y=0 attack; pred=1 accept as live.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    apcer = fp / (fp + tn) if (fp + tn) else float('nan')  # attacks incorrectly accepted as bona fide
    bpcer = fn / (fn + tp) if (fn + tp) else float('nan')  # bona fide incorrectly rejected
    acer = (apcer + bpcer) / 2 if np.isfinite(apcer) and np.isfinite(bpcer) else float('nan')
    return {'TN_attack_reject': int(tn), 'FP_attack_accept': int(fp), 'FN_bona_fide_reject': int(fn), 'TP_bona_fide_accept': int(tp),
            'APCER': float(apcer), 'BPCER': float(bpcer), 'ACER': float(acer)}


def main():
    parser = argparse.ArgumentParser(description='Select PAD threshold on validation CSV, optionally evaluate on separate test CSV.')
    parser.add_argument('--validation-csv', required=True)
    parser.add_argument('--test-csv', default=None)
    parser.add_argument('--score-column', default='pad_live_score')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    val = pd.read_csv(args.validation_csv)
    for c in [args.score_column, args.label_column]:
        if c not in val.columns: raise ValueError(f'Kolom {c} tidak ditemukan')
    val = val.dropna(subset=[args.score_column, args.label_column]).copy()
    yv = val[args.label_column].astype(int).to_numpy()
    sv = val[args.score_column].astype(float).to_numpy()
    if set(np.unique(yv)) != {0, 1}:
        raise ValueError('Validation harus berisi kedua kelas: 0=attack dan 1=bona fide')
    thresholds = np.unique(sv)
    candidates = np.r_[thresholds[0]-1e-6, (thresholds[:-1]+thresholds[1:])/2, thresholds[-1]+1e-6]
    rows = []
    for t in candidates:
        pred = (sv >= t).astype(int)
        m = rates(yv, pred)
        rows.append({'threshold': float(t), **m})
    table = pd.DataFrame(rows)
    # Primary criterion: minimize ACER; tie-break toward lower APCER (stricter against attacks).
    best = table.sort_values(['ACER', 'APCER', 'BPCER'], ascending=[True, True, True]).iloc[0]
    output = {
        'method': 'validation_min_ACER_tiebreak_APCER',
        'threshold': float(best['threshold']),
        'validation_metrics': {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in best.drop(labels=['threshold']).to_dict().items()},
        'validation_samples_used': int(len(val)),
        'warning': 'Threshold hanya berlaku pada model PAD, kamera, dan protokol capture ini. Bekukan threshold sebelum menguji test set.'
    }
    if args.test_csv:
        test = pd.read_csv(args.test_csv).dropna(subset=[args.score_column, args.label_column]).copy()
        yt = test[args.label_column].astype(int).to_numpy()
        st = test[args.score_column].astype(float).to_numpy()
        pt = (st >= output['threshold']).astype(int)
        test_metrics = rates(yt, pt)
        if len(np.unique(yt)) == 2:
            test_metrics['ROC_AUC'] = float(roc_auc_score(yt, st))
        output['test_metrics_at_frozen_validation_threshold'] = test_metrics
        output['test_samples_used'] = int(len(test))
    out = Path(args.output_json).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding='utf-8')
    table.to_csv(out.with_name('pad_validation_threshold_search.csv'), index=False)
    print(json.dumps(output, indent=2))
    print('Saved:', out)

if __name__ == '__main__':
    main()
