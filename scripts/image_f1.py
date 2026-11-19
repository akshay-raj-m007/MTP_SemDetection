"""Image-level macro-F1 from the saved cross-validation predictions (no GPU, no training).

Question per class: "does this image contain a defect of this class?" An image counts as predicted positive when it has
at least one box of that class with score >= the cutoff. F1 is computed per class over the pooled held-out images,
then averaged over the classes that occur in the truth.

Usage: python scripts/image_f1.py ssd_pitch32_weighted ssd_v2_weighted retinanet_pitch32_weighted retinanet_v2_weighted
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import argparse
from sem_yolos.common import ROOT, CATEGORIES, read

DATA = lambda setup: 'pitch32' if 'pitch32' in setup else 'v2'


def image_f1(setup, cutoff):
    oof = read(ROOT/'outputs/cv5/summary'/f'{setup}_oof.json')
    truth = {}  # (image_id, category_id) pairs present in the held-out folds, plus all image ids
    images = set()
    for fold in oof['folds']:
        part = read(ROOT/f'outputs/cv5/prepared/{DATA(setup)}/fold{fold}/val.json')
        images |= {i['id'] for i in part['images']}
        truth.update({(a['image_id'], a['category_id']): 1 for a in part['annotations']})
    predicted = {(p['image_id'], p['category_id']) for p in oof['predictions'] if p['score'] >= cutoff}
    rows = {}
    for category in CATEGORIES:
        cid = category['id'] if isinstance(category, dict) else category
        name = category['name'] if isinstance(category, dict) else str(category)
        tp = sum((i, cid) in truth and (i, cid) in predicted for i in images)
        fp = sum((i, cid) not in truth and (i, cid) in predicted for i in images)
        fn = sum((i, cid) in truth and (i, cid) not in predicted for i in images)
        if tp+fn == 0:
            continue
        rows[name] = (2*tp/(2*tp+fp+fn), tp, fp, fn)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('setups', nargs='+'); p.add_argument('--cutoff', type=float, default=.3)
    args = p.parse_args()
    for setup in args.setups:
        rows = image_f1(setup, args.cutoff)
        macro = sum(v[0] for v in rows.values())/len(rows)
        detail = '  '.join(f'{k} {v[0]:.2f} (tp {v[1]} fp {v[2]} fn {v[3]})' for k, v in rows.items())
        print(f'{setup:32} macro-F1 {macro:.3f}   {detail}')
