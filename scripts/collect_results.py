"""Collect every cross-validated setup into one results file: settings, CV, final test and pitch32-vs-v2 comparisons.

Writes outputs/results/results.json (everything) and outputs/results/results.csv (one headline row per setup).
Reads only files that the runs saved; a missing result is written as null with the reason, never filled in.
Run after cv5.py, final_test.py and compare_cv.py; safe to rerun at any time.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import csv
import socket
import subprocess
from datetime import datetime
from cv5 import SETUPS, schedule
from final_ssd import CUTOFF
from final_test import folders
from sem_yolos.common import ROOT, read, write

SUMMARY = ROOT/'outputs/cv5/summary'
OUT = ROOT/'outputs/results'
# The first two finals were run before final_test.py, with epochs fixed by hand before the test run.
OLD_FINALS = dict(
    ssd_pitch32=dict(metrics='outputs/mobilevitv2_ssd/final/pitch32_seed42/eval_test/metrics.json',
                     cutoff='outputs/mobilevitv2_ssd/final/pitch32_seed42/eval_test/cutoff.json', stage_b_epoch=31, rule='fixed by hand before the test run'),
    yolos_pitch32_weighted=dict(metrics='outputs/final_yolos/run/eval_test/metrics.json', cutoff=None, stage_b_epoch=13, rule='fixed by hand before the test run'))
SETTINGS = ['size', 'native_pitch', 'tile_overlap', 'resize_mode', 'sampling_power', 'sampling_cap', 'stage_a_epochs', 'stage_b_epochs',
            'batch_size', 'accumulation', 'stage_a_lr', 'stage_b_head_lr', 'stage_b_encoder_lr', 'seed']


def maybe(path):
    path = ROOT/path if not isinstance(path, _ProjectPath) else path
    return read(path) if path.exists() else None


def settings(setup):
    config = read(ROOT/SETUPS[setup]['config'])
    config.update(schedule(setup))
    return dict(model=SETUPS[setup]['model'], images=SETUPS[setup]['data'], sampling=SETUPS[setup].get('sampling', 'natural'),
                config=SETUPS[setup]['config'], **{k: config.get(k) for k in SETTINGS})


def cv(setup, rows):
    row = rows.get(setup)
    if row is None:
        return None
    return dict(best_epoch=row['epoch'], map=row['map'], map50=row['map50'], per_fold_map=row['per_fold_map'], boxes=row['boxes'],
                boxes_found_at_score_05=row['correct_at_05'], boxes_found_at_score_30=row['correct_at_30'],
                per_class={k: dict(ap=v['ap'], ap50=v['ap50'], boxes=v['support']) for k, v in row['per_class_ap'].items()})


def test(setup):
    if setup in OLD_FINALS:
        spec = OLD_FINALS[setup]
        metrics, fixed = maybe(spec['metrics']), maybe(spec['cutoff']) if spec['cutoff'] else None
        epoch, rule = spec['stage_b_epoch'], spec['rule']
    else:
        base, run = folders(setup, False)
        selection = maybe(base/'selection.json')
        metrics = maybe(run/'eval_test/metrics.json')
        fixed = maybe(run/'eval_test/cutoff.json') if SETUPS[setup]['model'] == 'ssd' else metrics
        epoch, rule = (selection or {}).get('stage_b_epoch'), (selection or {}).get('rule')
    if metrics is None:
        return None
    fixed = fixed or metrics
    d = fixed.get('detection')
    return dict(stage_b_epoch=epoch, epoch_rule=rule, cutoff=CUTOFF, map=metrics['map'], map50=metrics['map50'],
                per_class={k: dict(ap=v['ap'], ap50=v['ap50']) for k, v in metrics['per_class_ap'].items()},
                boxes=d['tp']+d['fn'] if d else None, boxes_found=d['tp'] if d else None, false_boxes=d['fp'] if d else None,
                per_class_found={k: dict(found=v['tp'], boxes=v['support']) for k, v in fixed.get('per_class_detection', {}).items()},
                clean_images_with_false_box=fixed.get('fraction_clean_with_false_detection'))


def comparisons():
    found = []
    for path in sorted(SUMMARY.glob('compare_*.json')):
        c = read(path)
        found.append(dict(first=c['first'], second=c['second'], epochs=c['epochs'], resamples=c['resamples_used'], seed=c['seed'],
                          folds_won_by_first=c['folds_won_by_first'], folds=c['folds'],
                          **{m: dict(difference=c[m]['difference'], interval_95=c[m]['interval_95'], share_first_better=c[m]['share_first_better'],
                                     supported=c[m]['interval_95'][0] > 0) for m in ('map', 'map50')}))
    return found


def provenance():
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(generated=datetime.now().isoformat(timespec='seconds'), machine=socket.gethostname(), git_commit=git('rev-parse', 'HEAD'),
                git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'), uncommitted_changes=bool(git('status', '--porcelain', '--untracked-files=no')))


def main():
    rows = {r['setup']: r for r in read(SUMMARY/'summary.json')} if (SUMMARY/'summary.json').exists() else {}
    setups = {}
    for name in SETUPS:
        entry = dict(settings=settings(name), cv=cv(name, rows), test=test(name))
        if entry['cv'] is None: entry['missing_cv'] = 'not in outputs/cv5/summary/summary.json: run cv5.py for this setup'
        if entry['test'] is None: entry['missing_test'] = 'no final test metrics: run final_test.py for this setup'
        setups[name] = entry
    result = dict(provenance=provenance(),
                  notes=dict(cv='Pooled out-of-fold predictions over 5 grouped folds of the 82 train+val images (61 boxes), best pooled-mAP stage B epoch.',
                             test='One run on the 13 test images (9 boxes) after training on all train+val; boxes found at the cutoff score, IoU .5, class must match.',
                             comparisons='Paired grouped bootstrap of pooled CV predictions; supported means the whole 95% interval is above zero.'),
                  setups=setups, comparisons=comparisons())
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'results.json', result)
    columns = ['setup', 'model', 'images', 'sampling', 'size', 'sampling_power', 'sampling_cap', 'stage_b_epochs', 'cv_epoch', 'cv_map', 'cv_map50',
               'cv_found_05', 'cv_found_30', 'cv_boxes', 'test_epoch', 'test_map', 'test_map50', 'test_found', 'test_boxes', 'test_false']
    with open(OUT/'results.csv', 'w', newline='', encoding='utf-8') as f:
        out = csv.writer(f); out.writerow(columns)
        for name, e in setups.items():
            s, c, t = e['settings'], e['cv'] or {}, e['test'] or {}
            out.writerow([name, s['model'], s['images'], s['sampling'], s['size'], s['sampling_power'], s['sampling_cap'], s['stage_b_epochs'],
                          c.get('best_epoch'), c.get('map'), c.get('map50'), c.get('boxes_found_at_score_05'), c.get('boxes_found_at_score_30'), c.get('boxes'),
                          t.get('stage_b_epoch'), t.get('map'), t.get('map50'), t.get('boxes_found'), t.get('boxes'), t.get('false_boxes')])
    print(f"{'setup':32}{'CV mAP':>8}{'CV mAP50':>9}{'test mAP':>9}{'test mAP50':>11}  found/false")
    number = lambda v: '   -   ' if v is None else f'{v:7.3f}'
    for name, e in setups.items():
        c, t = e['cv'] or {}, e['test'] or {}
        print(f"{name:32}{number(c.get('map')):>8}{number(c.get('map50')):>9}{number(t.get('map')):>9}{number(t.get('map50')):>11}  "
              f"{t.get('boxes_found', '-')}/{t.get('false_boxes', '-')}")
    print(f'\n{len(result["comparisons"])} comparisons. Written: {(OUT/"results.json").relative_to(ROOT).as_posix()} and results.csv')


if __name__ == '__main__': main()
