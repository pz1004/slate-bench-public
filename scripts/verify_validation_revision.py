"""Verify the eligibility reanalysis against preserved inputs; never train models.

Run after fresh-validation-corrected. This also verifies stored coverage summaries,
source-group boundaries and frozen-fit declarations against their recorded evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'analysis-output/v2/fresh_validation'
OUT = RUN / 'audit_2026-10-25'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(frozen_fit_replay: Path | None = None) -> None:
    audit = pd.read_csv(ROOT / 'configs/fresh_validation_eligibility_audit.csv')
    manifest_path = ROOT / 'configs/fresh_validation_cohort_2026-10-25.csv'
    manifest = pd.read_csv(manifest_path)
    meta = json.loads((RUN / 'metadata.json').read_text())
    paths = [ROOT / 'configs/fresh_validation_source_check.csv',
             ROOT / 'configs/fresh_validation_tasks.csv', RUN / 'candidates.parquet', RUN / 'metadata.json',
             ROOT / 'configs/fresh_validation_eligibility_audit.csv', manifest_path,
             OUT / 'evidence.json', OUT / 'summary.json', OUT / 'protocol_before_correction.md']
    for key, rel in [('config', 'configs/fresh_validation_experiment.yaml'),
                     ('tasks', 'configs/fresh_validation_tasks.csv'),
                     ('module', 'src/tabular_sota_benchmark/fresh_validation.py'),
                     ('runner', 'scripts/fresh_validation_experiment.py')]:
        path = ROOT / rel
        assert digest(path) == meta['sha256'][key], f'Historical training input changed: {rel}'
        paths.append(path)
    # Retain exactly the protocol recorded by the training run, as well as the
    # pre-correction snapshot that also contains its first worker-count deviation.
    protocol = subprocess.check_output(['git', 'show', f"{meta['git_head']}:docs/fresh_validation_protocol.md"], cwd=ROOT)
    assert hashlib.sha256(protocol).hexdigest() == meta['sha256']['protocol']
    (OUT / 'protocol_at_training.md').write_bytes(protocol)
    paths.append(OUT / 'protocol_at_training.md')
    published_tasks = pd.read_csv(io.BytesIO(subprocess.check_output(
        ['git', 'show', f"{meta['git_head']}:configs/fresh_validation_tasks.csv"], cwd=ROOT)), comment='#')
    pd.testing.assert_frame_equal(published_tasks.drop(columns='use'),
                                 pd.read_csv(ROOT / 'configs/fresh_validation_tasks.csv', comment='#').drop(columns='use'))
    # Original persisted results must be unchanged from the starting checkout.
    protected = [*paths[:4], *sorted((RUN / 'analysis').glob('*'))]
    for path in protected:
        original = subprocess.check_output(['git', 'show', 'HEAD:' + str(path.relative_to(ROOT))], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == digest(path), f'Original artifact changed: {path}'
    reproduced = []
    for old in sorted((RUN / 'analysis').glob('*.csv')):
        new = RUN / 'analysis_2026-10-25_original' / old.name
        pd.testing.assert_frame_equal(pd.read_csv(old), pd.read_csv(new), check_exact=True)
        reproduced.append(dict(file=old.name, frames_exact=True, byte_identical=digest(old) == digest(new)))
    original = json.loads((RUN / 'analysis/decisions.json').read_text())
    rebuilt = json.loads((RUN / 'analysis_2026-10-25_original/decisions.json').read_text())
    assert all(original[k] == rebuilt[k] for k in original if k != 'command')
    selections = {
        'original': set(manifest.dataset_id),
        'three-exclusion': set(manifest.dataset_id) - {46527, 40666, 46864},
        'eligible': set(manifest.loc[manifest.status.eq('eligible'), 'dataset_id']),
        'eligible-unresolved': set(manifest.loc[manifest.status.ne('ineligible'), 'dataset_id'])}
    results = {}
    candidates = pd.read_parquet(RUN / 'candidates.parquet')
    for selection, expected in selections.items():
        folder = RUN / f'analysis_2026-10-25_{selection}'
        dec = json.loads((folder / 'decisions.json').read_text())
        assert set(dec['cohort']['selected_dataset_ids']) == expected
        assert dec['cohort']['manifest_sha256'] == digest(manifest_path)
        assert dec['input_sha256']['candidates'] == digest(RUN / 'candidates.parquet')
        assert dec['input_sha256']['config'] == digest(ROOT / 'configs/fresh_validation_experiment.yaml')
        assert dec['input_sha256']['analysis_code'] == digest(ROOT / 'scripts/fresh_validation_analysis.py')
        names = set(candidates.loc[candidates.dataset_id.isin(expected), 'dataset'])
        assert set(pd.read_csv(folder / 'unit_choices.csv').dataset) == names
        assert set(pd.read_csv(folder / 'dataset_scores.csv').dataset) == names
        assert dec['n_datasets'] == len(expected) and dec['n_units'] == 5 * len(expected)
        assert dec['n_candidates'] == 18 and dec['units_with_fewer_candidates'] == 0
        if selection != 'original':
            assert dec['analysis_status'] == 'post hoc eligibility correction'
            assert all(d['confirmatory'] is False and d['conclusion'] != 'difference supported' for d in dec['decisions'].values())
        results[selection] = dict(n_datasets=len(expected), analysis_status=dec['analysis_status'], decisions=dec['decisions'])
        paths.extend(sorted(folder.glob('*')))
    three = results['three-exclusion']['decisions']['C1']
    assert round(three['mean_difference'], 5) == .03365
    assert [round(v, 5) for v in three['ci_95']] == [.01477, .05527]
    # Check source grouping at every recorded outer split; inner-boundary behavior
    # is checked by the synthetic perturbation tests in test_grouped_inner.py.
    grouped_dir = ROOT / 'analysis-output/v2/score_level_tabrepo_grouped_inner'
    groups = pd.read_csv(ROOT / 'configs/tabrepo_source_groups.csv', comment='#').set_index('dataset')['group'].to_dict()
    split_path = grouped_dir / 'grouped_splits.json'
    record = json.loads(split_path.read_text())
    assert record['inner_group_column'] == 'group'
    assert record['groups_file_sha256'] == digest(ROOT / 'configs/tabrepo_source_groups.csv')
    splits = record['splits']
    universe = set().union(*(set(split['heldout']) for split in splits))
    for split in splits:
        assert len(universe - set(split['heldout'])) == split['n_train']
        train = {groups.get(d, d) for d in universe - set(split['heldout'])}
        heldout = {groups.get(d, d) for d in split['heldout']}
        assert not train & heldout
    paths.extend([split_path, ROOT / 'configs/tabrepo_source_groups.csv'])
    ff = ROOT / 'analysis-output/v2/tabrepo_frozen_fit'
    ffmeta = json.loads((ff / 'metadata.json').read_text())
    assert ffmeta['declared_fold'] == 0 and ffmeta['split_seed'] == 20261022 and ffmeta['repeats'] == 1
    assert ffmeta['driver_sha256_now'] == digest(ROOT / 'scripts/grouped_inner_splits.py')
    fitted = pd.read_csv(ff / 'fitted_choices.csv')
    ensemble = fitted[fitted.pool.eq('ensemble')]
    assert ensemble.nested_choice.eq(ensemble.train_selected_ensemble_inclusive).all()
    paths.extend(sorted(ff.glob('*')))
    replay_files = []
    if frozen_fit_replay is not None:
        for path in sorted(ff.glob('*.csv')):
            replay = frozen_fit_replay / path.name
            pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(replay), check_exact=True)
            replay_files.append(dict(file=path.name, frames_exact=True, replay_sha256=digest(replay)))
    # Independently recompute the stored coverage calculation from replicate intervals
    # and independent truth estimates (no simulation reruns).
    cal = ROOT / 'analysis-output/v2/calibration_study'
    coverage = pd.read_csv(cal / 'coverage.csv')
    coverage_checks = 0
    for design, reps_name, truth_name in [('per-target, J=20', 'replicate_contrasts.csv', 'truth_runs.csv'),
                                          ('shared, J=100', 'bridge_contrasts.csv', 'truth_runs_shared.csv'),
                                          ('shared, J=20', 'bridge_contrasts_short.csv', 'truth_runs_shared.csv'),
                                          ('per-target, J=20, bridge tables', 'replicate_contrasts.csv', 'truth_runs.csv')]:
        part = coverage[coverage.design.eq(design)]
        if part.empty:
            raise AssertionError(f'Unknown coverage design {design}: {coverage.design.unique()}')
        reps = pd.read_csv(cal / reps_name)
        if design.endswith('bridge tables'):
            bridge_ids = pd.read_csv(cal / 'bridge_contrasts_short.csv').replicate.unique()
            reps = reps[reps.scenario.eq('base') & reps.replicate.isin(bridge_ids)]
        truth = pd.read_csv(cal / truth_name).groupby('scenario').mean(numeric_only=True)
        for row in part.itertuples():
            sample = reps[reps.scenario.eq(row.scenario) & reps.contrast.eq(row.contrast)]
            theta = truth.loc[row.scenario, row.contrast]
            covered = (sample[f'{row.method}_low'] <= theta) & (theta <= sample[f'{row.method}_high'])
            assert np.isclose(covered.mean(), row.coverage, atol=1e-14)
            centre = {'boot':'estimate', 't':'estimate', 'nb':'nb_estimate', 'nbt':'nbt_estimate'}[row.method]
            ratio = sample[f'{row.method}_se'].mean() / sample[centre].std(ddof=1)
            assert np.isclose(ratio, row.se_ratio, atol=1e-12)
            verdict = ('anti-conservative' if row.coverage_high < .925 else 'conservative' if row.coverage_low > .975
                       else 'adequate' if row.coverage >= .925 and ratio >= .9 else 'inconclusive')
            assert verdict == row.verdict
            coverage_checks += 1
        paths.extend([cal / reps_name, cal / truth_name])
    paths.append(cal / 'coverage.csv')
    paths.extend([ROOT / 'scripts/fresh_validation_analysis.py', ROOT / 'scripts/build_paper_assets.py', Path(__file__)])
    report = dict(date='2026-10-25', no_training=True, original_inputs_unchanged=True,
                  original_csv_reproduction=reproduced, original_decisions_reproduced=True,
                  screening_tasks=len(audit), cohorts=results, source_group_outer_splits=len(splits),
                  frozen_fit_declared_fold=0, frozen_fit_ensemble_reference_identity=True,
                  frozen_fit_stored_score_replay=replay_files,
                  coverage_rows_verified=coverage_checks,
                  input_and_output_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))})
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Original CSVs: {len(reproduced)} exact; four cohorts checked; coverage rows: {coverage_checks}; no training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--frozen-fit-replay', type=Path)
    verify(parser.parse_args().frozen_fit_replay)
