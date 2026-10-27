# Local correction package — 26 October 2026

`validation-reuse-correction-2026-10-26.tar.gz` accompanies the local manuscript
revision. It is **not yet published**. Its SHA-256 is in the adjacent `.sha256`
file and the local dated revision record. No public release or submission
readiness is claimed.

The package preserves the October 25 evidence and all original/corrected
results. `packaging_record.json` records the separate October 26 packaging work.
The explicit input allowlist is
`configs/validation_revision_package_2026-10-26.txt`; it includes untracked
correction files without depending on Git's tracked-file list. `manifest.json`
hashes every packaged file except itself; the archive checksum protects that
manifest too. Verify the archive checksum against the separately supplied
revision record before extraction.

## Reproduce after extraction

Use CPython 3.12. Install dependencies in an environment **outside** the extracted
package; dependencies can be installed before disconnecting the network:

```sh
sha256sum -c validation-reuse-correction-2026-10-26.tar.gz.sha256
tar -xzf validation-reuse-correction-2026-10-26.tar.gz
python3.12 -m venv /tmp/validation-reproduction-env
/tmp/validation-reproduction-env/bin/pip install -r validation-reuse-correction-2026-10-26/requirements-reproduction.txt
cd validation-reuse-correction-2026-10-26
/tmp/validation-reproduction-env/bin/python scripts/validation_revision_package.py verify
/tmp/validation-reproduction-env/bin/python scripts/validation_revision_package.py reproduce
/tmp/validation-reproduction-env/bin/python scripts/build_paper_assets.py --fresh-validation-only --generated generated-fresh
```

The `verify` command needs only the Python standard library. Reproduction reads
frozen candidate scores, not raw datasets, and disables Python network access.
It uses only included source code and snapshots, never the original checkout,
Git history, TabRepo, or ignored caches. Run each reproduction in an empty output
directory (default `reproduced/`); an alternative output must be outside the
package. Partial asset generation requires its own directory and refuses the
manuscript's full generated-assets directory. Existing full asset invocations
are unchanged.

`reproduced/verification.json` checks all four cohort memberships, all seven CSV
outputs per cohort (choices, every endpoint, intervals, adjusted tests, offsets,
shares and size strata), and all decision fields except the command line.
Original CSVs and decisions are also checked against the original `analysis/`.
The three-exclusion C1 remains +0.03365 [0.01477, 0.05527]. The verified C1 is
+0.021925 [−0.019786, +0.071658], or approximately +0.37 [−0.34, +1.22] rank
positions among 18 candidates. Neither harm nor equivalence is established.
All changed cohorts retain their post hoc status.

## Inspect the evidence

- `configs/fresh_validation_eligibility_audit.csv`: all 181 decisions, rationales,
  citations and original decisions.
- `cohorts/*.csv`: explicit, separate lists for original (46), three-exclusion
  (43), verified eligible (11), and eligible plus unresolved (34).
- `data/fresh_validation/catalogue/metadata/`: 181 cached OpenML descriptions
  and 180 cached feature schemas. The unavailable schema for 46520 is not filled
  in or silently omitted from the record.
- `source_index.json`: source URLs, metadata locations, recorded licences and
  citation fields. Linked primary web pages are not all archived here; their
  contents are not claimed to have been freshly retrieved or independently
  verified by packaging.
- `missing_documentation.json`: all unresolved source/target/grouping rationales
  and missing-schema records, preserving the audit's uncertainty.
- `analysis-output/v2/fresh_validation/audit_2026-10-25/evidence.json`: inspected
  target, retained/dropped feature names, aggregate counts and raw-file hashes
  for all 46 trained tasks. No individual rows are included. Raw-record checks
  are documented, not rerun by this package.
- The same audit folder holds protocol snapshots at training and before
  correction, and the historical verification record. Its supporting TabRepo
  and simulation checks are historical records, not rerun here. The changed
  asset builder is covered by the new package manifest, not its old hash.
- `docs/fresh_validation_protocol.md`: preserved dated deviations, including
  corrections to the original source-screening assertions.
- `configs/fresh_validation_excluded.csv`: the original TabArena-v0.1 / TabRepo-244
  source exclusion registry. Candidate scores, historical inputs, correction
  code and original/corrected outputs retain their original paths.

Recomputation verifies arithmetic conditional on the decisions and stored scores;
it does not certify eligibility judgments or repeat training. The package is
limited to the fresh-validation correction, not all supporting artifact audits.
Raw task rows, private colleague comments, manuscript files and unrelated caches
are excluded. See `NOTICE.md` for attribution and licensing scope.
