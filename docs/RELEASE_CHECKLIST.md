# Release Checklist

- [ ] SVQTD provider permission for segment-level labels has been confirmed.
- [ ] All authors have confirmed ownership or permission for the released code.
- [ ] No audio, URL, source identity, private path, credential, or model weight is present.
- [ ] `python scripts/check_release.py` passes.
- [ ] `python scripts/reproduce_metrics.py` reproduces the locked estimands.
- [ ] `python scripts/summarize_inner_support.py` reproduces the support table.
- [ ] Author names, order, affiliations, and ORCIDs are final.
- [ ] `CITATION.cff` version and release date are final.
- [ ] GitHub repository is public.
- [ ] GitHub release `v1.0.0` has been created.
- [ ] Zenodo DOI has been generated and checked.
- [ ] Code Availability in the manuscript contains both URLs.
