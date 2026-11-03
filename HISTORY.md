# History

0.3.0 (unreleased)
------------------
* Adds ModernProst support (`pholdlib.modernprost`). ModernProst maps an amino-acid sequence directly to Foldseek's 3Di alphabet and a 12-state alphabet in one forward pass, with no ProstT5 encoder or CNN head. Four checkpoints: `modernprost-base` and `modernprost-50M` (argmax 3Di/12-state), and `modernprost-pssm` and `modernprost-50M-pssm` (per-residue profiles)
* Builds Foldseek query databases from ModernProst outputs: combined 3Di + 12-state databases (search with `--ss-12st 1`), and 3Di / 12-state / amino-acid profile databases for the `-pssm` models
* Fast inference engines: token-budget batching, a sequence-packing FlexAttention engine with optional `torch.compile`, and a multi-GPU work queue (one process per GPU when compiling). Long proteins are chunked automatically, and out-of-memory batches are retried in smaller pieces
* Adds per-residue pLDDT prediction with the pLDDT head shipped in every ModernProst repository (`load_plddt_head`; `run_modernprost_inference(..., plddt_head=...)`). pLDDT is predicted in the same forward pass, at ~2-5% extra time
* Adds pLDDT masking of query databases (`plddt=`, `plddt_threshold=`). For `-pssm` models the 3Di and 12-state profile columns are zeroed; for the argmax models residues are set to X in both alphabets. Both match `foldseek lowconfquery`, and the amino-acid profile/database is never masked. Off unless a threshold is given
* Pins each ModernProst model to a specific Hugging Face commit (the 02/11/2026 release: nng `-pssm` weights + pLDDT heads). A cached older snapshot is detected and re-downloaded. The previous weights remain on the Hub at tag `jul26`
* Fixes ModernProst profile databases missing Foldseek's per-record null byte, which made Foldseek read every profile one residue short (fixed before any release)
* Requires `transformers >=4.51` (previously `>=4.34`) so the `-pssm` checkpoints load
* Hardens Zenodo backup tarball extraction against path traversal
* CI runs on every pull request, not only PRs into `main`
* Note: there was no 0.2.0 release. 0.2.0 marked the first ModernProst code on the development branch and was never published

0.1.2 (2026-06-12)
------------------
* Faster masking of low-confidence ProstT5 residues

0.1.1 (2026-06-09)
------------------
* Relaxes the `numpy` requirement from `>=2.0` to `>=1.24`, so phold and baktfold install in environments with system-managed numpy 1.26 (e.g. the Pawsey PyTorch ROCm image)

0.1.0 (2026-06-09)
------------------
* Initial release: shared ProstT5 inference library for phold and baktfold, covering ProstT5 loading, CNN 3Di prediction and 3Di/probability output writing
* Multi-GPU ProstT5 inference (models loaded sequentially across devices)
* Tests and GitHub Actions CI (including ProstT5 integration tests with a cached model), and a release workflow publishing to PyPI on version tags
