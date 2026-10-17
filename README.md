# ptblank3 nonlinear-bit research

Attempt to replace MAME's `ptblank3_nonlinear` lookup table (Namco System 10, Point Blank 3 / Gunbalina,
mask bit 11) with a formula. Status and results: [FINDINGS.md](FINDINGS.md).

`scripts/` holds the analysis scripts, the GF(2) solver (`gf2solve.c`) and the MAME instrumentation patch.
`data/` holds derived indices only. The decrypt-stream dumps (`data/*.bin`) are not committed: they contain
the decrypted game program. Regenerate them with the patch and the steps at the end of FINDINGS.md.
