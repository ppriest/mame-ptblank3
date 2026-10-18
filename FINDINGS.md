# ptblank3 / gunbalina nonlinear bit (ns10crypt "ptblank3_nonlinear" table)

Goal: replace the 0x2ec00-byte residual table (one bit per decrypted word, folded into
mask bit 11) with a formula.

## Data
- Stream dumped from an instrumented namcos10 build (patch: ns10crypt_dump_instrumentation.patch),
  records of (cipher, plain, table bit, index): ../mamerun/dump.bin (ptblank3), ../mamerun_gb/dump.bin (gunbalina).
- 1,531,904 words, one activation (crypto_switch_w 8000, IV 0), 187 NAND blocks, 256 words/page, no spare bytes.
- gunbalina's encrypted stream is identical to ptblank3's; the shared table is legitimate.
- Python model ns10.py reproduces every plaintext word from the dump.

Notation: word i = the i-th decrypted word; t_i = table bit used in the mask of word i+1;
c, p, m = cipher, plain, mask (= c ^ p); state words i..i-k.

## Exact facts
1. Mask bits other than 11 are linear in 3 words of (c, p) (MAME's model; replay confirms).
2. t_i is NOT a function of 3 words of (c, p): 2835 conflicts in 3163 identical-state pairs.
3. Deep zero plaintext (>= 40 zero words): t_i = par(c_i & 0x078d) ^ par(c_{i-1} & 0x0005) ^ t_{i-2}, exact (450,425 samples).
4. Cell m_{i-2}[0] == 0 and m_{i-3}[0] == 1 (25% of words): t_i = L4'(c, p of words i..i-3) exactly (383,098 samples). L4' terms:
   c0 {0,1,3,4,7,12,15} p0 {1,2,3,6,8,12,15} c1 {4,8,9,10,12,13,15} p1 {0,2,4,8,10,11}
   c2 {0,4,5,6,8,9,10,11,14} p2 {0,2} c3 {0,1,2,6,8,11,13,15} p3 {0,2}   (c_k = word i-k, bit list), no constant.
   The other three cells are not affine in up to 6 words + 12 t-lags + c_{i+1}.
5. So t_i = L4' ^ (m_{i-2}[0] | ~m_{i-3}[0]) & h_i. The OR gate is instantaneous (no effect on later words).
   h_i ("event", rate 0.163 on gate-open rows, 0 in deep zero) is the unknown.

## Properties of h
- Does not depend on plaintext words i..i-3 (after >= 30 zero words, residual stays ~0 for the first
  4 nonzero words and switches on at word 4). A lone nonzero word triggers h only at depth 4.
- Decays gradually over ~15-20 words after plaintext becomes zero; clusters: P(h_i | h_{i-1}) = 0.39 vs 0.09 after 3 quiet.
- MI with every plaintext word i+1..i-6 ~0.08 bit (regional), ~0 with every cipher/mask word.
- Turn-on after quiet is ~determined by (c, p) of words i-4, i-5 (1 conflict / 984 pairs); staying on is
  not determined by p words 3..7 or masks 3..4.

## Ruled out for h (adequate 6-word linear span unless noted)
- quadratic in (c,p) of <= 8 words, +t lags <= 32, +c_{i+1}, +e lags; also restricted to gate-open rows / per cell
- quadratic / cubic in plaintext (and masks) of words 4..9 back
- counters (word index, page position, bus-cycle strides, NAND address), prefix/periodic accumulators (period 1..8)
- 1-bit hold registers: single-bit select with single-bit load (120k tests) or linear load of fire-time words
- any affine factor (exact or >= 97% of events), any single-literal or 2-literal forbidden conjunction (words 2..12),
  3-bit XOR literals (5-word window), single-bit mux selectors inside cells, linear recurrence on h's past (order <= 12)
- greedy raw-bit conditional-entropy selection: only regional bits, entropy falls slowly

## Robust linear model (87.8% of all words; not needed given 4/5 above)
t_i = par(c_i & 0x078d) ^ par(c_{i-1} & 0x0005) ^ t_{i-2} ^ par(p_i & 0x0658) ^ par(p_{i-1} & 0x3527) ^ par(p_{i-2} & 0xbefe)
      ^ par(p_{i-3} & 0x03d8) ^ par(p_{i-4} & 0x6cc3) ^ par(p_{i-5} & 0x7431)

## Pitfalls met
- GF(2) elimination amplifies any nonlinear residual into ~50% inconsistency; never use it to judge a noisy fit.
- Never let the target leak into features: t prefix parities, m_{i+1}[11], p_{i+1}[11] all contain t_i.
- Selecting rows by "no event yet" and then finding them linear is circular.

## Scripts (this directory)
ns10.py model/loader; gf2.py + gf2solve.c exact GF(2) solver; fwht.py Walsh fits; features.py;
analyze*.py collisions/fits; zero*.py deep-zero laws; robust.py FWHT linear model; cells.py/subcell.py cell tests;
pairmin.py/percell.py forbidden conjunctions; hull2.py/isd.py affine factors; events*.py/tree.py/greedy.py event structure;
runend.py/impulse2.py/trans.py timing; holdsearch.c/holdlin.py/strided.py hidden registers; triple.c 3-literal search.

## Late additions
- 3-literal forbidden conjunctions over m words 1..5 + p words 4..6 (128 bits, all polarities): none.
- Quadratic over (m2,m3,p4,p5), (m2,m3,m4,p4,p5,p6), (m1,m2,m3,c4,c5,p4,p5), also on gate-open rows only: fail.
- h is not a linear recurrence on its own past (orders 4..12 with words) -> not a hidden linear filter.
- "Turn-on determined by (c4,c5)" is state repetition: a control key of words that cannot matter (c1,c2) is equally consistent.
- Lone 0xffff impulses at depth 4 fire in 6 of 27 cases with no visible pattern in masks or t history.
- Reference plaintext provenance (PR #11166: decrypted program from Guru, real hardware) cannot be the cause:
  the other 15 mask bits obey their exact linear recurrences at every word, which a modified RAM image would break.
- PR #11236/#11244 (the table's origin) only states the nonlinear calculation "is still proving illusive".

## Status
Not solved. Linear part and the OR gate on (m[i-2].0, m[i-3].0) are exact; the gated hidden bit h
(rate 0.16 on gate-open rows) has resisted every closed-form and small-state model tried.
MAME tree restored (patch saved here); E:/mame/namcos10.exe is the instrumented build (hook inert without NS10_DUMP).

## Layout of this folder
- FINDINGS.md (this file); scripts/ (all analysis scripts, gf2solve.c solver, dump patch); data/ (dumps: 10-byte records
  type,b,cipher16,plain16,count32; L4_events.npy = indices where t != L4'; linmodel.npy = FWHT masks).
- Rebuild the instrumented MAME: apply scripts/ns10crypt_dump_instrumentation.patch to src/mame/namco/ns10crypt.cpp,
  make SUBTARGET=namcos10 SOURCES=namco/namcos10.cpp (needs OS=Windows_NT in the msys shell), run with NS10_DUMP=<file>
  and -nodebug (the user's mame.ini enables the debugger, which blocks headless runs).
- Scripts expect to be run from scripts/ with ../data/ptblank3_dump.bin as argv[1] (some take ../mamerun/dump.bin: adjust).

## Round 2 (residual e = t ^ L4' analysed against the whole stream)
Ruled out, with held-out checks where a fit looked good:
- Periodic keystream: no autocorrelation spikes at structural lags (256, 516, 2^k), no spectral line,
  no forbidden triple coincidences (whole stream, lags <= 4096; within page, all lag pairs).
- One-bit hidden register R (e = gate & R): R_i = A ^ R_{i-1}&B with A,B linear (lags 1..12, lag pairs)
  or A quadratic in the 6-word state: held-out accuracy 0.72 vs 0.885 for predicting 0.
- Cubic in a small vocabulary (mask bit 0 at depths 0..11, plaintext bits 11/14/15 at depths 2..10,
  residual lags): looked near-exact (6 inconsistent of 7446) but is memorisation: 9074 distinct patterns
  for rank ~9054, held-out accuracy 0.49.
- Plaintext-selected multiplexer; per-value plaintext tables; carry and sum bits of every word pair
  (+,-) and running sums of p, c, m: nothing beats the plain p[i-4].11 correlation.
- Exact affine factor over 24 and 48 words of history: none.
New facts:
- An enable exists: if no plaintext word at depth 4..96 has any of bits 7,8,11,14,15 set, e = 0 (0 leaks).
  Bit 11 at depth 4 is the strongest driver (e rate 0.34 vs 0.13). Bit 8 alone never enables.
- e decays with words since the last enabling word: ~0.26 for 0..3, ~0.1 at 10, ~0.01 at 25, 0 beyond ~64.
- Words with identical 24-word plaintext windows but different cipher agree on e only 81% (chance 71%):
  e depends strongly on cipher/mask, yet no single cipher/mask bit or word carries information (hash-like).

## Pitfall (important)
GF(2) "inconsistent of free" counts are inflated by duplicate and trivial rows (zero-plaintext regions).
A low count can be pure memorisation. Always score a candidate on held-out rows or held-out patterns.
