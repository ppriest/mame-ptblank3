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

## Step 1: predictability upper bound (LightGBM, held-out = last 30% of the stream)
Features per word: cipher, plain, mask bits over W words, next cipherword without bit 11; optionally
table-bit, residual and gate lags. Target e on gate-open rows (scripts/gbm.py, gbm2.py).

| rows        | W  | lags | held-out acc | base   | entropy explained |
|-------------|----|------|--------------|--------|-------------------|
| gate-open   | 8  | no   | 0.890        | 0.882  | 36%               |
| gate-open   | 8  | yes  | 0.916        | 0.882  | 51%               |
| gate-open   | 16 | yes  | 0.915        | 0.882  | 51%               |
| gate-open   | 32 | yes  | 0.913        | 0.881  | 50%               |
| enabled     | 8  | no   | 0.811        | 0.796  | 22%               |
| enabled     | 16 | yes  | 0.855        | 0.796  | 41%               |

"enabled" = some plaintext word at depth 4..96 has a bit in 0xc980. The plateau does not move with W.
Caveat: trees cannot represent many-bit XOR, so this does not by itself prove hidden state.
Complementary test (scripts/grouplin2.py): within groups of identical 8-, 16- or 24-word plaintext windows,
e is not XOR-linear in mask bits of words i..i-5 plus t lags (held-out on novel mask states: 0.51 vs base 0.77).
So fixing plaintext context does not reduce e to a parity of recent masks.

## Pitfall: c[i+1] bit 11 leaks the target
c[i+1].11 = p[i+1].11 ^ m[i+1].11 and m[i+1].11 contains t_i. In repeated code p[i+1] is predictable,
so c[i+1].11 reveals t_i. Exclude it (a within-group fit scored 99% with it, 51% without).
Earlier failing fits that included it remain valid failures; LightGBM numbers are unchanged without it.

## Step 2: hidden-register tests
- Linear register fed by plaintext, any width, decaying to zero, output bilinear in register x recent words
  (scripts/bilin.py: plaintext bits at depth d x cipher/plain bits of 3-4 words, exact GF(2), held-out on
  the last 30%): fails. Bits {7,8,11,14,15} depths 4..24 x c of 4 words: held-out 0.57 on enabled rows
  (base 0.80); all 16 bits depths 4..20 x c,p of 3 words: 0.54.
- GRU, 2 layers x 128, one word per step (c, p, m bits, previous t and e, current gate), trained on the
  first 70% (scripts/gru.py): best held-out 0.873 on enabled rows (base 0.796) at epoch 30, then overfits
  (train loss 0.08, held-out falls). Slightly above trees (0.855), far from deterministic.
- Other 15 plaintext and mask bits of the word being decrypted added to the trees (scripts/gbm3.py): no gain (0.855).
- No z3 search was run: every tractable SAT state-machine formulation needs a small input alphabet, and
  e depends on cipher/mask bits in a hash-like way, so any such alphabet discards the signal.

## Where this leaves it
No learner or closed-form family tried generalises past ~87% on enabled rows. Either the hidden logic has
more state or mixing than any family tried, or it depends on something not in the word stream
(e.g. CPLD timing/counter state sampled per read). Data from this one fixed program is unlikely to settle it;
chosen-input experiments on a real board (reflashed NAND) or the original researchers' notes are the next step.

## Round 3: is h a function of plaintext history?
- No. Far-apart enabled rows with identical plaintext over the last W words (scripts/longwin.py):
  among pairs where at least one row has an event, they disagree ~80% of the time for every W
  (W = 8, 24, 48, 96, 160, 256). Overall agreement rises with W only because long repeats sit in quiet code.
  So h depends on the cipher/mask side, not only on plaintext, even with 256 words of identical plaintext.
- Aligned pairs (identical 24/48-word plaintext, >= 1 event): h_A ^ h_B is not Q(m_A) ^ Q(m_B) for a fixed
  linear or quadratic Q over masks of 4-6 words + t lags + c[i+1] without bit 11 (scripts/pairQ.py; held-out 0.50-0.52).
  Within such repeats the mask states span low rank (~42 for 6 words): the masks are fixed by the linear
  recurrence from earlier state, so "mask dependence" is dependence on the state at earlier times.
- Shift-register-of-latched-products family: h = XOR of p[d].b & m[d+off].k (same word or +-2 words, depths 4..20)
  (scripts/samedepth.py): fails, 0.54-0.55 held-out on enabled rows (base 0.79).

## Round 4: algebraic annihilators (scripts/annih.py)
Exact GF(2) rank test on enabled, gate-open rows: does a polynomial of degree <= d vanish on every event row
(or every non-event row) without vanishing on all rows? Any hit is verified on all 186,810 event rows.
Positive control (gate-closed rows included, scripts/annih_ctrl.py): recovers m3.0 + m2.0*m3.0, the gate, exactly.
No annihilator of events or of non-events for:
| variables                                   | degree | monomials | data rank |
|---------------------------------------------|--------|-----------|-----------|
| m words 0..5 + t lags 1..4                  | 1, 2   | 5051      | 3321      |
| c, p words 0..11                            | 1      | 385       | 228       |
| c, p words 0..3                             | 2      | 8257      | 4186      |
| c, p words 0..7                             | 2      | 32897     | 12720     |
| m words 4..11                               | 2      | 8257      | 5152      |
| m 0..1 + p bits 11/14/15 depths 3..10 + t 1..6 | 2   | 1954      | 1771      |
| m words 0..2 + t 1..4                       | 3      | 23479     | 15226     |
| m 0..1 + p bits 11/14/15 depths 3..8 + t 1..4 | 3    | 26290     | 22152     |
| m 2..3 + p bits 11/14/15 depths 3..8 + t 1..6 | 3    | 29317     | 24806     |
Consequences: h is not an OR of products of linear forms over these bits, not a choice among 2 or 3 linear
forms selected by state expressible in these bits, and has algebraic immunity > 3 over them.
- Comparators (A > B on full or low 4/8/11/12 bits), carries and borrows of every pair of c, p, m words 0..8,
  on enabled rows only (scripts/arith2.py): best |corr| 0.093, below raw p[i-4].11 (0.125).

## Round 5: register templates
Start of stream (scripts/head.py): the residual is 1 at words 0, 1, 2, 4 with no history, the first plaintext
words (0x5cbc, 0x0800) carry enabling bits at depth 0 and 1, and the gate is never violated in the first 400 words.
The depth-4 rule is therefore a steady-state property; the hidden state is non-empty or behaves differently at
activation.

Plaintext-only register rules, output solved exactly as XOR_k R[k] & L_k(c, p of 2 words) (scripts/rtempl.py),
held-out on the last 30%, enabled-row baseline 0.798:
- OR-enable shift register (masks 0xc980, 0x0800, 0x4800, 0xc800; depths 4..19): best 0.66
- words-since-enable (binary or one-hot): best 0.66; saturating charge counters: best 0.63
FIFO of words latched at enabling plaintext (scripts/fifo.py), linear in FIFO contents, with or without an
activity one-hot gate, latching m, c or p, depth 1..8: best 0.64.

z3 (scripts/z3nlfsr.py, z3nlfsr2.py). Encoding verified each time: planted random instance with parameters
pinned is SAT and predicts the next 20000 words exactly; real data with the same pin is UNSAT.
- NLFSR, new bit = parity(a.p[i-4]) ^ (parity(b.R) & parity(c.p[i-4])), output XOR_k R[k] & parity(W_k . m[i], m[i-1]):
  UNSAT on 300 real words for n = 4, 6, 8, 12, 16 (unknown at 24, 32 after 900 s). Cause: with zero plaintext the
  register empties in n words, but events persist ~33 words into zero plaintext.
- Self-sustaining variant (adds parity(l.R) and parity(d.R) & parity(f.R) to the new bit): unknown after 300 s at
  n = 6 on 300 words (and 300 words underdetermine its ~254 unknowns).
- Same, output over m[i] only (NX=16, ~150 unknowns at n = 6): unknown after 3600 s at n = 4 and n = 6 on 600
  words from 288896. Pinned planted instance at n = 6 is SAT and predicts the next 20000 words at 1.0000 (base
  0.5123), so the encoding is sound; z3 cannot decide this template at this size.
- Incremental z3 (scripts/z3cegar.py): transitions asserted over all 600 words, output constraints added from the
  candidate's wrong words, 8 per iteration. On planted instances (n = 4 and 6, 4 seeds each, 900 s) every run stalled
  at 72-80 of 448 constraints after 5-6 iterations with z3 unknown. The limit is the depth of the unrolled nonlinear
  register, not the constraint count: this template is beyond z3 at these sizes even when a solution exists.
- CryptoMiniSat (scripts/cms_nlfsr.py; parities as native XOR clauses). Encoding check at n = 4 on 300 words: planted
  data with all parameters pinned SAT in 0 s, next 20000 words predicted at 1.0000 (base 0.5316); real data, same pin,
  UNSAT in 0 s. Unpinned planted instances: unknown after 900 s at n = 4 and 6 on 300 and 600 words.
  Pinning groups of the planted parameters (n = 4, 300 words, 300 s): W alone, or a,b,c,l,d,f alone, SAT in 0 s;
  a,c / b,c / l,d,f / l,d,f,b / l,d,f,b,c unknown; W columns 0..11 pinned SAT in 0 s, 0..7 or fewer unknown.
  The hardness is the product of unknown state and unknown output masks: once either is fixed the rest is linear.
  Splitting the search by pinning W needs about 48 bits at n = 4, too many to enumerate.
- Exhaustive search with a, c restricted to plaintext bits 0xc980 (scripts/brute_prep.py, brute.c): every
  (a, c, l, d, f, b, init) is simulated and W solved by online GF(2) elimination over the 448 gated words of
  288896..289496; survivors are tested on the next 20000 words. Planted check (n = 4): the planted candidate and
  5 equivalent forms found, each 1.0000 on 15033 held-out gated words; 2^24 candidates in 25 s on one core.
  Real data, n = 4 (2^30 candidates, covers n <= 4): none fits; deepest reject at 123 of 448 gated words (64 unknowns).
- Real data, n = 5 (2^35 candidates, covers n <= 5): none fits; deepest reject at 142 of 448 gated words (80 unknowns).
  1 h 56 min on 20 processes (log timestamps).
- Same n = 4 search with the output over other words (brute_prep.py XSRC): m[i-2], m[i-3], c[i] none fit, deepest
  reject 124, 121, 124 of 448. p[i-4]: none fit, deepest 9; a zero p[i-4] forces output 0, and gated words with
  e = 1 and p[i-4] = 0 are common, so this variant is excluded almost at once.
