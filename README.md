# STARK-Based DFA Membership Proof with Poseidon Hashing

A research implementation of a **STARK (Scalable Transparent ARgument of Knowledge)** proof system that proves membership of a string in a language recognized by a **Deterministic Finite Automaton (DFA)**, jointly with the integrity of a **Poseidon hash** computation over that string.

This project was developed as a TIPE (Travaux d'Initiative Personnelle Encadrés) research work.

---

## Overview

The system allows a **prover** to convince a **verifier** — without revealing the string itself — that:

1. The string is accepted by a given DFA (i.e., it belongs to the recognized language).
2. The Poseidon hash of the string matches a publicly committed digest.

The proof is **succinct** (~772 KB), **non-interactive** (via the Fiat-Shamir heuristic), and **transparent** (no trusted setup required).

> **Known Limitation:** The current implementation is cryptographically incomplete. There is no binding between the word fed into the automata and the word fed into the hash function. A malicious prover could use two different strings for the two components. Fixing this requires a shared witness commitment linking both traces.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          PROVER                                │
│                                                                │
│  word ──► DFA Trace ──► State/Character Polynomials           │
│       │                                                        │
│       └──► Poseidon Trace ──► Trace Polynomials               │
│                                                                │
│  Constraint Polynomials (automata + Poseidon)                  │
│       └──► Composition Polynomial (cp)                        │
│               └──► FRI Protocol ──► Proof                     │
└────────────────────────────────────────────────────────────────┘
                              │  proof.txt
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                         VERIFIER                               │
│                                                                │
│  Read proof ──► Replay Fiat-Shamir randomness                  │
│              ──► Verify Merkle decommitments                   │
│              ──► Check constraint evaluations at queries       │
│              ──► Verify FRI layer consistency                  │
│              ──► ACCEPT or REJECT                              │
└────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
.
├── prover.ipynb          # Main prover notebook
├── verifier.ipynb        # Main verifier notebook
├── automata.json         # DFA definition (JSON)
├── proof.txt             # Generated proof transcript
├── text.txt              # Example input text
├── field.py              # Finite field arithmetic (F_{3·2^30+1})
├── polynomial.py         # Polynomial arithmetic (NTT-optimized)
├── poseidon.py           # Poseidon hash (Hades permutation)
├── merkle.py             # Merkle tree commitment scheme
├── channel.py            # Fiat-Shamir channel
├── utils.py              # Automata class, prime sieve
└── constants.py          # Protocol parameters
```

---

## The DFA

The automata defined in `automata.json` has **15 states** (q0–q14) over an alphabet of printable ASCII characters. It recognizes strings that contain the substring **"STARKS are fun"**:

```
q0 ──S──► q1 ──T──► q2 ──A──► q3 ──R──► q4 ──K──► q5
q5 ──S──► q6 ──(space)──► q7 ──a──► q8 ──r──► q9 ──e──► q10
q10 ──(space)──► q11 ──f──► q12 ──u──► q13 ──n──► q14 (accept, sink)
```

State q14 is the accepting sink state: once reached, all characters loop back to q14. Any character that does not continue the pattern resets to q0 (rejecting sink behavior on wrong transitions).

States are encoded as distinct primes. Characters are also encoded as distinct primes (disjoint set from state primes). This prime encoding makes it possible to encode the transition function as a polynomial: `T(state_prime · char_prime) = next_state_prime`.

---

## Protocol Description

### 1. Trace Generation

The prover runs the DFA on the input word, producing a sequence of (step, state, character) triples. Simultaneously, it computes the Poseidon hash of the word, obtaining an execution trace of the sponge permutation.

### 2. Polynomial Interpolation

Both traces are interpolated into polynomials over the finite field `F_p` (where `p = 3·2^30 + 1`):

| Polynomial | Description |
|---|---|
| `state_poly(x)` | Encodes the DFA state at each step |
| `char_poly(x)` | Encodes the input character at each step |
| `transition_poly(s·c)` | Encodes the DFA transition function |
| `ptrace_poly[j](x)` | Encodes the j-th column of the Poseidon sponge trace |
| `arc_poly[j](x)` | Encodes round constants (ARC) of the Poseidon permutation |
| `hash_input_poly[j](x)` | Encodes the hash input absorption |

### 3. Constraint Polynomials

The prover constructs **29 constraint polynomials** (divided by their respective vanishing polynomials) that must all be low-degree if and only if the traces are valid:

**Automata constraints:**
- **Initial state**: `(state_poly(x) - q0_prime) / (x - 1) ≡ 0`
- **Character validity**: `∏_{c ∈ Σ} (char_poly(x) - prime(c)) / Z_G(x) ≡ 0`
- **Accepting state**: `∏_{f ∈ F} (state_poly(x) - prime(f)) / (x - x_{last}) ≡ 0`
- **Step transition**: `(state_poly(g·x) - transition_poly(state_poly(x)·char_poly(x))) / Z_G(x) ≡ 0`

**Poseidon constraints:**
- **Initial state**: `ptrace_poly[j](x) / (x - 1) ≡ 0` for each column j
- **Full rounds**: `(ptrace[j](g·x) - MDS·(ptrace[j](x)+hash+arc)^5) / X_G ≡ 0`
- **Partial rounds**: `(ptrace[j](g·x) - MDS·S_partial(ptrace[j](x)+hash+arc)) / Z_G ≡ 0`
- **Output**: `(ptrace_poly[0](x) - hash_digest) / (x - z_{last}) ≡ 0`

### 4. Composition and FRI

The constraint polynomials are combined into a single **composition polynomial** using random linear combination (sampled via Fiat-Shamir):

```
cp(x) = Σ α_i · constraint_i(x)
```

The **FRI (Fast Reed-Solomon IOP of Proximity)** protocol then proves that `cp` has degree less than the trace length, without the verifier evaluating it directly. FRI reduces the polynomial to a constant through successive folding steps, each halving the degree.

### 5. Merkle Commitments

All polynomial evaluations over the blowup domain are committed via **Merkle trees**. The verifier checks Merkle decommitment paths for `N_QUERY = 12` randomly sampled positions.

---

## Cryptographic Components

### Finite Field

`F_p` where `p = 3 · 2^30 + 1 = 3,221,225,473`

This is a STARK-friendly prime: its multiplicative group has a large 2-adic subgroup (up to order 2^30), enabling efficient NTT-based polynomial multiplication.

### Poseidon Hash (Hades Permutation)

| Parameter | Value |
|---|---|
| State width (t) | 8 elements |
| Rate (r) | 7 |
| Capacity (c) | 1 |
| Full rounds (r_f) | 8 |
| Partial rounds (r_p) | 57 |
| S-box | x^5 |
| MDS matrix | Cauchy matrix |

The Hades design alternates **full rounds** (S-box on all t elements) with **partial rounds** (S-box on first element only), balancing security and proof efficiency.

### Channel (Fiat-Shamir)

The `Channel` class implements the Fiat-Shamir heuristic: it maintains a SHA-256 state that is updated each time the prover sends a message, and uses it to derive verifier randomness deterministically. This converts the interactive protocol into a non-interactive proof.

### Merkle Tree

A binary Merkle tree built with SHA-256. Authentication paths are logarithmic in the domain size. The verifier checks decommitments to confirm that queried evaluations are consistent with the committed roots.

### Polynomial Arithmetic

The `Polynomial` class supports:
- **NTT multiplication** for polynomials with degree ≥ 64 (Cooley-Tukey FFT over `F_p`)
- **Schoolbook multiplication** for small polynomials
- **Synthetic division** for fast division by linear factors (used in Lagrange interpolation)
- **Horner evaluation**

---

## Protocol Parameters

| Parameter | Value | Description |
|---|---|---|
| Field prime | 3·2^30 + 1 | STARK-friendly prime |
| Blowup factor | 128 | Rate ρ = 1/128 (Reed-Solomon rate) |
| FRI queries | 12 | Number of consistency checks |
| Constraints | 29 | Total number of constraint polynomials |
| Claimed security | ~64 bits | P(cheating) ≈ 5.4 × 10^-20 |

---

## Performance

Running on the example string `"I think STARKS are fun to learn"`:

| Metric | Value |
|---|---|
| Proof generation | ~1015 seconds |
| Verification | ~0.25 seconds |
| Trace length (Poseidon) | 326 steps |
| Evaluation domain size | 65,536 |
| Proof size | ~772 KB |

The prover is slow due to unoptimized Python polynomial interpolation and evaluation over the blowup domain. The verifier is fast as it only evaluates at O(N_QUERY) positions.

---

## Running the Project

**Requirements:**
```
python >= 3.10
jupyter
numpy
tqdm
```

**Run the prover:**
Open and execute `prover.ipynb` in Jupyter. It will write the proof to `proof.txt`.

**Run the verifier:**
Open and execute `verifier.ipynb` in Jupyter. It reads `proof.txt` and verifies the proof.

---

## Theoretical Background

**STARKs** are probabilistic proof systems where the prover encodes a computation trace as a polynomial, uses a low-degree extension, and the verifier probabilistically checks consistency via FRI. Security relies on the distance between low-degree polynomials: a cheating prover would need to produce an evaluation consistent with a valid degree-bounded polynomial, which is statistically unlikely when probed at random positions.

**FRI** (Ben-Sasson et al., 2018) is the core proximity test. It iteratively halves the polynomial degree by splitting into even/odd parts and combining them with a random coefficient: `f_next(x) = f_even(x^2) + β · f_odd(x^2)`. After `log2(degree)` rounds, the polynomial reduces to a constant, which the prover reveals directly.

**The transition function encoding** uses the prime map: each state and character is assigned a distinct prime, so `state_prime · char_prime` is a unique index into the transition table, and the transition can be expressed as a polynomial evaluated at this product.

---

## Known Issues and Future Work

- **Missing witness binding**: The word used for the DFA trace and the word used as Poseidon hash input are not cryptographically linked. A correct implementation would commit to a shared encoding of the word and use it as input to both traces.
- **Performance**: Python-based NTT and Lagrange interpolation are slow. A production system would use a compiled backend.
- **Blowup factor**: Currently hardcoded to 128. It should be derived dynamically from the maximum constraint degree to ensure soundness.
- **Out-of-domain sampling**: The verifier uses in-domain queries only. True STARK soundness benefits from out-of-domain sampling (DEEP-FRI).

---

## References

- Ben-Sasson, E., Bentov, I., Horesh, Y., & Riabzev, M. (2018). *Scalable, transparent, and post-quantum secure computational integrity.* (STARK paper)
- Grassi, L., Khovratovich, D., Rechberger, C., Roy, A., & Schofnegger, M. (2021). *Poseidon: A new hash function for zero-knowledge proof systems.*
- StarkWare Industries. *STARK Math* tutorial series. (Source for `channel.py` and `merkle.py`)
- Gabizon, A., & Williamson, Z. J. (2020). *plookup: A simplified polynomial protocol for lookup tables.*
