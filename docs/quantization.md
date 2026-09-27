# Quantization decisions

## Quantization decision log

The quantization policy is measurement-driven and deliberately conservative:

1. Uniform Q4 was rejected at block 0. Its best complete-block candidate was
   already 13.68% nRMSE, so propagating it through 32 blocks had no plausible
   path to the 1% full-transformer budget.
2. Affine Q8/group-64 passed the original isolated block-0 fixture at 0.837%.
   That justified implementing the packed format and Metal kernel, but not
   generalizing the policy: one four-token block does not expose accumulated
   error or prompt/timestep/resolution sensitivity.
3. Uniform Q8 across all 32 blocks was then tested on six full-transformer
   cases. It failed four, reaching 1.921% final-noise nRMSE. Error at sampled
   internal outputs reached 4.000% around block 15, showing that local block
   success does not compose uniformly.
4. Coarse mixed policies did not solve this. Quantizing either 16-block half,
   all attention matrices, all MLP matrices, or any single matrix role across
   every block exceeded the gate. Even every contiguous eight-block group
   failed; blocks 24-31 were best at 1.130% while blocks 0-7 reached 1.986%.
   The strong depth dependence is why the policy is expressed per block rather
   than only per tensor role.
5. Four-block scans found only blocks 28-31 safe when all seven roles were Q8
   (0.906%). A role refinement then found that attention can extend through
   blocks 24-31. Adding both MLP projection and MLP output in blocks 28-31
   produced the smallest tested passing layout at 0.985%; substituting the MLP
   gate crossed the threshold. This is the current measured frontier, not a
   final release claim.

The calibration uses real prompt embeddings and exact scheduler timesteps,
but deterministic Gaussian latent states rather than a replayed denoising
trajectory. It also emulates packed FP16 values with PyTorch MPS matmul, which
is not bit-exact Metal product rounding. Because the selected frontier has only
a narrow margin, it remains provisional until the mixed packed runtime passes
native Metal comparisons and replayed-trajectory calibration. The simpler
blocks-28-31-only policy remains the fallback; it measured 0.906% at the cost
of about 61 MiB more block-matrix storage.

The full writer intentionally keeps the nine non-block tensors in BF16: they
were not part of the 224-matrix calibration, so quantizing them would extend
the policy beyond its evidence. MLP gates and dense Q/K/V are not quantized;
v3 stores their already-selected FP16 execution operands instead of converting
BF16 on every denoising step. QIPACK1 still accepts v1, v2, and the original
block-0 scope, preserving existing artifacts and fixtures.

The original four-token fixture remains useful as a cheap block-chain
regression. The newer 15-token fixture proves the whole transformer boundary
and the real interleaved token semantics, but it is still not a throughput
claim. The packed mapping is exposed to Metal without copying 12.42 GiB of
weights; only activations, cache arenas, and 128-element Q/K norm vectors are
copied.

Tokenization, text encoding, prompt-sized transformer metadata/RoPE, seeded
noise, denoising, VAE decode, postprocessing, and PNG output now form one
native `generate-256` command. The pinned fox case verifies the complete path;
arbitrary prompts use the same path but naturally have no numeric oracle unless
a matching reference fixture is generated. Production use still needs larger
output-size validation and further kernel optimization. Text weights now use
the exact two-slot layer streamer rather than the rejected affine-Q8 candidates.
The current VAE path intentionally implements the pinned one-frame first-chunk
semantics; temporal continuation and tiled decode remain outside its verified
scope.
