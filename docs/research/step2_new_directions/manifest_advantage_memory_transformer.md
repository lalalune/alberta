# Advantage-Memory Transformer Manifest

The replay-capped advantage-memory transformer runner now writes a
paper-audit manifest inside every `results.json` artifact under
`payload["manifest"]`.

Authoritative fields:

- `manifest.schema_version`: manifest schema identifier.
- `manifest.argv`, `manifest.command`, `manifest.cwd`, `manifest.executable`:
  reconstructed invocation and runtime location.
- `manifest.config.raw_args`: every parsed CLI argument, including paths.
- `manifest.config.effective_config`: effective dataclass config used by the
  runner, including derived `final_window`.
- `manifest.config.prototype_block`: exact prototype-basis block config.
- `manifest.data`: Tiny Shakespeare source URL, file path, byte count, sha256,
  split token index, train/eval token counts, and vocabulary hash.
- `manifest.source`: sha256 and byte counts for the runner and its two local
  helper modules.
- `manifest.git`: commit, branch, describe string, dirty flag, short status,
  status hash, and per-command fallback diagnostics.
- `manifest.environment`: platform, Python executable/version, JAX/JAXLIB/NumPy
  versions, selected JAX/XLA environment variables, backend, and device list.
- `manifest.prng`: root/profile key data plus the seed derivation protocol.
- `manifest.seed_runs`: per-seed key data and per-method train/eval offsets.

Each row in `payload["records"]` also carries `data_offsets` so a metric row can
be traced to its train/eval stream without joining through the top-level
manifest. Baseline, post-FFN memory, and pre-FFN KV memory rows share the same
offsets and initialization key for a seed.

Remaining gaps are outside the runner manifest: it does not capture peak memory,
compile/hot-loop timing splits, hardware energy, immutable lockbox byte ranges,
or dependency lockfiles. Those still need the confirmatory benchmark protocol
before using the result as a paper claim.
