# zarr-python interop tests

Round-trips `zarr-matlab` against a real [`zarr-python`](https://zarr.readthedocs.io/)
install, in both directions (zarr-matlab writes / zarr-python checks, and
zarr-python writes / zarr-matlab checks), for both Zarr v2 and v3. This is
different from the hand-crafted-fixture interop tests in
`zarr-matlab/tests/ZarrArrayV2Test.m` (`testOpenExternallyWrittenArray`,
`testOpenBigEndianArray`), which simulate an externally-written array by
hand-writing `.zarray` JSON — this suite actually invokes `zarr-python`.

**This suite is separate from `zarr-matlab/tests/`:**
- It is **not** run by `runtests('tests')` and is **not** included in the
  packaged toolbox (`.mltbx`) or the release zip.
- It is only exercised in CI on the `ubuntu-latest` leg.
- Run it explicitly, locally, via `runtests('tests-interop')` (see below).

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/). That's
the only setup step — `zarr_interop_cli.py` declares its own dependencies
(`zarr>=3,<4`, `numcodecs>=0.16`) as
[PEP 723 inline script metadata](https://peps.python.org/pep-0723/), so
`uv run zarr_interop_cli.py ...` resolves and caches them automatically on
first use. There's no `requirements.txt` and no separate `pip install` step.

If `uv` (or the dependencies it resolves) isn't available, the tests fail
immediately in `TestClassSetup` with a `zarr:interop:uvMissing` error and
installation instructions — they do not skip silently.

## Running

```matlab
cd zarr-matlab
zarrBuild();  % if you haven't already
results = runtests('tests-interop');
```

Or just the CLI, standalone, for debugging:

```bash
cd zarr-matlab/tests-interop
uv run zarr_interop_cli.py write-array --path /tmp/x --zarr-format 3 \
    --shape 12,9,5 --chunks 4,3,5 --dtype uint16 --compressor zstd
uv run zarr_interop_cli.py check-array --path /tmp/x --zarr-format 3 \
    --shape 12,9,5 --dtype uint16
```

## Layout

- `zarr_interop_cli.py` — the Python side. `write-array` / `check-array` /
  `write-group` / `check-group` subcommands, invoked by the MATLAB test
  classes via `system('uv run ...')`. Arguments are plain flags (no embedded
  JSON) deliberately, since `system()` invokes `cmd.exe` on Windows, which
  doesn't treat `'` as a shell-quoting character the way `sh`/`bash` do.
- `ZarrInteropHelper.m` — shared MATLAB-side helpers used by all three test
  classes below: the CLI path, the deterministic-data formula, and the fixed
  attribute set.
- `ZarrPythonInteropTest.m` — the core `{v2,v3} x {zstd,gzip,blosc}` matrix,
  both directions.
- `ZarrPythonInteropEdgeCasesTest.m` — one-off cases: v2-only compressors
  (zlib, bz2), explicit codec configuration, `'none'`, separators, byte
  order, fill values (including NaN), big-endian, and v3-specific features
  (filters off, a codec sequence, sharding).
- `ZarrPythonInteropGroupTest.m` — a small two-level group hierarchy
  (attributes at both levels), both directions, both formats.

## Deterministic data, no shared files

Both languages independently regenerate identical N-D test arrays from
`(shape, dtype)` alone — no data interchange file, no shared random seed.
Zarr's codec pipeline already makes `arr(i,j,k,...)` addressing agree between
writer and reader regardless of on-disk byte layout (MATLAB's default v3
transpose codec / v2 `order: 'F'` vs. numpy's natural C order), so the two
languages only need to agree on *value as a function of logical index*. See
the module docstring in `zarr_interop_cli.py` for the exact formula.
**`deterministic_data`/`DEFAULT_ATTRS` there and `deterministicData`/
`defaultAttrs` in `ZarrInteropHelper.m` must be kept in sync** — that's why
both are centralized in one place per language rather than duplicated.

## Known gaps this suite found (not fixed here — out of scope)

- **v3 cannot express a `NaN` fill value from MATLAB.** `jsonencode(NaN)`
  renders as JSON `null`; Zarr v2 special-cases that into the spec's `"NaN"`
  string form (`ZarrArray.m`'s `fillValueForV2`), but v3 has no equivalent,
  so `zarrs` rejects a `null` v3 fill_value outright. `testNaNFillValue`
  therefore only covers the v3 direction Python can actually produce
  (Python-write / MATLAB-read).
- **`zarr-python` 3.3.0 rejects a spec-legal Zarr v3 codec shorthand.** When a
  codec has no configuration (e.g. the `bytes` codec for a single-byte dtype
  like `uint8`, which has no endianness to declare), `zarrs` re-serializes it
  using the Zarr v3 spec's bare-string form (`"bytes"` instead of
  `{"name":"bytes"}`) when writing `zarr.json` back to disk. `zarr-python`'s
  codec parser errors on that compact form. Confirmed directly by comparing
  the on-disk `zarr.json` for a `uint8` array against a `uint16` one (whose
  `bytes` codec always carries an `endian` configuration and is therefore
  never minified). The interop tests route around this by using `uint16`
  wherever the test is about compressor/codec selection rather than
  dtype-specific behavior (which is already covered by
  `zarr-matlab/tests/ZarrArrayV2Test.m`'s own dtype-mapping tests).
