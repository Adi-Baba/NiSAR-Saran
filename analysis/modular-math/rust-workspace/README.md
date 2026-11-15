# Rust audit workspace

This workspace separates reusable statistical concerns into crates:

- `audit-core` — paired values and robust summaries.
- `audit-complex` — magnitude and phase checks.
- `audit-distribution` — distribution and shift tests.
- `audit-inference` — scan and conditional inference.
- `audit-ledger` — evidence ledger records.
- `audit-model` — audit model structures.
- `audit-spatial` — spatial support and connected evidence.
- `audit-hdf5` — bounded HDF5 ingestion helpers.
- `audit-integration` and `audit-runner` — composition and executable workflow.

Run `cargo test --workspace` from this directory. `target/` is local build output and is excluded from Git.
