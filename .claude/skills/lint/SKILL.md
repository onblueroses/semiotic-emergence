---
name: lint
description: "Run the full 6-step verification pipeline: fmt, clippy, test, machete, deny, doc. Manual: invoke with /lint."
---

# /lint - Full Verification Pipeline

Run all six checks in sequence. Stop on first failure.

## Steps

1. **Format check**
   ```bash
   cargo fmt -- --check
   ```
   If it fails, run `cargo fmt` to fix, then re-check.

2. **Clippy lint**
   ```bash
   cargo lint
   ```
   This is an alias for `cargo clippy --all-targets -- -D warnings`.
   Show the first 3 errors if it fails.

3. **Test**
   ```bash
   cargo ta
   ```
   This is an alias for `cargo nextest run`.
   Show failing test names and first assertion failure.

4. **Unused dependencies**
   ```bash
   cargo machete
   ```
   Finds crates in Cargo.toml that are never imported. False positives for
   optional/feature-gated deps are handled via `[package.metadata.cargo-machete]`.

5. **Supply chain check**
   ```bash
   cargo deny check
   ```
   Checks licenses, advisories, bans, and sources against `deny.toml`.
   Show the first 3 errors if it fails.

6. **Doc check**
   ```bash
   cargo doc --no-deps 2>&1
   ```
   Check for broken doc links or missing types.

## Output

Report results as a summary table:

```
| Step     | Result |
|----------|--------|
| fmt      | pass   |
| clippy   | pass   |
| test     | pass   |
| machete  | pass   |
| deny     | pass   |
| doc      | pass   |
```

On failure, show the step that failed, first 3 errors, and stop. Do not continue to later steps.
