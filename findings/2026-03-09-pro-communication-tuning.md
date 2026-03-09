# Pro-Communication Parameter Tuning - 2026-03-09

Changes motivated by the 100k-gen run (see `2026-03-09-100k-8x-scale.md`) where communication failed to emerge.

## Changes

**neuron_cost: 0.0002 -> 0.00002** (10x cheaper brains)
Brain of 16 now costs 0.56 total energy vs brain of 4 at 0.44. Delta of 0.12 is negligible - evolution can explore big brains freely without metabolic collapse to 4 neurons.

**signal_cost: 0.01 -> 0.0** (free signals)
Removes the -0.34 sender fitness penalty that selected against signaling. Only remaining signal cost is informational (noise degrades neighbor decisions).

**predator_speed: round(3 * scale) -> round(1.5 * scale)** (halved)
At 56 grid: 4 cells/tick (was 8). A warned prey at signal range now has ~6 ticks to react instead of ~3.

**Evasion boost** (new mechanic)
When a prey receives a signal AND a predator is within signal_range, the prey moves 2 cells instead of 1 that tick. Creates direct fitness benefit for signal-responsive prey. Honest signaling favored: only real danger triggers the boost.

## Rationale

The 100k run identified five root causes for communication failure. These changes address three directly:
1. Brain collapse (cheap neurons fix)
2. Signaling penalty (free signals fix)
3. Speed mismatch (halved predator speed + evasion boost fix)

The remaining two (no receiver benefit, symbol monopoly) should resolve as consequences: evasion boost creates receiver benefit, and differentiated receiver responses should create differentiation pressure on symbols.
