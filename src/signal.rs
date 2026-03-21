use crate::brain::{softmax, SIGNAL_OUTPUTS};
use crate::world::wrap_coord;
#[cfg(test)]
use crate::world::wrap_delta;

pub const NUM_SYMBOLS: usize = 6;

#[derive(Clone, Debug)]
pub struct Signal {
    pub x: i32,
    pub y: i32,
    pub symbol: u8,
    pub tick_emitted: u32,
}

/// Spatial grid for fast signal lookup. Cells are coarser than the world grid
/// (`cell_size` >= 1), so multiple world cells map to one signal cell.
/// Cell size is chosen to evenly divide `grid_size` (no uneven edge cells).
/// Rebuilt each tick from the active signal list.
///
/// Struct-of-arrays layout for signal data: x, y, and symbol stored in
/// separate contiguous arrays for cache-friendly access in the inner loop.
/// Reception searches cells in ring order (center outward) with per-symbol
/// early exit to skip outer rings when all symbols already found closer.
pub struct SignalGrid {
    /// Signal x coordinates, grouped by cell then by symbol within cell.
    sig_x: Vec<i16>,
    /// Signal y coordinates, grouped by cell then by symbol within cell.
    sig_y: Vec<i16>,
    /// Signal symbols, grouped by cell then by symbol within cell.
    sig_sym: Vec<u8>,
    /// Per-cell (start, len) into the signal arrays.
    offsets: Vec<(u32, u16)>,
    /// Per-cell cumulative symbol counts. Entry `[ci][s]` = offset within
    /// cell `ci` where symbol `s` starts. `[ci][NUM_SYMBOLS]` = cell len.
    /// Enables per-symbol scans with independent early exit.
    sym_offsets: Vec<[u16; NUM_SYMBOLS + 1]>,
    cell_size: i32,
    cells_per_axis: i32,
    cells_radius: i32,
    grid_size: i32,
}

impl SignalGrid {
    pub fn new(grid_size: i32, signal_range: f32) -> Self {
        // Find largest divisor of grid_size <= signal_range/2 for ~5x5 neighborhood.
        let target = (signal_range as i32 / 2).max(1);
        let cell_size = (1..=target)
            .rev()
            .find(|&cs| grid_size % cs == 0)
            .unwrap_or(1);
        let cells_per_axis = grid_size / cell_size;
        let cells_radius = (signal_range / cell_size as f32).ceil() as i32;
        let total = (cells_per_axis * cells_per_axis) as usize;
        Self {
            sig_x: Vec::new(),
            sig_y: Vec::new(),
            sig_sym: Vec::new(),
            offsets: vec![(0, 0); total],
            sym_offsets: vec![[0; NUM_SYMBOLS + 1]; total],
            cell_size,
            cells_per_axis,
            cells_radius,
            grid_size,
        }
    }

    pub fn rebuild(&mut self, signals: &[Signal], current_tick: u32) {
        let total_cells = self.offsets.len();
        // Count signals per cell AND per symbol within cell
        for o in &mut self.offsets {
            *o = (0, 0);
        }
        for so in &mut self.sym_offsets {
            *so = [0; NUM_SYMBOLS + 1];
        }
        let mut count = 0u32;
        for sig in signals {
            if sig.tick_emitted >= current_tick {
                continue;
            }
            let cx = sig.x.rem_euclid(self.grid_size) / self.cell_size;
            let cy = sig.y.rem_euclid(self.grid_size) / self.cell_size;
            let ci = (cy * self.cells_per_axis + cx) as usize;
            self.offsets[ci].1 += 1;
            let s = (sig.symbol as usize).min(NUM_SYMBOLS - 1);
            self.sym_offsets[ci][s] += 1;
            count += 1;
        }
        // Prefix sum for cell start offsets
        let mut running = 0u32;
        for ci in 0..total_cells {
            self.offsets[ci].0 = running;
            running += u32::from(self.offsets[ci].1);
        }
        // Convert per-cell symbol counts to cumulative offsets within cell
        for ci in 0..total_cells {
            let so = &mut self.sym_offsets[ci];
            let mut cum = 0u16;
            for s in 0..NUM_SYMBOLS {
                let c = so[s];
                so[s] = cum;
                cum += c;
            }
            so[NUM_SYMBOLS] = cum;
        }
        // Fill SoA arrays sorted by symbol within each cell (counting sort).
        // Use a copy of sym_offsets as write cursors.
        let n = count as usize;
        self.sig_x.clear();
        self.sig_x.resize(n, 0);
        self.sig_y.clear();
        self.sig_y.resize(n, 0);
        self.sig_sym.clear();
        self.sig_sym.resize(n, 0);
        let mut cursors = self.sym_offsets.clone();
        for sig in signals {
            if sig.tick_emitted >= current_tick {
                continue;
            }
            let cx = sig.x.rem_euclid(self.grid_size) / self.cell_size;
            let cy = sig.y.rem_euclid(self.grid_size) / self.cell_size;
            let ci = (cy * self.cells_per_axis + cx) as usize;
            let s = (sig.symbol as usize).min(NUM_SYMBOLS - 1);
            let pos = self.offsets[ci].0 as usize + cursors[ci][s] as usize;
            self.sig_x[pos] = sig.x as i16;
            self.sig_y[pos] = sig.y as i16;
            self.sig_sym[pos] = sig.symbol;
            cursors[ci][s] += 1;
        }
    }
}

/// Compute received signal strengths for a prey at (rx, ry) on the given tick.
/// Returns strength per symbol `[0..NUM_SYMBOLS]`. Only signals from previous ticks
/// are receivable (1-tick delay). Strongest-per-symbol wins.
#[cfg(test)]
pub fn receive(
    signals: &[Signal],
    rx: i32,
    ry: i32,
    current_tick: u32,
    signal_range: f32,
    grid_size: i32,
) -> [f32; NUM_SYMBOLS] {
    let mut strengths = [0.0_f32; NUM_SYMBOLS];
    for sig in signals {
        if sig.tick_emitted >= current_tick {
            continue; // 1-tick delay: only past signals
        }
        let dx = wrap_delta(rx, sig.x, grid_size) as f32;
        let dy = wrap_delta(ry, sig.y, grid_size) as f32;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > signal_range {
            continue;
        }
        let strength = 1.0 - dist / signal_range;
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && strength > strengths[sym] {
            strengths[sym] = strength;
        }
    }
    strengths
}

#[derive(Clone, Copy, Debug)]
pub struct ReceivedSignal {
    pub strength: f32,
    pub dx: f32,
    pub dy: f32,
}

/// Compute detailed received signals using spatial grid for O(nearby) instead of O(all).
/// Per-symbol ring scan: each symbol gets its own ring traversal with independent early
/// exit, so common nearby symbols terminate fast while only rare symbols scan far.
/// Signals are sorted by symbol within each cell (counting sort in rebuild).
#[allow(clippy::similar_names)]
pub fn receive_detailed_grid(
    grid: &SignalGrid,
    rx: i32,
    ry: i32,
    grid_size: f32,
    signal_range: f32,
) -> [ReceivedSignal; NUM_SYMBOLS] {
    let grid_size_i = grid_size as i32;
    let half_gs = grid_size_i >> 1;
    let range_sq = signal_range * signal_range;
    let range_i = signal_range as i32;
    let cs = grid.cell_size;
    let cpa = grid.cells_per_axis;
    let r_max = grid.cells_radius;

    let cx = rx.rem_euclid(grid_size_i) / cs;
    let cy = ry.rem_euclid(grid_size_i) / cs;

    let mut best_dist_sq = [f32::MAX; NUM_SYMBOLS];
    let mut best_dx = [0.0_f32; NUM_SYMBOLS];
    let mut best_dy = [0.0_f32; NUM_SYMBOLS];

    for sym in 0..NUM_SYMBOLS {
        for ring in 0..=r_max {
            // Per-symbol early exit: once this symbol is found closer than any
            // signal in the next ring could be, skip remaining rings.
            if ring >= 2 {
                let min_gap = ((ring - 1) * cs) as f32;
                if best_dist_sq[sym] <= min_gap * min_gap {
                    break;
                }
            }

            for dcy in -ring..=ring {
                for dcx in -ring..=ring {
                    if ring > 0 && dcx.abs().max(dcy.abs()) != ring {
                        continue;
                    }
                    let ncx = wrap_coord(cx + dcx, cpa) as usize;
                    let ncy = wrap_coord(cy + dcy, cpa) as usize;
                    let ci = ncy * cpa as usize + ncx;
                    let cell_start = grid.offsets[ci].0 as usize;
                    let so = &grid.sym_offsets[ci];
                    let s = cell_start + so[sym] as usize;
                    let end = cell_start + so[sym + 1] as usize;

                    for k in s..end {
                        let ddx = {
                            let d = i32::from(grid.sig_x[k]) - rx;
                            if d > half_gs {
                                d - grid_size_i
                            } else if d < -half_gs {
                                d + grid_size_i
                            } else {
                                d
                            }
                        };
                        if ddx > range_i || ddx < -range_i {
                            continue;
                        }
                        let ddy = {
                            let d = i32::from(grid.sig_y[k]) - ry;
                            if d > half_gs {
                                d - grid_size_i
                            } else if d < -half_gs {
                                d + grid_size_i
                            } else {
                                d
                            }
                        };
                        if ddy > range_i || ddy < -range_i {
                            continue;
                        }
                        let dxf = ddx as f32;
                        let dyf = ddy as f32;
                        let dist_sq = dxf * dxf + dyf * dyf;
                        if dist_sq >= range_sq {
                            continue;
                        }
                        if dist_sq < best_dist_sq[sym] {
                            best_dist_sq[sym] = dist_sq;
                            best_dx[sym] = dxf;
                            best_dy[sym] = dyf;
                        }
                    }
                }
            }
        }
    }

    std::array::from_fn(|s| {
        if best_dist_sq[s] < f32::MAX {
            let dist = best_dist_sq[s].sqrt();
            ReceivedSignal {
                strength: 1.0 - dist / signal_range,
                dx: best_dx[s] / grid_size,
                dy: best_dy[s] / grid_size,
            }
        } else {
            ReceivedSignal {
                strength: 0.0,
                dx: 0.0,
                dy: 0.0,
            }
        }
    })
}

/// Fallback: compute detailed received signals by scanning all signals (no grid).
#[cfg(test)]
pub fn receive_detailed(
    signals: &[Signal],
    rx: i32,
    ry: i32,
    current_tick: u32,
    grid_size: f32,
    signal_range: f32,
) -> [ReceivedSignal; NUM_SYMBOLS] {
    let grid_size_i = grid_size as i32;
    let range_sq = signal_range * signal_range;
    let mut result = std::array::from_fn::<_, NUM_SYMBOLS, _>(|_| ReceivedSignal {
        strength: 0.0,
        dx: 0.0,
        dy: 0.0,
    });
    for sig in signals {
        if sig.tick_emitted >= current_tick {
            continue;
        }
        let dx = wrap_delta(rx, sig.x, grid_size_i) as f32;
        let dy = wrap_delta(ry, sig.y, grid_size_i) as f32;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq > range_sq {
            continue;
        }
        let dist = dist_sq.sqrt();
        let strength = 1.0 - dist / signal_range;
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && strength > result[sym].strength {
            result[sym].strength = strength;
            result[sym].dx = dx / grid_size;
            result[sym].dy = dy / grid_size;
        }
    }
    result
}

/// Decide whether to emit a signal based on softmax of NN signal outputs.
/// Returns `Some(symbol)` if max softmax probability > `1/NUM_SYMBOLS`, None otherwise.
pub fn maybe_emit(signal_outputs: &[f32; SIGNAL_OUTPUTS], threshold: f32) -> Option<u8> {
    let probs = softmax(signal_outputs);
    let (max_idx, &max_prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    if max_prob > threshold {
        Some(max_idx as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SIGNAL_RANGE: f32 = 8.0;
    const TEST_GRID_SIZE: i32 = 20;

    #[test]
    fn delay_blocks_same_tick() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 0,
            tick_emitted: 5,
        }];
        let strengths = receive(&signals, 1, 0, 5, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!(strengths[0].abs() < 1e-6);
    }

    #[test]
    fn previous_tick_received() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 1,
            tick_emitted: 4,
        }];
        let strengths = receive(&signals, 0, 0, 5, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!((strengths[1] - 1.0).abs() < 1e-6); // same cell = max strength
    }

    #[test]
    fn out_of_range_not_received() {
        let signals = vec![Signal {
            x: 0,
            y: 0,
            symbol: 0,
            tick_emitted: 0,
        }];
        // Place receiver 10 cells away (> SIGNAL_RANGE of 8)
        let strengths = receive(&signals, 10, 0, 1, TEST_SIGNAL_RANGE, TEST_GRID_SIZE);
        assert!(strengths[0].abs() < 1e-6);
    }

    #[test]
    fn receive_detailed_returns_direction() {
        let signals = vec![Signal {
            x: 5,
            y: 3,
            symbol: 0,
            tick_emitted: 4,
        }];
        let result = receive_detailed(&signals, 2, 1, 5, 20.0, TEST_SIGNAL_RANGE);
        assert!(result[0].strength > 0.0);
        assert!((result[0].dx - 3.0 / 20.0).abs() < 1e-6);
        assert!((result[0].dy - 2.0 / 20.0).abs() < 1e-6);
        // Other symbols should be zero
        assert!(result[1].strength.abs() < 1e-6);
        assert!(result[2].strength.abs() < 1e-6);
    }

    #[test]
    fn emit_uniform_returns_none() {
        // Equal logits -> softmax = 1/6 each -> max == threshold -> no emission
        let outputs = [0.0_f32; SIGNAL_OUTPUTS];
        assert!(maybe_emit(&outputs, 1.0 / NUM_SYMBOLS as f32).is_none());
    }

    #[test]
    fn emit_concentrated_returns_symbol() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[3] = 5.0; // symbol 3 dominant
        assert_eq!(maybe_emit(&outputs, 1.0 / NUM_SYMBOLS as f32), Some(3));
    }

    #[test]
    fn emit_slight_difference_returns_some() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[1] = 0.01; // barely above others
        assert_eq!(maybe_emit(&outputs, 1.0 / NUM_SYMBOLS as f32), Some(1));
    }

    #[test]
    fn high_threshold_suppresses_emission() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[1] = 0.01; // barely above others, would emit at default 1/6
                           // At threshold 0.5, softmax(0.01 vs 5 zeros) ~ 0.169, below 0.5
        assert!(maybe_emit(&outputs, 0.5).is_none());
    }

    #[test]
    fn grid_matches_brute_force() {
        // Test on both small (20) and large (72) grids with different ranges
        for &(grid_size, signal_range) in &[(20, 8.0_f32), (72, 16.0), (72, 28.8)] {
            let current_tick = 10;
            let signals: Vec<Signal> = (0..200)
                .map(|i| Signal {
                    x: (i * 7) % grid_size,
                    y: (i * 13) % grid_size,
                    symbol: (i % 6) as u8,
                    tick_emitted: current_tick - 1 - (i as u32 % 3),
                })
                .collect();
            let mut grid = SignalGrid::new(grid_size, signal_range);
            grid.rebuild(&signals, current_tick);
            for rx in [0, 1, grid_size / 4, grid_size / 2, grid_size - 1] {
                for ry in [0, 1, grid_size / 3, grid_size / 2, grid_size - 1] {
                    let brute = receive_detailed(
                        &signals,
                        rx,
                        ry,
                        current_tick,
                        grid_size as f32,
                        signal_range,
                    );
                    let fast = receive_detailed_grid(&grid, rx, ry, grid_size as f32, signal_range);
                    for s in 0..NUM_SYMBOLS {
                        assert!(
                            (brute[s].strength - fast[s].strength).abs() < 1e-6,
                            "grid={grid_size} range={signal_range} at ({rx},{ry}) sym {s}: brute={} grid={}",
                            brute[s].strength,
                            fast[s].strength
                        );
                    }
                }
            }
        }
    }
}
