use crate::brain::{softmax, SIGNAL_OUTPUTS};
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
pub struct SignalGrid {
    /// Each cell holds indices into the signal vec.
    cells: Vec<Vec<u16>>,
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
            cells: vec![Vec::new(); total],
            cell_size,
            cells_per_axis,
            cells_radius,
            grid_size,
        }
    }

    pub fn rebuild(&mut self, signals: &[Signal], current_tick: u32) {
        for cell in &mut self.cells {
            cell.clear();
        }
        for (i, sig) in signals.iter().enumerate() {
            if sig.tick_emitted >= current_tick {
                continue;
            }
            let cx = sig.x.rem_euclid(self.grid_size) / self.cell_size;
            let cy = sig.y.rem_euclid(self.grid_size) / self.cell_size;
            let ci = (cy * self.cells_per_axis + cx) as usize;
            self.cells[ci].push(i as u16);
        }
    }

    /// Iterate over signal indices in cells within `cells_radius` of (rx, ry).
    fn nearby_indices(&self, rx: i32, ry: i32) -> impl Iterator<Item = u16> + '_ {
        let cx = rx.rem_euclid(self.grid_size) / self.cell_size;
        let cy = ry.rem_euclid(self.grid_size) / self.cell_size;
        let cpa = self.cells_per_axis;
        let r = self.cells_radius;
        (-r..=r).flat_map(move |dy| {
            (-r..=r).flat_map(move |dx| {
                let ncx = (cx + dx).rem_euclid(cpa);
                let ncy = (cy + dy).rem_euclid(cpa);
                let ci = (ncy * cpa + ncx) as usize;
                self.cells[ci].iter().copied()
            })
        })
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
/// Defers sqrt: tracks closest signal per symbol by `dist_sq`, computes strength only
/// for the 6 winners at the end (saves N-6 sqrt calls per receiver).
#[allow(clippy::similar_names)]
pub fn receive_detailed_grid(
    signals: &[Signal],
    grid: &SignalGrid,
    rx: i32,
    ry: i32,
    grid_size: f32,
    signal_range: f32,
) -> [ReceivedSignal; NUM_SYMBOLS] {
    let grid_size_i = grid_size as i32;
    let range_sq = signal_range * signal_range;
    let range_i = signal_range as i32;
    let mut best_dist_sq = [f32::MAX; NUM_SYMBOLS];
    let mut best_dx = [0.0_f32; NUM_SYMBOLS];
    let mut best_dy = [0.0_f32; NUM_SYMBOLS];
    for idx in grid.nearby_indices(rx, ry) {
        let sig = &signals[idx as usize];
        let dx = wrap_delta(rx, sig.x, grid_size_i);
        if dx > range_i || dx < -range_i {
            continue;
        }
        let dy = wrap_delta(ry, sig.y, grid_size_i);
        if dy > range_i || dy < -range_i {
            continue;
        }
        let dxf = dx as f32;
        let dyf = dy as f32;
        let dist_sq = dxf * dxf + dyf * dyf;
        if dist_sq > range_sq {
            continue;
        }
        let sym = sig.symbol as usize;
        if sym < NUM_SYMBOLS && dist_sq < best_dist_sq[sym] {
            best_dist_sq[sym] = dist_sq;
            best_dx[sym] = dxf;
            best_dy[sym] = dyf;
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
pub fn maybe_emit(signal_outputs: &[f32; SIGNAL_OUTPUTS]) -> Option<u8> {
    let probs = softmax(signal_outputs);
    let (max_idx, &max_prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    let threshold = 1.0 / NUM_SYMBOLS as f32;
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
        assert!(maybe_emit(&outputs).is_none());
    }

    #[test]
    fn emit_concentrated_returns_symbol() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[3] = 5.0; // symbol 3 dominant
        assert_eq!(maybe_emit(&outputs), Some(3));
    }

    #[test]
    fn emit_slight_difference_returns_some() {
        let mut outputs = [0.0_f32; SIGNAL_OUTPUTS];
        outputs[1] = 0.01; // barely above others
        assert_eq!(maybe_emit(&outputs), Some(1));
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
                    let fast = receive_detailed_grid(
                        &signals,
                        &grid,
                        rx,
                        ry,
                        grid_size as f32,
                        signal_range,
                    );
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
