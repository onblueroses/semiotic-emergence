#!/usr/bin/env python3
"""Generate figures for FINDINGS.md from run CSV data."""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(FIG_DIR), 'data')

def load_run(name):
    path = os.path.join(DATA_DIR, name, 'output.csv')
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8',
                         invalid_raise=False)
    return data

def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

def smooth_gen(gen, window=50):
    if len(gen) < window:
        return gen
    return gen[window-1:]


# --- Figure 1: Signal Value Across Eras ---
# Fitness comparison: signal vs mute for v7 (pop=1000) and Era 5 (accidental drain)

def fig1_signal_value():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Signal Value: Fitness Impact Across Experimental Conditions', fontweight='bold', fontsize=14)

    # Panel A: Era 5 - baseline-s100 vs mute-s100 (drain=0.10)
    ax = axes[0, 0]
    sig = load_run('baseline-s100')
    mute = load_run('mute-s100')
    w = 200
    ax.plot(smooth_gen(sig['generation'], w), smooth(sig['avg_fitness'], w),
            color='#2196F3', alpha=0.8, linewidth=1, label='Signal (s100)')
    ax.plot(smooth_gen(mute['generation'], w), smooth(mute['avg_fitness'], w),
            color='#F44336', alpha=0.8, linewidth=1, label='Mute (s100)')
    ax.set_title('A. Era 5: High drain (0.10), 384 prey')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Fitness')
    ax.legend(loc='lower right')
    ax.text(0.02, 0.95, 'Mute +20-25%', transform=ax.transAxes, fontsize=9,
            va='top', color='#F44336', fontweight='bold')

    # Panel B: v6 - signal vs mute (freeze zones, drain=0.02)
    ax = axes[0, 1]
    sig = load_run('v6-signal-s300')
    mute = load_run('v6-mute-s300')
    w = 200
    ax.plot(smooth_gen(sig['generation'], w), smooth(sig['avg_fitness'], w),
            color='#2196F3', alpha=0.8, linewidth=1, label='Signal (s300)')
    ax.plot(smooth_gen(mute['generation'], w), smooth(mute['avg_fitness'], w),
            color='#F44336', alpha=0.8, linewidth=1, label='Mute (s300)')
    ax.set_title('B. Era 6: Freeze+flee zones, 384 prey')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Fitness')
    ax.legend(loc='lower right')
    ax.text(0.02, 0.95, 'Mute +8%', transform=ax.transAxes, fontsize=9,
            va='top', color='#F44336', fontweight='bold')

    # Panel C: v7 - signal vs mute (pop=1000, demes, death echoes)
    ax = axes[1, 0]
    sig = load_run('v7-signals-42')
    mute = load_run('v7-mute-42')
    w = 200
    ax.plot(smooth_gen(sig['generation'], w), smooth(sig['avg_fitness'], w),
            color='#2196F3', alpha=0.8, linewidth=1, label='Signal (v7)')
    ax.plot(smooth_gen(mute['generation'], w), smooth(mute['avg_fitness'], w),
            color='#F44336', alpha=0.8, linewidth=1, label='Mute (v7)')
    ax.set_title('C. Era 7: pop=1000, demes, death echoes')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Fitness')
    ax.legend(loc='lower right')
    ax.text(0.02, 0.95, 'Mute +12.8%', transform=ax.transAxes, fontsize=9,
            va='top', color='#F44336', fontweight='bold')

    # Panel D: Summary bar chart of signal cost across eras
    ax = axes[1, 1]
    eras = ['Era 5\n(drain=0.10)', 'Era 6\n(freeze)', 'Era 7\n(pop=1k)', 'GPU\n(pop=5k)']
    signal_penalty = [-25.5, -8.0, -12.8, None]
    signal_benefit = [None, None, None, 51.0]  # GPU correlation
    colors = ['#F44336', '#F44336', '#F44336', '#4CAF50']
    values = [-25.5, -8.0, -12.8, 51.0]
    bars = ax.bar(eras, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Signal fitness impact (%)\nor correlation (GPU)')
    ax.set_title('D. Signal value by condition')
    for bar, val in zip(bars, values):
        ypos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, ypos + (2 if val > 0 else -3),
                f'{val:+.1f}{"%" if val < 0 else " r"}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_signals_adaptive_at_scale.png'))
    plt.close()
    print('Saved fig1_signals_adaptive_at_scale.png')


# --- Figure 2: Semiotic Metrics Evolution ---
# MI, iconicity, signal entropy, JSD across key runs

def fig2_semiotic_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Semiotic Metric Evolution Across Eras', fontweight='bold', fontsize=14)

    runs = {
        'v7-signals-42': ('#2196F3', 'v7 (pop=1k)'),
        'v11-cap6-42': ('#FF9800', 'v11-cap6'),
        'v11-cap32-42': ('#9C27B0', 'v11-cap32'),
        'v10-2k-42': ('#4CAF50', 'v10 (pop=2k)'),
    }

    # Panel A: Mutual Information
    ax = axes[0, 0]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['mutual_info'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('A. Mutual Information I(Signal; ZoneDistance)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('MI (nats)')
    ax.legend(loc='upper right')

    # Panel B: Iconicity
    ax = axes[0, 1]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['iconicity'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('B. Iconicity (zone signal rate - baseline)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Iconicity')
    ax.legend(loc='upper right')

    # Panel C: Signal Entropy
    ax = axes[1, 0]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['signal_entropy'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.axhline(y=np.log(6), color='gray', linestyle=':', alpha=0.5, label='max (ln6)')
    ax.set_title('C. Signal Entropy')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.legend(loc='lower right')

    # Panel D: Receiver JSD (in-zone)
    ax = axes[1, 1]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['jsd_pred'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('D. Receiver JSD (in-zone context)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('JSD')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_metric_evolution_across_eras.png'))
    plt.close()
    print('Saved fig2_metric_evolution_across_eras.png')


# --- Figure 3: Brain Evolution ---
# Hidden layer sizes and their relationship to signal metrics

def fig3_brain_evolution():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Neural Architecture Evolution', fontweight='bold', fontsize=14)

    runs = {
        'v7-signals-42': ('#2196F3', 'v7-signal'),
        'v7-mute-42': ('#F44336', 'v7-mute'),
        'v11-cap6-42': ('#FF9800', 'v11-cap6'),
        'v11-cap32-42': ('#9C27B0', 'v11-cap32'),
    }

    # Panel A: Base hidden size
    ax = axes[0, 0]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['avg_base_hidden'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('A. Base Hidden Layer Size')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg neurons')
    ax.legend(loc='best')

    # Panel B: Signal hidden size
    ax = axes[0, 1]
    for name, (color, label) in runs.items():
        d = load_run(name)
        w = min(100, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['avg_signal_hidden'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('B. Signal Hidden Layer Size')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg neurons')
    ax.legend(loc='best')

    # Panel C: v10 large population brain evolution
    ax = axes[1, 0]
    d = load_run('v10-2k-42')
    ax.plot(d['generation'], d['avg_base_hidden'],
            color='#4CAF50', alpha=0.8, linewidth=1, label='Base hidden')
    ax.plot(d['generation'], d['avg_signal_hidden'],
            color='#FF9800', alpha=0.8, linewidth=1, label='Signal hidden')
    ax.fill_between(d['generation'], d['min_base_hidden'], d['max_base_hidden'],
                    color='#4CAF50', alpha=0.1)
    ax.fill_between(d['generation'], d['min_signal_hidden'], d['max_signal_hidden'],
                    color='#FF9800', alpha=0.1)
    ax.set_title('C. v10 (pop=2000) Brain Size + Range')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Neurons')
    ax.legend(loc='best')

    # Panel D: Zone deaths comparison
    ax = axes[1, 1]
    for name, (color, label) in [
        ('v7-signals-42', ('#2196F3', 'v7-signal')),
        ('v7-mute-42', ('#F44336', 'v7-mute')),
    ]:
        d = load_run(name)
        w = min(200, max(5, len(d['generation']) // 20))
        ax.plot(smooth_gen(d['generation'], w), smooth(d['zone_deaths'], w),
                color=color, alpha=0.8, linewidth=1, label=label)
    ax.set_title('D. Zone Deaths per Generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Deaths')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_neural_architecture_growth.png'))
    plt.close()
    print('Saved fig3_neural_architecture_growth.png')


# --- Figure 4: Volume Knob Experiment (Era 9) ---
# v11-cap6 vs v11-cap32 detailed comparison

def fig4_volume_knob():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Era 9: Volume Knob Experiment (cap=6 vs cap=32)', fontweight='bold', fontsize=14)

    cap6 = load_run('v11-cap6-42')
    cap32 = load_run('v11-cap32-42')

    w6 = 100
    w32 = 100

    # Panel A: Fitness
    ax = axes[0, 0]
    ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['avg_fitness'], w6),
            color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['avg_fitness'], w32),
            color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.set_title('A. Avg Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()

    # Panel B: MI
    ax = axes[0, 1]
    ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['mutual_info'], w6),
            color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['mutual_info'], w32),
            color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.set_title('B. Mutual Information')
    ax.set_xlabel('Generation')
    ax.set_ylabel('MI (nats)')
    ax.legend()

    # Panel C: Signal entropy
    ax = axes[0, 2]
    ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['signal_entropy'], w6),
            color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['signal_entropy'], w32),
            color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.axhline(y=np.log(6), color='gray', linestyle=':', alpha=0.5)
    ax.set_title('C. Signal Entropy')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.legend()

    # Panel D: response_fit_corr
    ax = axes[1, 0]
    ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['response_fit_corr'], w6),
            color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['response_fit_corr'], w32),
            color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_title('D. Response-Fitness Correlation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Pearson r')
    ax.legend()

    # Panel E: Food MI
    ax = axes[1, 1]
    if 'food_mi' in cap6.dtype.names:
        ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['food_mi'], w6),
                color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    if 'food_mi' in cap32.dtype.names:
        ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['food_mi'], w32),
                color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.set_title('E. Food MI')
    ax.set_xlabel('Generation')
    ax.set_ylabel('MI (nats)')
    ax.legend()

    # Panel F: Signal hidden layer size
    ax = axes[1, 2]
    ax.plot(smooth_gen(cap6['generation'], w6), smooth(cap6['avg_signal_hidden'], w6),
            color='#FF9800', alpha=0.8, linewidth=1, label='cap=6')
    ax.plot(smooth_gen(cap32['generation'], w32), smooth(cap32['avg_signal_hidden'], w32),
            color='#9C27B0', alpha=0.8, linewidth=1, label='cap=32')
    ax.set_title('F. Signal Hidden Size')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg neurons')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_capacity_vs_encoding_quality.png'))
    plt.close()
    print('Saved fig4_capacity_vs_encoding_quality.png')


# --- Figure 5: Blind Mode Early Results ---
# v12-blind6-42 early trajectory

def fig5_blind_mode():
    d = load_run('v12-blind6-42')
    if len(d) < 10:
        print('Skipping fig5_blind_mode.png - insufficient data')
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Era 12: Blind Mode (spatial perception stripped, 100k gens)', fontweight='bold', fontsize=14)

    w = max(5, len(d['generation']) // 20)

    # Panel A: Fitness
    ax = axes[0, 0]
    ax.plot(smooth_gen(d['generation'], w), smooth(d['avg_fitness'], w),
            color='#607D8B', alpha=0.8, linewidth=1)
    ax.set_title('A. Avg Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

    # Panel B: MI + Iconicity
    ax = axes[0, 1]
    ax.plot(smooth_gen(d['generation'], w), smooth(d['mutual_info'], w),
            color='#2196F3', alpha=0.8, linewidth=1, label='MI')
    ax.plot(smooth_gen(d['generation'], w), smooth(d['iconicity'], w),
            color='#FF9800', alpha=0.8, linewidth=1, label='Iconicity')
    ax.set_title('B. MI + Iconicity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Value')
    ax.legend()

    # Panel C: Signal entropy + signals emitted
    ax = axes[1, 0]
    ax.plot(smooth_gen(d['generation'], w), smooth(d['signal_entropy'], w),
            color='#9C27B0', alpha=0.8, linewidth=1, label='Entropy')
    ax.axhline(y=np.log(6), color='gray', linestyle=':', alpha=0.5, label='max (ln6)')
    ax.set_title('C. Signal Entropy')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.legend()

    # Panel D: Brain size
    ax = axes[1, 1]
    ax.plot(smooth_gen(d['generation'], w), smooth(d['avg_base_hidden'], w),
            color='#4CAF50', alpha=0.8, linewidth=1, label='Base hidden')
    ax.plot(smooth_gen(d['generation'], w), smooth(d['avg_signal_hidden'], w),
            color='#FF9800', alpha=0.8, linewidth=1, label='Signal hidden')
    ax.set_title('D. Brain Size')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg neurons')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_blind_mode_no_emergence.png'))
    plt.close()
    print('Saved fig5_blind_mode_no_emergence.png')


# --- Figure 6: Blind vs Sighted Comparison ---
# v12-blind6-42 (blind, cap=6) vs v11-cap6-42 (sighted, cap=6) - same signal capacity, different perception

def fig6_blind_vs_sighted():
    blind = load_run('v12-blind6-42')
    sighted = load_run('v11-cap6-42')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Blind vs Sighted: Same Signal Capacity (cap=6), 100k Gens', fontweight='bold', fontsize=14)

    wb = max(5, len(blind['generation']) // 20)
    ws = max(5, len(sighted['generation']) // 20)

    # Panel A: Fitness
    ax = axes[0, 0]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['avg_fitness'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted (v11-cap6)')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['avg_fitness'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind (v12)')
    ax.set_title('A. Avg Fitness')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()

    # Panel B: MI
    ax = axes[0, 1]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['mutual_info'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['mutual_info'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind')
    ax.set_title('B. Mutual Information')
    ax.set_xlabel('Generation')
    ax.set_ylabel('MI (nats)')
    ax.legend()

    # Panel C: Signal entropy
    ax = axes[0, 2]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['signal_entropy'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['signal_entropy'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind')
    ax.axhline(y=np.log(6), color='gray', linestyle=':', alpha=0.5)
    ax.set_title('C. Signal Entropy')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Entropy (nats)')
    ax.legend()

    # Panel D: Iconicity
    ax = axes[1, 0]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['iconicity'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['iconicity'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind')
    ax.set_title('D. Iconicity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Iconicity')
    ax.legend()

    # Panel E: Signal hidden size
    ax = axes[1, 1]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['avg_signal_hidden'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['avg_signal_hidden'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind')
    ax.set_title('E. Signal Hidden Size')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg neurons')
    ax.legend()

    # Panel F: Zone deaths
    ax = axes[1, 2]
    ax.plot(smooth_gen(sighted['generation'], ws), smooth(sighted['zone_deaths'], ws),
            color='#2196F3', alpha=0.8, linewidth=1, label='Sighted')
    ax.plot(smooth_gen(blind['generation'], wb), smooth(blind['zone_deaths'], wb),
            color='#607D8B', alpha=0.8, linewidth=1, label='Blind')
    ax.set_title('F. Zone Deaths per Generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Deaths')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_perception_required_for_signals.png'))
    plt.close()
    print('Saved fig6_perception_required_for_signals.png')


if __name__ == '__main__':
    fig1_signal_value()
    fig2_semiotic_metrics()
    fig3_brain_evolution()
    fig4_volume_knob()
    fig5_blind_mode()
    fig6_blind_vs_sighted()
    print('All figures generated.')
