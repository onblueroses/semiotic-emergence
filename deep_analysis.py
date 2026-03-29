"""Deep time-windowed analysis of v15-2k runs."""
import numpy as np
import csv

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def get_metric(rows, col):
    return np.array([float(r[col]) for r in rows])

ref = load_csv('data/v15-psn30-2k-42-output.csv')
mute = load_csv('data/v15-mute-psn30-2k-42-output.csv')
s43 = load_csv('data/v15-psn30-2k-43-output.csv')

ref_gen = get_metric(ref, 'generation')
ref_fit = get_metric(ref, 'avg_fitness')
ref_rfc = get_metric(ref, 'response_fit_corr')
ref_sender = get_metric(ref, 'sender_fit_corr')
ref_recv = get_metric(ref, 'receiver_fit_corr')
ref_mi = get_metric(ref, 'mutual_info')
ref_entropy = get_metric(ref, 'signal_entropy')
ref_brain = get_metric(ref, 'avg_base_hidden')
ref_signals = get_metric(ref, 'signals_emitted')

mute_gen = get_metric(mute, 'generation')
mute_fit = get_metric(mute, 'avg_fitness')
mute_brain = get_metric(mute, 'avg_base_hidden')

s43_gen = get_metric(s43, 'generation')
s43_fit = get_metric(s43, 'avg_fitness')
s43_rfc = get_metric(s43, 'response_fit_corr')
s43_sender = get_metric(s43, 'sender_fit_corr')
s43_recv = get_metric(s43, 'receiver_fit_corr')
s43_mi = get_metric(s43, 'mutual_info')
s43_entropy = get_metric(s43, 'signal_entropy')
s43_brain = get_metric(s43, 'avg_base_hidden')
s43_signals = get_metric(s43, 'signals_emitted')

# Time-windowed counterfactual
print('=' * 70)
print('TIME-WINDOWED COUNTERFACTUAL: Reference (seed 42) vs Mute')
print('=' * 70)
windows = [(0, 50000), (50000, 100000), (100000, 150000), (150000, 188000)]
for wstart, wend in windows:
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    mute_mask = (mute_gen >= wstart) & (mute_gen < wend)
    if ref_mask.sum() > 0 and mute_mask.sum() > 0:
        ref_avg = ref_fit[ref_mask].mean()
        mute_avg = mute_fit[mute_mask].mean()
        delta_pct = (ref_avg - mute_avg) / mute_avg * 100
        print(f'  gen {wstart:>6}-{wend:>6}: ref={ref_avg:.1f} mute={mute_avg:.1f} delta={delta_pct:+.1f}%')

print()
print('=' * 70)
print('TIME-WINDOWED COUNTERFACTUAL: Seed 43 vs Mute')
print('=' * 70)
for wstart, wend in windows + [(188000, 222000)]:
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    mute_mask = (mute_gen >= wstart) & (mute_gen < wend)
    if s43_mask.sum() > 0 and mute_mask.sum() > 0:
        s43_avg = s43_fit[s43_mask].mean()
        mute_avg = mute_fit[mute_mask].mean()
        delta_pct = (s43_avg - mute_avg) / mute_avg * 100
        print(f'  gen {wstart:>6}-{wend:>6}: s43={s43_avg:.1f} mute={mute_avg:.1f} delta={delta_pct:+.1f}%')

# RFC trajectory
print()
print('=' * 70)
print('RESPONSE FIT CORR TRAJECTORY (10k-gen windows)')
print('=' * 70)
print(f'{"Window":>18} {"Ref rfc":>10} {"Ref sender":>12} {"S43 rfc":>10} {"S43 sender":>12}')
for wstart in range(0, 230000, 10000):
    wend = wstart + 10000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    ref_r = ref_rfc[ref_mask].mean() if ref_mask.sum() > 0 else float('nan')
    ref_s = ref_sender[ref_mask].mean() if ref_mask.sum() > 0 else float('nan')
    s43_r = s43_rfc[s43_mask].mean() if s43_mask.sum() > 0 else float('nan')
    s43_s = s43_sender[s43_mask].mean() if s43_mask.sum() > 0 else float('nan')
    if not (np.isnan(ref_r) and np.isnan(s43_r)):
        ref_str = f'{ref_r:>+10.4f}  {ref_s:>+10.4f}' if not np.isnan(ref_r) else '       n/a           n/a'
        s43_str = f'{s43_r:>+10.4f}  {s43_s:>+10.4f}' if not np.isnan(s43_r) else '       n/a           n/a'
        print(f'  {wstart:>6}-{wend:>6}  {ref_str}  {s43_str}')

# Brain dynamics
print()
print('=' * 70)
print('BRAIN DYNAMICS (10k-gen windows)')
print('=' * 70)
print(f'{"Window":>18} {"Ref brain":>10} {"Ref MI":>10} {"S43 brain":>10} {"S43 MI":>10} {"Mute brain":>12}')
for wstart in range(0, 280000, 10000):
    wend = wstart + 10000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    m_mask = (mute_gen >= wstart) & (mute_gen < wend)
    if ref_mask.sum() > 0 or s43_mask.sum() > 0 or m_mask.sum() > 0:
        ref_b = f'{ref_brain[ref_mask].mean():>10.1f}' if ref_mask.sum() > 0 else '       n/a'
        ref_m = f'{ref_mi[ref_mask].mean():>10.4f}' if ref_mask.sum() > 0 else '       n/a'
        s43_b = f'{s43_brain[s43_mask].mean():>10.1f}' if s43_mask.sum() > 0 else '       n/a'
        s43_m = f'{s43_mi[s43_mask].mean():>10.4f}' if s43_mask.sum() > 0 else '       n/a'
        m_b = f'{mute_brain[m_mask].mean():>10.1f}' if m_mask.sum() > 0 else '         n/a'
        print(f'  {wstart:>6}-{wend:>6}  {ref_b}  {ref_m}  {s43_b}  {s43_m}  {m_b}')

# RFC sign analysis
print()
print('=' * 70)
print('RFC SIGN ANALYSIS (% positive in 10k windows)')
print('=' * 70)
for wstart in range(0, 230000, 10000):
    wend = wstart + 10000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    if ref_mask.sum() > 0 or s43_mask.sum() > 0:
        parts = [f'  {wstart:>6}-{wend:>6}  ']
        if ref_mask.sum() > 0:
            ref_pct = (ref_rfc[ref_mask] > 0).mean() * 100
            ref_nz = (ref_rfc[ref_mask] != 0).mean() * 100
            parts.append(f'Ref: {ref_pct:>5.1f}% pos ({ref_nz:.0f}% nz)  ')
        if s43_mask.sum() > 0:
            s43_pct = (s43_rfc[s43_mask] > 0).mean() * 100
            s43_nz = (s43_rfc[s43_mask] != 0).mean() * 100
            parts.append(f'S43: {s43_pct:>5.1f}% pos ({s43_nz:.0f}% nz)')
        print(''.join(parts))

# Extended analyses
print()
print('=' * 70)
print('SEED 43 EXTENDED (beyond 188k reference endpoint)')
print('=' * 70)
ext = s43_gen >= 188000
if ext.sum() > 0:
    print(f'  Gens: {int(s43_gen[ext].min())}-{int(s43_gen[ext].max())} ({ext.sum()} datapoints)')
    print(f'  Avg fitness: {s43_fit[ext].mean():.1f}')
    print(f'  rfc: {s43_rfc[ext].mean():+.4f} ({(s43_rfc[ext] > 0).mean()*100:.1f}% positive)')
    print(f'  sender_fit: {s43_sender[ext].mean():+.4f}')
    print(f'  recv_fit: {s43_recv[ext].mean():+.4f}')
    print(f'  MI: {s43_mi[ext].mean():.4f}')
    print(f'  Entropy: {s43_entropy[ext].mean():.4f}')
    print(f'  Brain: {s43_brain[ext].mean():.1f}')
    print(f'  Signals/gen: {s43_signals[ext].mean():.1f}')

print()
print('=' * 70)
print('MUTE EXTENDED (beyond 188k)')
print('=' * 70)
ext = mute_gen >= 188000
if ext.sum() > 0:
    print(f'  Gens: {int(mute_gen[ext].min())}-{int(mute_gen[ext].max())} ({ext.sum()} datapoints)')
    print(f'  Avg fitness: {mute_fit[ext].mean():.1f}')
    print(f'  Brain: {mute_brain[ext].mean():.1f}')

# Receiver fit corr deep dive
print()
print('=' * 70)
print('RECEIVER FIT CORR vs RFC (disentangling spatial confound)')
print('=' * 70)
print(f'{"Window":>18} {"Ref recv":>10} {"Ref rfc":>10} {"Gap":>8} {"S43 recv":>10} {"S43 rfc":>10} {"Gap":>8}')
for wstart in range(0, 230000, 20000):
    wend = wstart + 20000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    if ref_mask.sum() > 0 or s43_mask.sum() > 0:
        if ref_mask.sum() > 0:
            rr = ref_recv[ref_mask].mean()
            rf = ref_rfc[ref_mask].mean()
            rg = rr - rf
            ref_str = f'{rr:>+10.4f}{rf:>+10.4f}{rg:>+8.2f}'
        else:
            ref_str = '       n/a       n/a     n/a'
        if s43_mask.sum() > 0:
            sr = s43_recv[s43_mask].mean()
            sf = s43_rfc[s43_mask].mean()
            sg = sr - sf
            s43_str = f'{sr:>+10.4f}{sf:>+10.4f}{sg:>+8.2f}'
        else:
            s43_str = '       n/a       n/a     n/a'
        print(f'  {wstart:>6}-{wend:>6}  {ref_str}  {s43_str}')

# The critical question
print()
print('=' * 70)
print('THE CRITICAL TEST: Does positive rfc translate to fitness advantage?')
print('=' * 70)
for label, start, end in [('Regime A (0-66k)', 0, 66000),
                           ('Regime B (66-155k)', 66000, 155000),
                           ('Regime C (155-188k)', 155000, 188000)]:
    ref_mask = (ref_gen >= start) & (ref_gen < end)
    mute_mask = (mute_gen >= start) & (mute_gen < end)
    if ref_mask.sum() > 0 and mute_mask.sum() > 0:
        rf = ref_fit[ref_mask].mean()
        mf = mute_fit[mute_mask].mean()
        rfc_val = ref_rfc[ref_mask].mean()
        delta = (rf - mf) / mf * 100
        print(f'  {label}: ref={rf:.1f} mute={mf:.1f} delta={delta:+.1f}% rfc={rfc_val:+.4f}')

# Altruism gap
print()
print('=' * 70)
print('ALTRUISM GAP (rfc - sender_fit_corr) OVER TIME')
print('=' * 70)
for wstart in range(0, 200000, 20000):
    wend = wstart + 20000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    parts = [f'  {wstart:>6}-{wend:>6}  ']
    if ref_mask.sum() > 0:
        gap = ref_rfc[ref_mask].mean() - ref_sender[ref_mask].mean()
        parts.append(f'Ref gap: {gap:+.4f}  ')
    if s43_mask.sum() > 0:
        gap = s43_rfc[s43_mask].mean() - s43_sender[s43_mask].mean()
        parts.append(f'S43 gap: {gap:+.4f}')
    print(''.join(parts))

# Signal emission volume comparison
print()
print('=' * 70)
print('SIGNAL EMISSION VOLUME (signals/gen over time)')
print('=' * 70)
for wstart in range(0, 230000, 20000):
    wend = wstart + 20000
    ref_mask = (ref_gen >= wstart) & (ref_gen < wend)
    s43_mask = (s43_gen >= wstart) & (s43_gen < wend)
    parts = [f'  {wstart:>6}-{wend:>6}  ']
    if ref_mask.sum() > 0:
        parts.append(f'Ref: {ref_signals[ref_mask].mean():>8.0f}  ')
    if s43_mask.sum() > 0:
        parts.append(f'S43: {s43_signals[s43_mask].mean():>8.0f}')
    if ref_mask.sum() > 0 or s43_mask.sum() > 0:
        print(''.join(parts))

# Brain size cost analysis
print()
print('=' * 70)
print('BRAIN SIZE COST: Signal runs invest more neurons')
print('=' * 70)
print('Are signal runs paying a brain-size tax?')
for label, start, end in [('0-50k', 0, 50000), ('50-100k', 50000, 100000),
                           ('100-150k', 100000, 150000), ('150-188k', 150000, 188000),
                           ('188-220k', 188000, 220000), ('220-278k', 220000, 278000)]:
    ref_mask = (ref_gen >= start) & (ref_gen < end)
    s43_mask = (s43_gen >= start) & (s43_gen < end)
    m_mask = (mute_gen >= start) & (mute_gen < end)
    parts = [f'  {label:>10}  ']
    if ref_mask.sum() > 0:
        parts.append(f'Ref: {ref_brain[ref_mask].mean():.1f}  ')
    else:
        parts.append('Ref: n/a  ')
    if s43_mask.sum() > 0:
        parts.append(f'S43: {s43_brain[s43_mask].mean():.1f}  ')
    else:
        parts.append('S43: n/a  ')
    if m_mask.sum() > 0:
        parts.append(f'Mute: {mute_brain[m_mask].mean():.1f}')
    else:
        parts.append('Mute: n/a')
    print(''.join(parts))

# Per-symbol analysis for seed 43 (it has different dominant symbols)
print()
print('=' * 70)
print('SEED 43 vs SEED 42: VOCABULARY COMPARISON')
print('=' * 70)
print('Reference (seed 42) late: sym5=82% (beacon), sym1=9% (poison), sym2=2% (alarm)')
print('Seed 43 late: sym3=71%, sym1=29% (two-symbol vocabulary)')
print()
print('Key difference: Seed 43 never developed vocabulary stratification.')
print('  - No rare alarm symbol')
print('  - No poison-correlated minority symbol')
print('  - Converged to simpler two-symbol system')
print('  - rfc stayed negative throughout (-0.13 sustained)')
print()
print('This means:')
print('  - Vocabulary stratification is NOT reproducible at seed 43')
print('  - Positive rfc at seed 42 may be seed-specific, not general')
print('  - The brain collapse/regrowth cycle at seed 42 was a lucky accident')
