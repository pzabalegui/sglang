#!/usr/bin/env python3
"""
GLM-5-FP8 Serving Configuration Trade-off Report
Machine: 95.133.252.73 (8x NVIDIA H200, 143GB/GPU)
Model: zai-org/GLM-5-FP8 (744B params, 40B active, MoE)
"""

total_vram = 143.77
weight_main = 89.85
cg_main = 2.02
eagle_total = 2.58 + 0.35 + 0.71 + 1.63
bytes_per_token = 27.16 * 1024**3 / 291136
page_size = 64

def calc_kv(mem_frac, eagle, ctx):
    usable = total_vram * mem_frac
    rem = usable - weight_main - cg_main
    if eagle:
        rem -= eagle_total
    rem -= 48 * ctx * 4 / 1024**3
    rem -= ctx * 256 * 2 * 4 / 1024**3
    kv_tok = int(rem * 1024**3 / bytes_per_token)
    return (kv_tok // page_size) * page_size

# Configs
tok_cur = calc_kv(0.85, True, 1000000)
tok_a = calc_kv(0.92, False, 1000000)
tok_b = calc_kv(0.90, True, 202752)
tok_c = calc_kv(0.90, False, 1000000)

# Helpers
def bar(val, max_val, width=15):
    filled = int(val / max_val * width)
    return "█" * filled + "░" * (width - filled)

def delta(new, old):
    pct = (new / old - 1) * 100
    if pct > 0:
        return f"+{pct:.0f}%"
    elif pct < 0:
        return f"{pct:.0f}%"
    return "="

W = 105

print()
print("=" * W)
print("  GLM-5-FP8 SERVING CONFIGURATION TRADE-OFF ANALYSIS")
print("  Hardware: 8x NVIDIA H200 (143 GB/GPU) | Model: 744B params (40B active, MoE)")
print("  Data source: Production logs (4,295 requests, 14h uptime)")
print("=" * W)

# ── CONFIG SECTION ──
print()
print("  CONFIGURATION" + " " * 18 + "│  CURRENT (baseline)  │  A) MAX THROUGHPUT   │  B) MIN LATENCY      │  C) BALANCED")
print("  " + "─" * 33 + "┼" + "──────────────────────┼──────────────────────┼──────────────────────┼" + "─" * 22)
rows = [
    ("Speculative Decoding (EAGLE)", "ON", "OFF", "ON", "OFF"),
    ("VRAM Allocation (mem_frac)", "0.85", "0.92", "0.90", "0.90"),
    ("Context Length", "1,000,000", "1,000,000", "202,752", "1,000,000"),
    ("RoPE Scaling (YaRN)", "Yes", "Yes", "No (native)", "Yes"),
]
for label, *vals in rows:
    print(f"  {label:<33s}│  {vals[0]:<20s}│  {vals[1]:<20s}│  {vals[2]:<20s}│  {vals[3]:<20s}")

# ── CONCURRENCY SECTION ──
print()
print("  " + "━" * (W - 2))
max_conc_vals = [48, 205, 48, 189]
max_conc = max(max_conc_vals)
print(f"  {'CONCURRENCY':33s}│{'':22s}│{'':22s}│{'':22s}│")
print("  " + "─" * 33 + "┼" + "──────────────────────┼──────────────────────┼──────────────────────┼" + "─" * 22)

print(f"  {'Max Running Requests':33s}│  {'48 (hard cap)':20s}│  {'unlimited':20s}│  {'48 (hard cap)':20s}│  {'unlimited':20s}")

for avg_tok in [2000, 4000, 8000]:
    all_toks = [tok_cur, tok_a, tok_b, tok_c]
    eagles = [True, False, True, False]
    vals = []
    for t, e in zip(all_toks, eagles):
        c = t // avg_tok
        if e:
            c = min(c, 48)
        vals.append(c)
    label = f"Concurrent Reqs @ {avg_tok//1000}K tok/req"
    line = f"  {label:33s}"
    for i, v in enumerate(vals):
        d = delta(v, vals[0])
        b = bar(v, max(vals))
        if i == 0:
            line += f"│  {v:>4d} {b:15s}  │"
        else:
            line += f"  {v:>4d} {d:>5s} {b:11s}  │"
    print(line)

# Queue
print(f"  {'Peak Queue (observed/est.)':33s}│  {'432':>4s} {'█' * 15:15s}  │  {'~0':>4s} {'░' * 15:15s}  │  {'~300':>4s} {'█' * 10 + '░' * 5:15s}  │  {'~0':>4s} {'░' * 15:15s}  ")

kv_vals = [tok_cur, tok_a, tok_b, tok_c]
max_kv = max(kv_vals)
label = "KV Cache Capacity (tokens)"
line = f"  {label:33s}"
for i, v in enumerate(kv_vals):
    d = delta(v, kv_vals[0])
    b = bar(v, max_kv)
    if i == 0:
        line += f"│  {v//1000:>4d}K {b:14s}  │"
    else:
        line += f"  {v//1000:>4d}K {d:>5s} {b:9s}  │"
print(line)

# ── LATENCY SECTION ──
print()
print("  " + "━" * (W - 2))
print(f"  {'LATENCY (per request)':33s}│{'':22s}│{'':22s}│{'':22s}│")
print("  " + "─" * 33 + "┼" + "──────────────────────┼──────────────────────┼──────────────────────┼" + "─" * 22)

load_data = [(1, 93.4), (10, 62.5), (48, 35.4)]
for running, eagle_tps in load_data:
    no_eagle_tps = eagle_tps / 2.1
    vals = [eagle_tps, no_eagle_tps, eagle_tps, no_eagle_tps]
    max_v = max(vals)
    label = f"Decode Speed @ {running} req (tok/s)"
    line = f"  {label:33s}"
    for i, v in enumerate(vals):
        d = delta(v, vals[0])
        b = bar(v, max_v)
        if i == 0:
            line += f"│  {v:>5.1f} {b:14s}  │"
        else:
            line += f"  {v:>5.1f} {d:>5s} {b:9s}  │"
    print(line)

# Time to generate
print(f"  {'':33s}│{'':22s}│{'':22s}│{'':22s}│")
for tok_gen in [4000]:
    for running, eagle_tps in [(1, 93.4), (48, 35.4)]:
        no_eagle_tps = eagle_tps / 2.1
        vals = [tok_gen/eagle_tps, tok_gen/no_eagle_tps, tok_gen/eagle_tps, tok_gen/no_eagle_tps]
        min_v = min(vals)  # lower is better for time
        label = f"Time for {tok_gen//1000}K tokens @ {running} req"
        line = f"  {label:33s}"
        for i, v in enumerate(vals):
            d = delta(v, vals[0])
            b = bar(v, max(vals))
            if i == 0:
                line += f"│  {v:>5.1f}s {b:13s}  │"
            else:
                line += f"  {v:>5.1f}s{d:>5s} {b:8s}  │"
        print(line)

# ── CONTEXT SECTION ──
print()
print("  " + "━" * (W - 2))
print(f"  {'CONTEXT WINDOW':33s}│{'':22s}│{'':22s}│{'':22s}│")
print("  " + "─" * 33 + "┼" + "──────────────────────┼──────────────────────┼──────────────────────┼" + "─" * 22)
ctx_rows = [
    ("Max Position Embeddings", "1,000,000", "1,000,000", "202,752", "1,000,000"),
    ("Quality <200K tokens", "native", "native", "native", "native"),
    ("Quality >200K tokens", "extrapolated*", "extrapolated*", "n/a", "extrapolated*"),
    ("VRAM Cost vs 202K ctx", "~580 MB/GPU", "~580 MB/GPU", "baseline", "~580 MB/GPU"),
]
for label, *vals in ctx_rows:
    print(f"  {label:33s}│  {vals[0]:<20s}│  {vals[1]:<20s}│  {vals[2]:<20s}│  {vals[3]:<20s}")

# ── SUMMARY ──
print()
print("  " + "━" * (W - 2))
print(f"  {'SUMMARY':33s}│{'':22s}│{'':22s}│{'':22s}│")
print("  " + "─" * 33 + "┼" + "──────────────────────┼──────────────────────┼──────────────────────┼" + "─" * 22)
summary = [
    ("Best for", "---", "Many users, APIs", "Few users, chat", "General purpose"),
    ("Concurrency vs Current", "baseline", "+112% to +327%", "+0% to +40%", "+96% to +294%"),
    ("Latency vs Current", "baseline", "-52% (2.1x slower)", "= (same)", "-52% (2.1x slower)"),
    ("Context vs Current", "baseline", "= (1M)", "-80% (202K only)", "= (1M)"),
    ("Queue Elimination", "no (432 peak)", "YES", "no (~300 peak)", "YES"),
]
for label, *vals in summary:
    print(f"  {label:33s}│  {vals[0]:<20s}│  {vals[1]:<20s}│  {vals[2]:<20s}│  {vals[3]:<20s}")

print()
print("  " + "=" * (W - 2))
print("  * YaRN extrapolates RoPE beyond training range (202K). Quality is native for contexts <200K tokens.")
print("  * Latency estimates: EAGLE produces 2.73 tok/step avg (measured), draft model adds ~30% overhead/step.")
print("  * Concurrency data: from production logs showing 60.6% of time at >90% KV cache utilization.")
print("  " + "=" * (W - 2))
print()
