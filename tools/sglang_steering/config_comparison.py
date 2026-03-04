#!/usr/bin/env python3
"""Tabla comparativa de configs GLM-5-FP8"""

total_vram = 143.77
weight_main = 89.85
cg_main = 2.02
eagle_total = 2.58 + 0.35 + 0.71 + 1.63
bytes_per_token = 27.16 * 1024**3 / 291136
page_size = 64

def calc(mem_frac, eagle, ctx):
    usable = total_vram * mem_frac
    rem = usable - weight_main - cg_main
    if eagle:
        rem -= eagle_total
    rem -= 48 * ctx * 4 / 1024**3
    rem -= ctx * 256 * 2 * 4 / 1024**3
    kv_tok = int(rem * 1024**3 / bytes_per_token)
    kv_tok = (kv_tok // page_size) * page_size
    return kv_tok

tok_actual = calc(0.85, True, 1000000)
tok_a = calc(0.92, False, 1000000)
tok_b = calc(0.90, True, 202752)
tok_c = calc(0.90, False, 1000000)

W = 95
print()
print("=" * W)
h = ["", "ACTUAL", "A) Max conc", "B) Min lat", "C) Equilib"]
print(f"{h[0]:30s} | {h[1]:>12s} | {h[2]:>12s} | {h[3]:>12s} | {h[4]:>12s}")
print("=" * W)

# Config
rows = [
    ("EAGLE", "ON", "OFF", "ON", "OFF"),
    ("mem_fraction_static", "0.85", "0.92", "0.90", "0.90"),
    ("context_length", "1,000,000", "1,000,000", "202,752", "1,000,000"),
    ("YaRN RoPE", "si", "si", "no (nativo)", "si"),
]
for label, *vals in rows:
    print(f"{label:30s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s}")

print("-" * W)
# KV cache
toks = [tok_actual, tok_a, tok_b, tok_c]
line = f"{'KV cache (tokens)':30s}"
for i, t in enumerate(toks):
    if i == 0:
        line += f" | {t:>12,}"
    else:
        pct = (t / toks[0] - 1) * 100
        line += f" | {t:>7,} {pct:>+4.0f}%"
print(line)

print("-" * W)
print(f"{'--- LATENCIA ---':30s} |{'':>13s}|{'':>13s}|{'':>13s}|")

# Per-request latency at different loads
load_data = [(1, 93.4), (5, 74.2), (20, 49.8), (48, 35.4)]
for running, eagle_tps in load_data:
    no_eagle_tps = eagle_tps / 2.1
    vals = [eagle_tps, no_eagle_tps, eagle_tps, no_eagle_tps]
    label = f"  tok/s/req @ {running} running"
    line = f"{label:30s}"
    for i, v in enumerate(vals):
        if i == 0:
            line += f" | {v:>12.1f}"
        else:
            pct = (v / vals[0] - 1) * 100
            line += f" | {v:>7.1f} {pct:>+4.0f}%"
    print(line)

print()
# Time to generate 4K tokens
for running, eagle_tps in [(1, 93.4), (48, 35.4)]:
    no_eagle_tps = eagle_tps / 2.1
    vals = [4000/eagle_tps, 4000/no_eagle_tps, 4000/eagle_tps, 4000/no_eagle_tps]
    label = f"  seg para 4K tok @ {running} req"
    line = f"{label:30s}"
    for i, v in enumerate(vals):
        if i == 0:
            line += f" | {v:>11.1f}s"
        else:
            pct = (v / vals[0] - 1) * 100
            line += f" | {v:>6.1f}s {pct:>+4.0f}%"
    print(line)

print("-" * W)
print(f"{'--- CONCURRENCIA ---':30s} |{'':>13s}|{'':>13s}|{'':>13s}|")

print(f"{'  Max running requests':30s} | {'48 (cap)':>12s} | {'sin cap':>12s} | {'48 (cap)':>12s} | {'sin cap':>12s}")

for avg_tok in [2000, 4000, 8000]:
    all_toks = [tok_actual, tok_a, tok_b, tok_c]
    eagles = [True, False, True, False]
    vals = []
    for t, e in zip(all_toks, eagles):
        c = t // avg_tok
        if e:
            c = min(c, 48)
        vals.append(c)
    label = f"  req simult @ {avg_tok} tok/req"
    line = f"{label:30s}"
    for i, v in enumerate(vals):
        if i == 0:
            line += f" | {v:>12d}"
        else:
            pct = (v / vals[0] - 1) * 100
            line += f" | {v:>8d} {pct:>+4.0f}%"
    print(line)

# Cola estimada
print()
print(f"{'  Cola max (real/estimada)':30s} | {'432':>12s} | {'~0':>12s} | {'~300':>12s} | {'~0':>12s}")

print("-" * W)
print(f"{'--- CONTEXTO ---':30s} |{'':>13s}|{'':>13s}|{'':>13s}|")
ctx_rows = [
    ("  Max context (tokens)", "1,000,000", "1,000,000", "202,752", "1,000,000"),
    ("  Calidad >200K tok", "degradada*", "degradada*", "N/A", "degradada*"),
    ("  Calidad <200K tok", "nativa", "nativa", "nativa", "nativa"),
]
for label, *vals in ctx_rows:
    print(f"{label:30s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s}")

print("=" * W)
print()
print("* YaRN extrapola posiciones, pero el modelo fue entrenado hasta 202K.")
print("  Para <200K tokens el contexto funciona identico en todas las configs.")
print()
print("NOTA: Latencia sin EAGLE estimada como x/2.1 (EAGLE da 2.73 tok/step")
print("      pero el draft model anade ~30% overhead por step).")
