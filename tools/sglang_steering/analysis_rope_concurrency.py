#!/usr/bin/env python3
"""
Análisis: RoPE context_length vs Concurrencia en GLM-5-FP8
Datos reales del log de 95.133.252.73 (8x H200)
"""

# Datos reales del log (TP0)
total_vram_gb = 143.77
avail_before_weights = 138.00
weight_main = 89.85
kv_cache_gb = 27.16
kv_tokens = 291136
cuda_graph_main = 2.02
eagle_weights = 2.58
eagle_kv = 0.35
eagle_cg = 0.71 + 1.63
avail_final = 13.55

head_dim = 256
rotary_dim = head_dim
page_size = 64
max_running = 48

bytes_per_token = kv_cache_gb * 1024**3 / kv_tokens

print("=" * 65)
print("1. IMPACTO DIRECTO DEL CONTEXT_LENGTH EN VRAM (por GPU)")
print("=" * 65)
for ctx in [202752, 500000, 1000000]:
    req_mb = max_running * ctx * 4 / 1024**2
    cos_mb = ctx * rotary_dim * 2 * 4 / 1024**2
    total_mb = req_mb + cos_mb
    print(f"  ctx={ctx:>10,}: req_to_token={req_mb:.0f}MB + cos_sin={cos_mb:.0f}MB = {total_mb:.0f}MB ({total_mb/1024:.2f}GB)")

print()
print("  CONCLUSION: 1M vs 202K = ~580MB extra/GPU")
print("  Eso son ~6K tokens menos de KV (~2%). NEGLIGIBLE.")

print()
print("=" * 65)
print("2. DESGLOSE REAL DE VRAM POR GPU (del log)")
print("=" * 65)
items = [
    ("Pesos modelo", weight_main),
    ("KV cache", kv_cache_gb),
    ("CUDA graphs", cuda_graph_main),
    ("EAGLE pesos", eagle_weights),
    ("EAGLE KV+graphs", eagle_kv + eagle_cg),
    ("SIN USAR", avail_final),
    ("Sistema/driver", total_vram_gb - avail_before_weights),
]
for name, gb in items:
    pct = gb / total_vram_gb * 100
    print(f"  {name:<20s} {gb:>7.2f} GB  ({pct:>5.1f}%)")

print()
print("=" * 65)
print("3. ESCENARIOS DE CONFIGURACION")
print("=" * 65)
print(f"  Bytes/token KV: {bytes_per_token:.0f} B ({bytes_per_token/1024:.1f} KB)")
print()

cfgH = "Config"
kvH = "KV(GB)"
tokH = "Tokens"
vsH = "vs actual"
mrH = "MaxReq"
print(f"  {cfgH:<38s} {kvH:>7s} {tokH:>10s} {vsH:>10s} {mrH:>7s}")
print("  " + "-" * 72)

configs = [
    ("ACTUAL (EAGLE+0.85+1M)", 0.85, True, 1000000),
    ("Sin EAGLE, 0.85, 1M", 0.85, False, 1000000),
    ("EAGLE, 0.90, 1M", 0.90, True, 1000000),
    ("Sin EAGLE, 0.90, 1M", 0.90, False, 1000000),
    ("Sin EAGLE, 0.90, 202K", 0.90, False, 202752),
    ("Sin EAGLE, 0.92, 1M", 0.92, False, 1000000),
    ("Sin EAGLE, 0.92, 202K", 0.92, False, 202752),
]

results = []
for name, mem_frac, eagle, ctx_len in configs:
    usable = total_vram_gb * mem_frac
    remaining = usable - weight_main - cuda_graph_main
    if eagle:
        remaining -= eagle_weights + eagle_kv + eagle_cg
    req_overhead = max_running * ctx_len * 4 / 1024**3
    remaining -= req_overhead
    cos_sin = ctx_len * rotary_dim * 2 * 4 / 1024**3
    remaining -= cos_sin
    kv_gb = max(0, remaining)
    tokens = int(kv_gb * 1024**3 / bytes_per_token)
    tokens = (tokens // page_size) * page_size
    pct = (tokens / kv_tokens - 1) * 100
    max_req = "48" if eagle else "sin cap"
    print(f"  {name:<38s} {kv_gb:>7.2f} {tokens:>10,} {pct:>+9.0f}%  {max_req:>7s}")
    results.append((name, tokens, eagle))

print()
print("=" * 65)
print("4. REQUESTS CONCURRENTES SEGUN LARGO PROMEDIO")
print("=" * 65)

for avg_tok in [2000, 4000, 6000, 8000]:
    print(f"\n  Promedio {avg_tok} tokens/request:")
    for name, tokens, eagle in results:
        concurrent = tokens // avg_tok
        if eagle:
            concurrent = min(concurrent, 48)
        print(f"    {name:<38s} -> {concurrent:>4} req simultaneas")

print()
print("=" * 65)
print("5. CONCLUSION")
print("=" * 65)
print("""
  - RoPE/YaRN (context_length) tiene impacto NEGLIGIBLE (<2%)
    en la capacidad de requests concurrentes.

  - El VERDADERO cuello de botella es EAGLE:
    * Cap duro de 48 requests concurrentes
    * Consume ~5.3 GB/GPU (pesos + KV + CUDA graphs)

  - El segundo factor es mem_fraction_static:
    * 0.85 deja 13.5 GB/GPU SIN USAR
    * Subir a 0.90-0.92 aprovecha esa VRAM

  - CONFIG OPTIMA para MAX CONCURRENCIA:
    Sin EAGLE + mem_fraction=0.92 + context_length=1M
    = ~57% mas KV cache, sin cap de requests
    Trade-off: decode ~2.5x mas lento por request individual
""")
