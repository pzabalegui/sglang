# Archive

Archivos históricos, versiones superadas y resultados intermedios. Preservados como referencia.
Fecha de archivado: 20 Feb 2026.

---

## Contenido

### superseded_sglang_fixes/
Scripts de debug iterativos del desarrollo de SGLang steering. Superados por los archivos finales en `tools/sglang_steering/patched_files_remote/`.
- `fix_additive_v2.py` — cambio de projective a additive (CAA)
- `fix_bug3.py` — fix per-request override + pesos Gaussianos
- `fix_clamped_steering.py` — implementación clamped `max(0, proj)`
- `fix_fullrepr_postlayer.py` — fix mismatch espacio extracción vs aplicación
- `fix_pre_layer_steering.py` — steering pre-capa (antes de layer forward)
- `fix_steering_and_capture.py` — fix captura h+residual en vez de solo h

### superseded_sglang_patches/
Versiones anteriores de archivos SGLang parcheados, antes de la versión final `patched_files_remote/`:
- `glm4_moe_patched.py`, `forward_batch_info_patched.py`, `model_runner_patched.py`
- `server_args_patched.py`, `protocol_patched.py`, `serving_chat_patched.py`

### hooks_ablation/
Método de steering via PyTorch hooks en HuggingFace transformers. Superado por SGLang nativo (80% ASR vs 99.4%). Solo funciona con modelos BF16 cargables via transformers.
- `ablation_sweep.py` — parameter sweep exhaustivo (capas, escalas, modos)
- `sglang_steering_server.py` — servidor FastAPI con API OpenAI-compatible
- `quick_test.py` — test rápido de hooks
- `README.md`

### intermediate_results/
Benchmarks parciales y resultados intermedios del desarrollo:
- `benchmark_das_*.json` — benchmarks DAS de 17-18 Feb (pre decode steering)
- `benchmark_results_batch_1_failed/`, `benchmark_results_batch_2/` — batches hooks
- `benchmark_results_full/` — benchmarks pre-finales
- `experiments.md` — log de experimentos OLMoE
- `results_individual_testing/` — tests individuales tempranos

### superseded_docs/
Documentación superada por el informe técnico y el research log:
- `PLAN_GLM47_ABLITERATION.md` — plan original (completado)
- `optimal_parameters.md` — resultados tempranos OLMoE/Qwen
- `resultados_finales.md` — resultados Flash (Feb 3)
- `investigacion_abliteracion.md` — notas de investigación iniciales
- `NOTION_EXPORT_abliteracion_LLMs.md` — export de Notion
- `analisis_huihui_*.md` — ingeniería inversa del modelo huihui-ai
- `readme_20260217.md` — README del 17 Feb (pre decode steering)

### superseded_notebooks/
Notebooks experimentales superados por `refusal_direction_GLM47Flash.ipynb`:
- `refusal_direction_manual.ipynb` — implementación manual
- `refusal_direction_manual_MoE.ipynb` — versión MoE
- `refusal_vectors_explicado.ipynb` — versión explicada
- `Copy_of_refusal_demo-2.ipynb` — copia del demo original de Colab

### old_scripts/ (pre-existente)
Scripts de versiones anteriores del proyecto (v1, v2, v3 de abliterate_glm, etc.)

### backups/ (pre-existente)
Backups de sesiones en servidores remotos:
- `backup_b200/` — servidor B200
- `backup_remote_2026-02-03/` — 3 Feb 2026
- `remote_backup_20260209/` — 9 Feb 2026

### steering_server_github_post/ (pre-existente)
Implementación original de steering basada en un post de GitHub.

### reference/ (pre-existente)
- `heretic/` — herramienta automática de abliteración (referencia externa)
