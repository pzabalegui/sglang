# Experiments Log - Refusal Direction Research

## Project Overview
Investigating refusal direction manipulation in Mixture of Experts (MoE) language models based on paper arXiv:2406.11717.

---

## Experiment 1: OLMoE-1B-7B-0924-Instruct

**Date:** 2025-01-29
**Server:** 31.22.104.70 (1x L40S 48GB)
**Model:** allenai/OLMoE-1B-7B-0924-Instruct
**Notebook:** refusal_direction_manual_MoE.ipynb

### 1.1 Baseline Configuration

| Parameter | Value |
|-----------|-------|
| LAYER | 8 (50% of 16 layers) |
| N_INST_TRAIN | 32 |
| USE_4BIT | True |
| Direction Method | Mean difference |

**Results:**
- Ablation Bypass: ~60%
- Orthogonalization: Not tested (requires USE_4BIT=False)

---

### 1.2 Optimized Ablation

| Parameter | Value |
|-----------|-------|
| LAYER | 12 (75% of 16 layers) |
| N_INST_TRAIN | 128 |
| USE_4BIT | True |
| Direction Method | Mean difference |

**Results:**
| Technique | Bypass Rate |
|-----------|-------------|
| Ablation | **100%** ✅ |

**Key Fix:** transformers 5.x compatibility - hooks now handle both tuple and tensor outputs:
```python
def make_ablation_hook(direction: Tensor):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # ... rest of hook
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        else:
            return modified
    return hook
```

---

### 1.3 Orthogonalization Tests

**Requirement:** USE_4BIT=False (full precision weights needed for modification)

#### 1.3.1 Simple Mean Direction + All Layers + lm_head

| Parameter | Value |
|-----------|-------|
| LAYER | 12 |
| N_INST_TRAIN | 128 |
| USE_4BIT | False |
| Direction | Single mean direction |
| Targets | All MLP layers + lm_head |

**Results:**
| Technique | Bypass Rate |
|-----------|-------------|
| Orthogonalization | **76%** ✅ |

---

#### 1.3.2 PCA Direction Extraction

| Parameter | Value |
|-----------|-------|
| Direction Method | PCA (first principal component) |
| Other params | Same as 1.3.1 |

**Results:**
| Technique | Bypass Rate |
|-----------|-------------|
| Orthogonalization (PCA) | 72% ❌ |

**Conclusion:** PCA extraction performs worse than simple mean difference.

---

#### 1.3.3 Per-Layer Orthogonalization

| Parameter | Value |
|-----------|-------|
| Direction | Different direction per layer |
| Targets | Selective layers (10-15) |

**Results:**
| Technique | Bypass Rate |
|-----------|-------------|
| Per-layer ortho | **BROKEN** ❌ |

**Failure Mode:** Model output became "|||IP_ADDRESS|||" repeated infinitely. Per-layer orthogonalization was too aggressive and destabilized the model.

---

#### 1.3.4 Selective Layer Orthogonalization (Simple Direction)

| Parameter | Value |
|-----------|-------|
| Direction | Single mean direction |
| Targets | Only layers 10-15 + lm_head |

**Results:**
| Technique | Bypass Rate |
|-----------|-------------|
| Selective ortho | 72% ❌ |

**Conclusion:** No improvement over full orthogonalization.

---

### 1.4 Induction Tests

| Parameter | Value |
|-----------|-------|
| Scale | 2.0 |
| Method | Activation addition |

**Results:**
| Technique | Effect |
|-----------|--------|
| Induction (scale=2.0) | Unstable ❌ |
| Induction (scale=5.0) | "NEVER NEVER NEVER" output |

**Conclusion:** Induction not viable for this model architecture.

---

## Summary - OLMoE-1B-7B

| Technique | Best Result | Status |
|-----------|-------------|--------|
| Ablation | 100% | ✅ Working |
| Orthogonalization | 76% | ✅ Working |
| Induction | N/A | ❌ Not viable |

**Best Configuration:**
- LAYER = 12 (75% depth)
- N_INST_TRAIN = 128
- Direction = Mean difference (not PCA)
- Orthogonalization = All layers + lm_head (not selective)

---

## Technical Notes

### Dimension Detection for Orthogonalization
MoE models have varying weight matrix shapes. Auto-detection required:

```python
def orthogonalize_matrix(matrix, vec):
    v = vec.to(matrix.device, dtype=matrix.dtype)
    v = v / v.norm()

    if matrix.shape[-1] == v.shape[0]:
        # Shape (..., d_model) - output is last dim
        Wv = matrix @ v
        return matrix - Wv.unsqueeze(-1) * v
    elif matrix.shape[0] == v.shape[0]:
        # Shape (d_model, ...) - output is first dim
        vW = v @ matrix
        return matrix - v.unsqueeze(1) * vW.unsqueeze(0)
    else:
        return matrix
```

### Dependencies
- numpy==1.26.4 (avoid sklearn incompatibility)
- transformers>=5.0
- torch with CUDA support
- bitsandbytes (optional, for 4-bit quantization)

---

---

## Experiment 2: GLM-4.7-Flash (Pending)

**Date:** TBD
**Server:** TBD
**Model:** zai-org/GLM-4.7-Flash
**Config:** configs/glm47flash_ablation.yaml

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Type | MoE (30B-A3B) |
| Total Params | 31B |
| Active Params | ~3B per token |
| Layers | 47 |
| Hidden Size | 2048 |
| Routed Experts | 64 |
| Shared Experts | 1 |
| Experts per Token | 4 |
| Attention | LoRA-based (q_lora=768, kv_lora=512) |

### Planned Configuration

| Parameter | Value |
|-----------|-------|
| LAYER | 35 (75% of 47 layers) |
| N_INST_TRAIN | 128 |
| USE_4BIT | True (ablation) / False (ortho) |
| Direction Method | Mean difference |

### Hardware Requirements

| Technique | VRAM Required |
|-----------|---------------|
| Ablation (4-bit) | ~24GB (L40S OK) |
| Orthogonalization | ~64-80GB (A100 80GB) |

### Results

| Technique | Bypass Rate | Status |
|-----------|-------------|--------|
| Ablation | TBD | ⏳ Pending |
| Orthogonalization | TBD | ⏳ Pending |

### Notes
- Model uses LoRA attention (different from OLMoE)
- 64 experts per layer makes orthogonalization expensive
- Consider testing only shared_experts first for ortho

---

## Next Experiments

- [x] Test GLM-4.7-Flash (larger MoE model) → Config ready
- [ ] Run GLM-4.7-Flash ablation experiment
- [ ] Test different layer depths for orthogonalization
- [ ] Investigate hybrid approaches (partial ablation + orthogonalization)
- [ ] Add quantitative metrics (see plan file)
