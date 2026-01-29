# Refusal Direction Research

Research on refusal direction manipulation in Large Language Models, based on [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).

## Project Structure

```
.
├── configs/              # Experiment configurations (YAML)
│   ├── olmoe_ablation.yaml
│   └── olmoe_orthogonalization.yaml
├── notebooks/            # Jupyter notebooks for experiments
│   ├── refusal_direction_manual_MoE.ipynb
│   └── refusal_direction_manual.ipynb
├── results/              # Experiment results and logs
│   ├── olmoe/
│   └── glm/
├── src/                  # Reusable Python modules
│   └── metrics.py        # Refusal detection and metrics
├── docs/                 # Documentation and papers
└── experiments.md        # Main experiment log
```

## Techniques

### 1. Ablation (Dynamic Intervention)
- Uses PyTorch hooks to remove refusal direction at inference time
- **Best result: 100% bypass rate** on OLMoE

### 2. Orthogonalization (Permanent Modification)
- Modifies weight matrices to remove refusal direction
- Requires full precision weights (no quantization)
- **Best result: 76% bypass rate** on OLMoE

### 3. Induction (Activation Addition)
- Adds refusal direction to increase refusals
- Currently unstable on MoE architectures

## Quick Start

1. **Configure experiment**: Edit YAML in `configs/`
2. **Run on remote GPU**: Upload notebook to server
3. **Record results**: Update `results/experiments.md`

## Remote Server Setup

```bash
# SSH to GPU server
ssh root@<IP>

# Start JupyterLab with Docker
docker run -d --gpus all -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1
```

## Dependencies

```
numpy==1.26.4
transformers>=5.0
torch
bitsandbytes  # optional, for 4-bit quantization
```

## Models Tested

| Model | Ablation | Ortho | Notes |
|-------|----------|-------|-------|
| OLMoE-1B-7B-0924-Instruct | 100% | 76% | MoE architecture |
| GLM-4.7-Flash | TBD | TBD | Pending |

## References

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
