# Reproducing Results

This file is a placeholder for final paper reproduction instructions.

## Environment

```bash
conda create -n coduar python=3.10 -y
conda activate coduar
pip install -r requirements.txt
```

## Training

```bash
bash scripts/train_coduar.sh configs/train_coduar.example.yaml
```

Fill before release:

| Item | Value |
| --- | --- |
| Dataset version | TODO |
| Train/val/test split | TODO |
| Backbone checkpoint | TODO |
| Hardware | TODO |
| Training time | TODO |
| Random seeds | TODO |

## Evaluation

```bash
bash scripts/evaluate_coduar.sh configs/evaluate_coduar.example.yaml
```

Fill final metrics after the manuscript tables are frozen:

| Method | Metric | Value |
| --- | --- | --- |
| CoDuAR | TODO | TODO |
| Baseline | TODO | TODO |

## Temporal Refinement

Document the final temporal refinement settings here:

- method: TODO
- window size: TODO
- PMI/sequence weight: TODO
- calibration procedure: TODO
