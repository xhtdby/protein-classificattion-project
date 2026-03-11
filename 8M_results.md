# COMP0082 Project - Final Results (Post-Fix Re-run)

## Best Model: XGBoost (ESM-2) with balanced sample weights
- Macro F1: 0.5306, Balanced Acc: 0.5079, MCC: 0.5812, Acc: 0.8645
- Saved at: outputs/models/best_model.joblib
- Feature source: ESM-2 (320-dim, esm2_t6_8M_UR50D)
- Key fix: Added compute_sample_weight("balanced") → +14% Macro F1 vs unweighted

## All 4 Advanced Models
| Model | Acc | F1 | BA | MCC |
|---|---|---|---|---|
| XGBoost (ESM-2) | 0.8645 | **0.5306** | 0.5079 | **0.5812** |
| LightGBM (ESM-2) | 0.8667 | 0.5150 | 0.4795 | 0.5751 |
| LightGBM+SMOTE | 0.8527 | 0.5259 | 0.5225 | 0.5654 |
| XGBoost+SMOTE | 0.8492 | 0.5269 | **0.5405** | 0.5685 |

## Ablation (all with balanced weights)
| Features | F1 | MCC |
|---|---|---|
| ESM-2 only | 0.5306 | 0.5812 |
| Handcrafted only | 0.2468 | 0.3043 |
| ESM-2+Handcrafted | 0.5021 | 0.5698 |
| **ESM-2+Physicochemical** | **0.5346** | **0.5877** |

## Confidence Calibration
- High (p≥0.80): 30,404 (76.5%), acc=95.0%
- Medium: 6,697 (16.8%), acc=65.9%
- Low: 2,663 (6.7%), acc=40.5%

## Key Insights
- ESM-2 >> handcrafted (F1: 0.53 vs 0.25)
- balanced sample_weight boosted XGBoost F1 by +14%
- ESM-2 + Physicochemical is best ablation combo (F1=0.535, MCC=0.588)
- Adding ALL handcrafted features to ESM-2 hurts (noise dilution)
- Weak classes: Lyase F1=0.24, Isomerase F1=0.30
w