# WoGp Rebuttal — All Experimental Results

## Exp 1: External 40/10/50 split for R2CCP (addresses R1)

- Per-seed records: 60 across 3 judges, 2 protocols, 10 seeds.

| Judge | Protocol | Cov(raw) | Width(raw) | Cov(adj) | Width(adj) |
|---|---|---|---|---|---|
| Gemini | A_internal_split | 0.8963±0.0122 | 2.8565±0.0900 | 0.9808±0.0030 | 3.4214±0.0566 |
| Gemini | B_external_split | 0.8973±0.0115 | 2.8865±0.0675 | 0.9805±0.0040 | 3.4345±0.0524 |
| LLaVA-Critic | A_internal_split | 0.8930±0.0141 | 3.0148±0.1036 | 0.9782±0.0050 | 3.5547±0.0871 |
| LLaVA-Critic | B_external_split | 0.8937±0.0147 | 3.0408±0.0948 | 0.9783±0.0063 | 3.5740±0.0781 |
| Phi-4 | A_internal_split | 0.8927±0.0087 | 3.1474±0.0531 | 0.9819±0.0049 | 3.7041±0.0410 |
| Phi-4 | B_external_split | 0.8898±0.0122 | 3.1454±0.0755 | 0.9808±0.0062 | 3.6909±0.0574 |

### Paired t-test (per-seed Protocol A vs Protocol B)

| Judge | Δ cov_raw (A−B) | p-value (cov_raw) | Δ width_raw (A−B) | p-value (width_raw) |
|---|---|---|---|---|
| LLaVA-Critic | -0.0006 | 0.772 | -0.0259 | 0.093 |
| Phi-4 | +0.0028 | 0.377 | +0.0020 | 0.908 |
| Gemini | -0.0010 | 0.566 | -0.0300 | 0.015 |

*Interpretation:* if p > 0.05 for both, the two protocols are statistically indistinguishable, confirming that the conformal quantile is unchanged whether R2CCP splits internally or whether we hold out an external 10% calibration set.


## Exp 2: Polaris single-annotator ablation (addresses R4)

- Per-seed records: 20 across 2 arms × 10 seeds.

| Arm | Cov(raw) | Width(raw) | Cov(adj) | Width(adj) |
|---|---|---|---|---|
| multi_annotator | 0.9015±0.0143 | 0.7168±0.1416 | 0.9868±0.0025 | 1.4798±0.0466 |
| single_annotator | 0.9048±0.0087 | 1.0011±0.0646 | 0.9780±0.0028 | 1.6127±0.0421 |

*Paired comparison:* single-annotator width is 1.40× the multi-annotator width on the same Polaris items, Δ = +0.284, p = 0.0004.

*Reference numbers from paper:*
- MLLM-Judge R2CCP raw width: 3.05 (single annotator, integer GT, 14-way VQA)
- Polaris   R2CCP raw width: 0.68 (multi annotator mean, continuous GT mapped to 1-5)
- Polaris   single-annotator: 1.001 (this experiment)

*Implication:* aggregation alone accounts for 14% of the 3.05 - 0.68 = 2.37 width gap (everything else — task type, label continuity, etc. — accounts for the remainder).

## Exp 3: Bootstrap 95% CIs on ρ, width, RSG (addresses R5c)

- Rows: 42 (dataset × judge); N=10 seeds per row; bootstrap B=2000.

### Table: RSG with 95% bootstrap CIs (replaces paper Table 17)

| Judge | Dataset | ρ (95% CI) | Width (95% CI) | RSG (95% CI) |
|---|---|---|---|---|
| Gemini | AesBench | +0.266 [+0.245, +0.288] | 2.193 [2.059, 2.330] | -0.186 [-0.233, -0.135] |
| Gemini | diffusiondb | +0.324 [+0.283, +0.364] | 3.288 [3.035, 3.536] | +0.146 [+0.100, +0.194] |
| Gemini | VisitBench | +0.535 [+0.507, +0.564] | 2.621 [2.383, 2.839] | +0.190 [+0.115, +0.258] |
| Gemini | coco | +0.341 [+0.315, +0.364] | 2.660 [2.439, 2.871] | +0.006 [-0.068, +0.077] |
| Gemini | llava_bench | +0.316 [+0.300, +0.330] | 3.062 [2.859, 3.279] | +0.082 [+0.023, +0.146] |
| Gemini | mm-vet | +0.235 [+0.209, +0.268] | 2.523 [2.099, 2.961] | -0.134 [-0.234, -0.036] |
| Gemini | Concept Caption | +0.440 [+0.425, +0.456] | 2.676 [2.481, 2.900] | +0.109 [+0.061, +0.161] |
| Gemini | WIT | +0.278 [+0.248, +0.309] | 2.712 [2.521, 2.892] | -0.044 [-0.104, +0.019] |
| Gemini | mind2web | +0.067 [+0.041, +0.094] | 2.914 [2.797, 3.042] | -0.205 [-0.233, -0.177] |
| Gemini | ChartQA | +0.492 [+0.467, +0.511] | 2.846 [2.536, 3.130] | +0.203 [+0.124, +0.279] |
| Gemini | ScienceQA | +0.525 [+0.500, +0.545] | 2.946 [2.594, 3.293] | +0.261 [+0.160, +0.351] |
| Gemini | infographicsVQA | +0.543 [+0.522, +0.563] | 2.929 [2.731, 3.113] | +0.275 [+0.218, +0.336] |
| Gemini | mathvista | +0.440 [+0.415, +0.463] | 3.240 [3.090, 3.376] | +0.249 [+0.208, +0.289] |
| Gemini | textVQA | +0.598 [+0.578, +0.614] | 2.233 [2.124, 2.321] | +0.156 [+0.114, +0.187] |
| LLaVA-Critic | AesBench | +0.420 [+0.403, +0.442] | 2.031 [1.855, 2.206] | -0.073 [-0.115, -0.034] |
| LLaVA-Critic | diffusiondb | +0.088 [+0.060, +0.116] | 3.374 [3.200, 3.553] | -0.067 [-0.121, -0.011] |
| LLaVA-Critic | VisitBench | +0.351 [+0.320, +0.378] | 2.937 [2.794, 3.066] | +0.085 [+0.022, +0.142] |
| LLaVA-Critic | coco | +0.375 [+0.345, +0.408] | 2.561 [2.446, 2.716] | +0.015 [-0.031, +0.078] |
| LLaVA-Critic | llava_bench | +0.245 [+0.222, +0.269] | 2.979 [2.700, 3.250] | -0.010 [-0.077, +0.063] |
| LLaVA-Critic | mm-vet | +0.255 [+0.197, +0.309] | 3.074 [2.758, 3.408] | +0.024 [-0.085, +0.128] |
| LLaVA-Critic | Concept Caption | +0.365 [+0.320, +0.405] | 2.670 [2.506, 2.858] | +0.032 [-0.021, +0.081] |
| LLaVA-Critic | WIT | +0.144 [+0.095, +0.194] | 2.625 [2.229, 2.934] | -0.200 [-0.314, -0.115] |
| LLaVA-Critic | mind2web | +0.279 [+0.253, +0.307] | 2.648 [2.446, 2.852] | -0.059 [-0.119, +0.008] |
| LLaVA-Critic | ChartQA | +0.500 [+0.485, +0.519] | 3.003 [2.844, 3.149] | +0.251 [+0.204, +0.294] |
| LLaVA-Critic | ScienceQA | +0.283 [+0.244, +0.325] | 3.159 [2.922, 3.364] | +0.073 [-0.012, +0.154] |
| LLaVA-Critic | infographicsVQA | +0.427 [+0.406, +0.447] | 3.405 [3.221, 3.542] | +0.278 [+0.219, +0.328] |
| LLaVA-Critic | mathvista | +0.383 [+0.362, +0.397] | 3.320 [3.197, 3.412] | +0.213 [+0.168, +0.246] |
| LLaVA-Critic | textVQA | +0.379 [+0.345, +0.411] | 2.878 [2.695, 3.033] | +0.098 [+0.049, +0.145] |
| Phi-4 | AesBench | +0.383 [+0.363, +0.406] | 1.853 [1.640, 2.013] | -0.153 [-0.213, -0.100] |
| Phi-4 | diffusiondb | +0.255 [+0.233, +0.279] | 3.093 [2.785, 3.419] | +0.028 [-0.063, +0.122] |
| Phi-4 | VisitBench | +0.303 [+0.284, +0.325] | 2.865 [2.685, 3.066] | +0.020 [-0.036, +0.085] |
| Phi-4 | coco | +0.240 [+0.180, +0.297] | 2.690 [2.481, 2.909] | -0.087 [-0.175, +0.000] |
| Phi-4 | llava_bench | +0.156 [+0.127, +0.187] | 2.999 [2.793, 3.214] | -0.095 [-0.166, -0.020] |
| Phi-4 | mm-vet | +0.279 [+0.245, +0.312] | 2.383 [1.972, 2.800] | -0.126 [-0.227, -0.017] |
| Phi-4 | Concept Caption | +0.295 [+0.271, +0.317] | 2.664 [2.499, 2.816] | -0.039 [-0.090, -0.005] |
| Phi-4 | WIT | +0.399 [+0.369, +0.437] | 2.601 [2.326, 2.800] | +0.049 [-0.036, +0.115] |
| Phi-4 | mind2web | +0.231 [+0.201, +0.263] | 2.844 [2.662, 3.033] | -0.058 [-0.102, -0.010] |
| Phi-4 | ChartQA | +0.229 [+0.191, +0.266] | 3.594 [3.438, 3.730] | +0.128 [+0.075, +0.178] |
| Phi-4 | ScienceQA | +0.311 [+0.285, +0.336] | 3.364 [3.197, 3.538] | +0.152 [+0.111, +0.197] |
| Phi-4 | infographicsVQA | +0.156 [+0.125, +0.182] | 3.474 [3.356, 3.601] | +0.025 [-0.020, +0.072] |
| Phi-4 | mathvista | +0.347 [+0.328, +0.368] | 3.463 [3.339, 3.585] | +0.212 [+0.183, +0.243] |
| Phi-4 | textVQA | +0.232 [+0.210, +0.255] | 2.927 [2.699, 3.162] | -0.036 [-0.098, +0.025] |

## Width-as-percent-of-score-range, corrected to ÷(K-1)=÷4

Paper §4 defines width range on a 1-5 scale as 0 to 4. Therefore the correct normalization is ÷(K-1)=÷4. Paper §5.4 already uses ÷4 correctly; Table 7 incorrectly used ÷5. Corrected numbers:

| Dataset | Width | Paper (÷5, wrong) | Corrected (÷4) |
|---|---|---|---|
| AesBench | 2.082 | 41.6% | 52.1% |
| mm-vet | 2.180 | 43.6% | 54.5% |
| WIT | 2.377 | 47.5% | 59.4% |
| coco | 2.427 | 48.5% | 60.7% |
| mind2web | 2.690 | 53.8% | 67.3% |
| Concept Caption | 2.703 | 54.1% | 67.6% |
| textVQA | 2.812 | 56.2% | 70.3% |
| llava_bench | 2.920 | 58.4% | 73.0% |
| VisitBench | 2.959 | 59.2% | 74.0% |
| ChartQA | 3.079 | 61.6% | 77.0% |
| ScienceQA | 3.269 | 65.4% | 81.7% |
| mathvista | 3.369 | 67.4% | 84.2% |
| diffusiondb | 3.414 | 68.3% | 85.4% |
| infographicsVQA | 3.504 | 70.1% | 87.6% |

**MLLM-Judge average:** width 2.84 → 71.0% (correct ÷4), NOT 61% as Table 7 states (61% came from incorrectly using 3.05/5).

**Polaris:** width 0.68 → 17.0% (correct ÷4), NOT 14% (14% came from 0.68/5).

**Abstract correction:** the narrowest task (AesBench) covers 52% of the range, not ~40%. The widest task (InfographicsVQA) covers 88%, not ~70%. We will update the abstract to: 'intervals span 52–88% of the score range across 14 task categories, with aesthetics/natural-image tasks at the low end and chart/math reasoning at the high end.'