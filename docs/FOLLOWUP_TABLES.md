# Follow-up experiments E1–E3: full tables (2026-10-19)

Rendered by `runs/setup/tools/followup_tables.py` from `runs/setup/followup-report-20261019-201913.json`. Narrative: `docs/FOLLOWUP_REPORT.md`.

### E1 success

| task | camera | full | crop | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| bin-picking | training | 0.618 | 0.606 | -0.012 [-0.955, +0.920] | -0.02 | within margin (0.87) | +0.008 [-0.234, +0.478] | -0.009 [-0.026, +0.000] |
| bin-picking | held-out | 0.000 | 0.003 | +0.003 [+0.000, +0.005] | +1.13 | within margin (0.10) | +0.001 [-0.000, +0.002] | +0.000 [+0.000, +0.000] |
| bin-picking | trajectory | 0.583 | 0.560 | -0.023 [-0.880, +0.865] | -0.04 | within margin (0.81) | -0.001 [-0.244, +0.443] | -0.009 [-0.028, +0.000] |
| coffee-push | training | 0.441 | 0.478 | +0.037 [+0.022, +0.055] | +0.30 | within margin (0.16) | +0.089 [-0.011, +0.186] | +0.035 [-0.003, +0.064] |
| coffee-push | held-out | 0.184 | 0.126 | -0.058 [-0.067, -0.047] | -1.63 | within margin (0.10) | -0.035 [-0.048, -0.020] | -0.030 [-0.058, +0.008] |
| coffee-push | trajectory | 0.462 | 0.540 | +0.078 [+0.030, +0.150] | +0.80 | within margin (0.13) | +0.095 [+0.016, +0.181] | +0.035 [+0.002, +0.088] |
| door-open | training | 0.966 | 0.980 | +0.014 [-0.002, +0.043] | +0.55 | within margin (0.10) | +0.065 [-0.008, +0.105] | +0.167 [-0.018, +0.286] |
| door-open | held-out | 0.355 | 0.327 | -0.028 [-0.060, +0.020] | -0.17 | within margin (0.21) | +0.012 [-0.016, +0.054] | +0.062 [-0.004, +0.145] |
| door-open | trajectory | 0.967 | 0.950 | -0.017 [-0.060, +0.030] | -0.44 | within margin (0.10) | +0.035 [-0.027, +0.071] | +0.140 [-0.035, +0.256] |
| hammer | training | 0.890 | 0.949 | +0.059 [+0.027, +0.115] | +1.17 | within margin (0.10) | +0.009 [-0.136, +0.098] | -0.094 [-0.276, +0.011] |
| hammer | held-out | 0.148 | 0.211 | +0.062 [+0.060, +0.068] | +2.05 | within margin (0.10) | +0.041 [+0.023, +0.058] | -0.006 [-0.034, +0.025] |
| hammer | trajectory | 0.865 | 0.893 | +0.028 [-0.020, +0.090] | +0.37 | within margin (0.10) | -0.014 [-0.135, +0.074] | -0.119 [-0.292, -0.032] |
| peg-insert-side | training | 0.122 | 0.254 | +0.132 [+0.047, +0.278] | +1.23 | within margin (0.14) | +0.046 [-0.021, +0.118] | +0.003 [-0.001, +0.007] |
| peg-insert-side | held-out | 0.003 | 0.014 | +0.011 [+0.005, +0.020] | +1.44 | within margin (0.10) | +0.002 [+0.002, +0.003] | +0.000 [-0.000, +0.000] |
| peg-insert-side | trajectory | 0.138 | 0.258 | +0.120 [+0.040, +0.225] | +0.82 | within margin (0.19) | +0.036 [-0.012, +0.097] | +0.002 [+0.001, +0.004] |
| peg-unplug-side | training | 0.823 | 0.885 | +0.062 [-0.017, +0.137] | +1.22 | within margin (0.10) | +0.020 [-0.023, +0.084] | -0.017 [-0.071, +0.040] |
| peg-unplug-side | held-out | 0.193 | 0.210 | +0.017 [-0.030, +0.057] | +0.56 | within margin (0.10) | +0.007 [-0.021, +0.053] | +0.002 [-0.031, +0.047] |
| peg-unplug-side | trajectory | 0.817 | 0.772 | -0.045 [-0.130, +0.035] | -0.83 | within margin (0.10) | -0.102 [-0.180, -0.053] | -0.119 [-0.159, -0.082] |
| pick-place | training | 0.506 | 0.356 | -0.149 [-0.285, +0.048] | -1.13 | within margin (0.17) | -0.069 [-0.113, +0.001] | +0.050 [+0.025, +0.065] |
| pick-place | held-out | 0.027 | 0.008 | -0.019 [-0.028, -0.012] | -1.23 | within margin (0.10) | -0.007 [-0.011, -0.004] | +0.001 [-0.003, +0.005] |
| pick-place | trajectory | 0.438 | 0.295 | -0.143 [-0.175, -0.085] | -3.12 | worse (0.10) | -0.083 [-0.112, -0.037] | +0.042 [+0.013, +0.073] |
| stick-push | training | 0.998 | 0.993 | -0.006 [-0.015, +0.002] | -0.82 | within margin (0.10) | -0.066 [-0.127, +0.015] | -0.207 [-0.402, +0.042] |
| stick-push | held-out | 0.156 | 0.081 | -0.075 [-0.100, -0.030] | -1.42 | within margin (0.10) | -0.016 [-0.084, +0.031] | -0.032 [-0.065, -0.012] |
| stick-push | trajectory | 0.980 | 0.782 | -0.198 [-0.430, -0.050] | -1.13 | within margin (0.23) | -0.188 [-0.330, -0.107] | -0.269 [-0.397, -0.141] |

### E1 learning efficiency and return (training cameras)

| task | time to 0.5, full → crop (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| bin-picking | 690 → 697 (2/3 → 2/3; Δ +7 [-520, +290]) | 800 → 717 (2/3 → 2/3; Δ -83 [-480, +160]) | -85 [-1637, +1393] | +45 [-352, +722] |
| coffee-push | 970 → 567 (1/3 → 2/3; Δ -403 [-630, +0]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | +46 [+2, +99] | +37 [-2, +69] |
| door-open | 143 → 90 (3/3 → 3/3; Δ -53 [-100, +0]) | 253 → 170 (3/3 → 3/3; Δ -83 [-160, +30]) | +41 [-49, +142] | +80 [-26, +181] |
| hammer | 177 → 187 (3/3 → 3/3; Δ +10 [-40, +50]) | 457 → 397 (3/3 → 3/3; Δ -60 [-240, +250]) | +78 [-40, +202] | +33 [-41, +178] |
| peg-insert-side | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | +250 [+94, +390] | +117 [-27, +217] |
| peg-unplug-side | 100 → 107 (3/3 → 3/3; Δ +7 [-10, +20]) | 513 → 430 (3/3 → 3/3; Δ -83 [-310, +160]) | +299 [+202, +472] | +208 [+143, +313] |
| pick-place | 617 → 917 (3/3 → 1/3; Δ +300 [+210, +440]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -219 [-369, -41] | -94 [-151, -42] |
| stick-push | 87 → 150 (3/3 → 3/3; Δ +63 [-20, +200]) | 200 → 233 (3/3 → 3/3; Δ +33 [-70, +90]) | -520 [-724, -408] | -354 [-474, -249] |

### E1 aggregate, eight tasks

| metric | mean Δ over tasks [task-stratified 95% CI] | IQM crop [CI] | IQM full [CI] |
|---|---|---|---|
| train/success/last5 | +0.017 [-0.072, +0.107] | 0.762 [0.677, 0.826] | 0.756 [0.670, 0.811] |
| heldout/success/last5 | -0.011 [-0.034, +0.013] | 0.102 [0.094, 0.112] | 0.126 [0.111, 0.140] |
| traj/success/last5 | -0.025 [-0.111, +0.062] | 0.686 [0.598, 0.765] | 0.721 [0.650, 0.771] |
| train/success/auc | +0.013 [-0.039, +0.062] | 0.524 [0.485, 0.566] | 0.510 [0.468, 0.556] |
| heldout/success/auc | +0.001 [-0.017, +0.017] | 0.104 [0.097, 0.111] | 0.105 [0.097, 0.114] |
| traj/success/auc | -0.028 [-0.077, +0.019] | 0.473 [0.430, 0.521] | 0.509 [0.470, 0.548] |
| train/success/early_auc | -0.009 [-0.054, +0.035] | 0.221 [0.170, 0.266] | 0.242 [0.200, 0.288] |
| heldout/success/early_auc | -0.000 [-0.018, +0.017] | 0.065 [0.052, 0.083] | 0.070 [0.060, 0.084] |
| traj/success/early_auc | -0.037 [-0.074, -0.001] | 0.214 [0.175, 0.256] | 0.269 [0.227, 0.312] |
| time_to_0.5/steps (k steps) | -9 [-80, +67] | | |
| time_to_0.8/steps (k steps) | -35 [-87, +20] | | |
| train/return/last5 | -14 [-156, +126] | | |
| train/return/auc | +9 [-64, +78] | | |

### E1 aggregate, six nondev tasks

| metric | mean Δ over tasks [task-stratified 95% CI] | IQM crop [CI] | IQM full [CI] |
|---|---|---|---|
| train/success/last5 | +0.038 [-0.078, +0.154] | 0.786 [0.665, 0.856] | 0.756 [0.610, 0.829] |
| heldout/success/last5 | -0.022 [-0.052, +0.008] | 0.101 [0.092, 0.115] | 0.139 [0.121, 0.155] |
| traj/success/last5 | -0.014 [-0.130, +0.101] | 0.704 [0.593, 0.786] | 0.749 [0.619, 0.802] |
| train/success/auc | +0.027 [-0.037, +0.088] | 0.561 [0.498, 0.606] | 0.524 [0.463, 0.587] |
| heldout/success/auc | -0.005 [-0.027, +0.017] | 0.114 [0.107, 0.123] | 0.128 [0.120, 0.140] |
| traj/success/auc | -0.021 [-0.082, +0.039] | 0.499 [0.432, 0.553] | 0.535 [0.478, 0.589] |
| train/success/early_auc | -0.005 [-0.057, +0.048] | 0.271 [0.197, 0.335] | 0.284 [0.249, 0.323] |
| heldout/success/early_auc | +0.000 [-0.022, +0.023] | 0.091 [0.072, 0.111] | 0.092 [0.077, 0.117] |
| traj/success/early_auc | -0.037 [-0.078, +0.005] | 0.255 [0.200, 0.304] | 0.316 [0.279, 0.351] |
| time_to_0.5/steps (k steps) | -63 [-151, +30] | | |
| time_to_0.8/steps (k steps) | -36 [-96, +27] | | |
| train/return/last5 | +5 [-185, +189] | | |
| train/return/auc | +22 [-68, +108] | | |

### E1 representation (crop vs full, 200k full-split evaluation)

- **bin-picking**: hammer margins: 3 better / 4 worse of 32 — psnr_traj -0.425 (worse), cd_centers_vis -7.85% (better), cd_motion_dyn +8.5% (worse), psnr_near -0.537 (worse), psnr_heldout -0.207 (worse), retrieval_heldout +0.152 (better), cd_centers -7.79% (better); pick-place margins: 3 better / 4 worse of 32 — psnr_traj -0.425 (worse), cd_centers_vis -7.85% (better), psnr_near -0.537 (worse), psnr_near_covered -0.335 (worse), psnr_traj_covered -0.246 (worse), retrieval_heldout +0.152 (better), cd_centers -7.79% (better)
- **coffee-push**: hammer margins: 4 better / 2 worse of 32 — retrieval_traj -0.0633 (worse), rel_epe -0.0556 (better), relepe_bin1 -0.0917 (better), relepe_bin2 -0.0597 (better), relepe_bin3 -0.0339 (better), cd_motion_dyn +18.5% (worse); pick-place margins: 4 better / 1 worse of 32 — retrieval_traj -0.0633 (worse), rel_epe -0.0556 (better), relepe_bin2 -0.0597 (better), relepe_bin3 -0.0339 (better), epe_mm -1.27 (better)
- **door-open**: hammer margins: 0 better / 1 worse of 32 — cd_motion_dyn +6.14% (worse); pick-place margins: 0 better / 6 worse of 32 — psnr -0.383 (worse), psnr_moving -0.464 (worse), epe_mm +1.11 (worse), cd_render_traj +5.94% (worse), psnr_near_covered -0.379 (worse), psnr_traj_covered -0.443 (worse)
- **hammer**: hammer margins: 0 better / 6 worse of 32 — r2_vel_traj -0.0607 (worse), r2_vel_train -0.0535 (worse), hidden +0.0433 (worse), cd_centers_vis +5.42% (worse), cd_motion_dyn +5.38% (worse), psnr_heldout -0.239 (worse)
- **peg-insert-side**: hammer margins: 0 better / 2 worse of 32 — cd_motion_dyn +73.9% (worse), cd_centers +6.1% (worse); pick-place margins: 0 better / 8 worse of 32 — psnr -0.396 (worse), psnr_moving -0.295 (worse), hidden +0.0372 (worse), cd_motion_dyn +73.9% (worse), psnr_near -0.33 (worse), psnr_near_covered -0.454 (worse), psnr_traj_covered -0.331 (worse), cd_centers +6.1% (worse)
- **peg-unplug-side**: hammer margins: 4 better / 2 worse of 32 — r2_vel_traj +0.109 (better), relepe_bin3 -0.0433 (better), cd_centers_vis +8.83% (worse), cd_motion_dyn -49% (better), psnr_near +0.488 (better), cd_centers +6.06% (worse); pick-place margins: 5 better / 3 worse of 32 — psnr_moving -0.578 (worse), r2_vel_traj +0.109 (better), relepe_bin3 -0.0433 (better), cd_centers_vis +8.83% (worse), cd_motion_dyn -49% (better), psnr_near +0.488 (better), r2_vel_near +0.0759 (better), cd_centers +6.06% (worse)
- **pick-place**: pick-place margins: 0 better / 3 worse of 32 — retrieval_traj -0.0458 (worse), cd_centers_vis +8.8% (worse), cd_centers +7.05% (worse)
- **stick-push**: hammer margins: 5 better / 1 worse of 32 — retrieval_traj -0.0401 (worse), r2_vel_traj +0.121 (better), r2_vel_train +0.0733 (better), rel_epe -0.0446 (better), relepe_bin2 -0.0713 (better), r2_vel_near +0.0939 (better); pick-place margins: 5 better / 2 worse of 32 — psnr_moving -0.488 (worse), retrieval_traj -0.0401 (worse), r2_vel_traj +0.121 (better), r2_vel_train +0.0733 (better), rel_epe -0.0446 (better), relepe_bin2 -0.0713 (better), r2_vel_near +0.0939 (better)

### E2 hammer: shift vs full

| task | camera | full | shift | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| hammer | training | 0.935 | 0.931 | -0.004 [-0.030, +0.035] | -0.06 | within margin (0.10) | +0.030 [-0.038, +0.166] | +0.081 [-0.009, +0.255] |
| hammer | held-out | 0.149 | 0.127 | -0.023 [-0.108, +0.092] | -0.33 | within margin (0.10) | -0.018 [-0.074, +0.055] | -0.011 [-0.068, +0.030] |
| hammer | trajectory | 0.882 | 0.908 | +0.027 [-0.030, +0.115] | +0.40 | within margin (0.10) | +0.030 [-0.046, +0.152] | +0.075 [-0.012, +0.219] |

| task | time to 0.5, full → shift (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| hammer | 213 → 153 (3/3 → 3/3; Δ -60 [-180, +0]) | 443 → 450 (3/3 → 3/3; Δ +7 [-120, +160]) | +3 [-43, +85] | +62 [-68, +262] |

### E2 hammer: shift vs crop

| task | camera | crop | shift | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| hammer | training | 0.970 | 0.931 | -0.039 [-0.118, +0.072] | -0.58 | within margin (0.10) | -0.083 [-0.156, +0.039] | -0.040 [-0.230, +0.088] |
| hammer | held-out | 0.228 | 0.127 | -0.102 [-0.130, -0.080] | -2.68 | worse (0.10) | -0.088 [-0.125, -0.056] | -0.046 [-0.106, -0.005] |
| hammer | trajectory | 0.917 | 0.908 | -0.008 [-0.070, +0.105] | -0.12 | within margin (0.10) | -0.067 [-0.152, +0.043] | -0.038 [-0.213, +0.054] |

| task | time to 0.5, crop → shift (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| hammer | 160 → 153 (3/3 → 3/3; Δ -7 [-120, +80]) | 300 → 450 (3/3 → 3/3; Δ +150 [-20, +390]) | -77 [-230, +182] | -33 [-209, +133] |

### E2 hammer: crop vs full

| task | camera | full | crop | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| hammer | training | 0.935 | 0.970 | +0.035 [-0.037, +0.102] | +0.66 | within margin (0.10) | +0.113 [+0.093, +0.128] | +0.121 [-0.097, +0.232] |
| hammer | held-out | 0.149 | 0.228 | +0.079 [-0.013, +0.172] | +1.19 | within margin (0.10) | +0.070 [+0.009, +0.111] | +0.035 [+0.031, +0.038] |
| hammer | trajectory | 0.882 | 0.917 | +0.035 [+0.010, +0.065] | +0.64 | within margin (0.10) | +0.097 [+0.075, +0.109] | +0.113 [-0.067, +0.232] |

| task | time to 0.5, full → crop (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| hammer | 213 → 160 (3/3 → 3/3; Δ -53 [-200, +120]) | 443 → 300 (3/3 → 3/3; Δ -143 [-230, -100]) | +80 [-96, +197] | +95 [-45, +200] |

### E2 pick-place: shift vs full

| task | camera | full | shift | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| pick-place | training | 0.463 | 0.493 | +0.031 [-0.213, +0.240] | +0.18 | within margin (0.22) | -0.016 [-0.129, +0.149] | -0.021 [-0.077, +0.016] |
| pick-place | held-out | 0.011 | 0.014 | +0.003 [+0.003, +0.005] | +0.38 | within margin (0.10) | -0.001 [-0.004, +0.003] | -0.003 [-0.009, +0.001] |
| pick-place | trajectory | 0.423 | 0.350 | -0.073 [-0.195, +0.045] | -0.84 | within margin (0.11) | -0.042 [-0.135, +0.099] | -0.020 [-0.070, +0.007] |

| task | time to 0.5, full → shift (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| pick-place | 807 → 713 (1/3 → 3/3; Δ -93 [-480, +280]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -63 [-342, +120] | -47 [-169, +75] |

### E2 pick-place: shift vs crop

| task | camera | crop | shift | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| pick-place | training | 0.455 | 0.493 | +0.038 [-0.075, +0.218] | +0.38 | within margin (0.13) | -0.016 [-0.094, +0.114] | -0.030 [-0.043, -0.007] |
| pick-place | held-out | 0.005 | 0.014 | +0.009 [-0.005, +0.020] | +1.14 | within margin (0.10) | +0.002 [+0.001, +0.004] | -0.001 [-0.007, +0.005] |
| pick-place | trajectory | 0.387 | 0.350 | -0.037 [-0.105, +0.070] | -0.52 | within margin (0.10) | -0.020 [-0.072, +0.061] | -0.018 [-0.032, -0.006] |

| task | time to 0.5, crop → shift (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| pick-place | 793 → 713 (2/3 → 3/3; Δ -80 [-480, +190]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | +28 [-34, +75] | -14 [-35, +19] |

### E2 pick-place: crop vs full

| task | camera | full | crop | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| pick-place | training | 0.463 | 0.455 | -0.008 [-0.138, +0.093] | -0.04 | within margin (0.23) | +0.001 [-0.060, +0.035] | +0.008 [-0.038, +0.040] |
| pick-place | held-out | 0.011 | 0.005 | -0.006 [-0.017, +0.010] | -0.65 | within margin (0.10) | -0.003 [-0.007, +0.002] | -0.002 [-0.007, +0.005] |
| pick-place | trajectory | 0.423 | 0.387 | -0.037 [-0.090, +0.005] | -0.34 | within margin (0.14) | -0.021 [-0.084, +0.038] | -0.002 [-0.038, +0.022] |

| task | time to 0.5, full → crop (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| pick-place | 807 → 793 (1/3 → 2/3; Δ -13 [-270, +230]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -91 [-309, +45] | -33 [-142, +57] |

### E2 representation

- **hammer, shift vs full**: hammer margins: 1 better / 3 worse of 32 — psnr -0.541 (worse), psnr_traj -0.273 (worse), retrieval_traj -0.0536 (worse), cd_motion_dyn -5.5% (better)
- **hammer, shift vs crop**: hammer margins: 1 better / 2 worse of 32 — psnr_traj -0.288 (worse), retrieval_traj -0.05 (worse), cd_motion_dyn -10.3% (better)
- **pick-place, shift vs full**: pick-place margins: 0 better / 9 worse of 32 — psnr -0.785 (worse), psnr_moving -1.71 (worse), psnr_traj -0.697 (worse), retrieval_traj -0.101 (worse), psnr_near -0.884 (worse), psnr_near_covered -0.779 (worse), psnr_traj_covered -0.66 (worse), psnr_heldout_covered -0.443 (worse), retrieval_near -0.0407 (worse)
- **pick-place, shift vs crop**: pick-place margins: 3 better / 9 worse of 32 — psnr -0.619 (worse), psnr_moving -1.51 (worse), psnr_traj -0.556 (worse), retrieval_traj -0.055 (worse), cd_centers_vis -11.2% (better), psnr_near -0.997 (worse), psnr_near_covered -0.646 (worse), psnr_traj_covered -0.568 (worse), psnr_heldout_covered -0.309 (worse), retrieval_near -0.0316 (worse), retrieval_heldout +0.0505 (better), cd_centers -9.75% (better)

### E3 success

| task | camera | CNN | no shift | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |
|---|---|---|---|---|---|---|---|---|
| bin-picking | training | 0.631 | 0.000 | -0.631 [-0.977, -0.312] | -2.14 | worse (0.38) | -0.276 [-0.429, -0.191] | -0.006 [-0.010, -0.001] |
| bin-picking | held-out | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | NA | within margin (0.10) | -0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.000] |
| bin-picking | trajectory | 0.512 | 0.000 | -0.512 [-0.830, +0.000] | -1.29 | within margin (0.52) | -0.218 [-0.494, -0.001] | -0.007 [-0.018, -0.001] |
| coffee-push | training | 0.721 | 0.119 | -0.602 [-0.643, -0.535] | -15.84 | worse (0.10) | -0.322 [-0.363, -0.294] | -0.112 [-0.138, -0.094] |
| coffee-push | held-out | 0.023 | 0.038 | +0.015 [-0.017, +0.045] | +0.44 | within margin (0.10) | +0.003 [-0.004, +0.016] | -0.014 [-0.034, +0.002] |
| coffee-push | trajectory | 0.752 | 0.138 | -0.613 [-0.730, -0.525] | -8.95 | worse (0.10) | -0.343 [-0.394, -0.312] | -0.134 [-0.137, -0.131] |
| door-open | training | 0.913 | 0.291 | -0.622 [-0.743, -0.407] | -2.91 | worse (0.28) | -0.628 [-0.717, -0.482] | -0.415 [-0.470, -0.381] |
| door-open | held-out | 0.033 | 0.000 | -0.033 [-0.083, +0.000] | -0.84 | within margin (0.10) | -0.035 [-0.064, -0.002] | -0.020 [-0.036, -0.003] |
| door-open | trajectory | 0.973 | 0.035 | -0.938 [-0.980, -0.875] | -17.20 | worse (0.10) | -0.808 [-0.831, -0.775] | -0.561 [-0.616, -0.533] |
| hammer | training | 0.522 | 0.023 | -0.499 [-0.693, -0.122] | -1.84 | worse (0.35) | -0.342 [-0.489, -0.097] | -0.069 [-0.139, -0.007] |
| hammer | held-out | 0.021 | 0.022 | +0.001 [-0.010, +0.015] | +0.03 | within margin (0.10) | +0.016 [-0.010, +0.040] | +0.028 [-0.042, +0.086] |
| hammer | trajectory | 0.107 | 0.022 | -0.085 [-0.160, -0.005] | -1.11 | within margin (0.10) | -0.031 [-0.091, +0.009] | +0.008 [+0.006, +0.010] |
| peg-insert-side | training | 0.486 | 0.002 | -0.484 [-0.570, -0.408] | -6.66 | worse (0.10) | -0.194 [-0.205, -0.182] | -0.004 [-0.009, -0.000] |
| peg-insert-side | held-out | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | NA | within margin (0.10) | -0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.000] |
| peg-insert-side | trajectory | 0.715 | 0.000 | -0.715 [-0.800, -0.635] | -9.79 | worse (0.10) | -0.316 [-0.321, -0.312] | -0.006 [-0.009, -0.002] |
| peg-unplug-side | training | 0.818 | 0.112 | -0.706 [-0.767, -0.633] | -10.05 | worse (0.10) | -0.450 [-0.481, -0.424] | -0.154 [-0.157, -0.151] |
| peg-unplug-side | held-out | 0.003 | 0.039 | +0.037 [+0.018, +0.050] | +2.79 | within margin (0.10) | +0.024 [+0.010, +0.040] | +0.015 [+0.005, +0.026] |
| peg-unplug-side | trajectory | 0.948 | 0.105 | -0.843 [-0.900, -0.785] | -21.16 | worse (0.10) | -0.592 [-0.660, -0.512] | -0.231 [-0.308, -0.166] |
| pick-place | training | 0.551 | 0.015 | -0.536 [-0.625, -0.478] | -7.34 | worse (0.10) | -0.197 [-0.236, -0.143] | -0.011 [-0.017, -0.006] |
| pick-place | held-out | 0.005 | 0.000 | -0.005 [-0.015, +0.000] | -0.65 | within margin (0.10) | -0.003 [-0.006, -0.002] | -0.002 [-0.003, -0.002] |
| pick-place | trajectory | 0.585 | 0.018 | -0.567 [-0.580, -0.550] | -26.54 | worse (0.10) | -0.167 [-0.189, -0.123] | -0.005 [-0.007, -0.003] |
| stick-push | training | 0.945 | 0.679 | -0.266 [-0.413, -0.080] | -2.35 | worse (0.15) | -0.324 [-0.458, -0.214] | -0.239 [-0.290, -0.200] |
| stick-push | held-out | 0.038 | 0.052 | +0.013 [-0.025, +0.062] | +0.40 | within margin (0.10) | -0.022 [-0.028, -0.015] | +0.000 [-0.008, +0.007] |
| stick-push | trajectory | 0.935 | 0.445 | -0.490 [-0.795, -0.230] | -1.81 | worse (0.35) | -0.448 [-0.677, -0.331] | -0.196 [-0.312, -0.033] |

### E3 learning efficiency and return (training cameras)

| task | time to 0.5, CNN → no shift (k steps; reach) | time to 0.8 (k steps; reach) | Δ return last-5 [CI] | Δ return AUC [CI] |
|---|---|---|---|---|
| bin-picking | 797 → 1000 (2/3 → 0/3; Δ +203 [+0, +470]) | 920 → 1000 (1/3 → 0/3; Δ +80 [+0, +240]) | -1051 [-1684, -540] | -470 [-732, -330] |
| coffee-push | 540 → 1000 (3/3 → 0/3; Δ +460 [+370, +580]) | 980 → 1000 (1/3 → 0/3; Δ +20 [+0, +60]) | -915 [-1006, -782] | -341 [-418, -290] |
| door-open | 150 → 933 (3/3 → 1/3; Δ +783 [+660, +870]) | 397 → 1000 (3/3 → 0/3; Δ +603 [+540, +640]) | -1017 [-1241, -701] | -890 [-1008, -687] |
| hammer | 590 → 1000 (2/3 → 0/3; Δ +410 [+0, +660]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -1007 [-1373, -378] | -650 [-847, -259] |
| peg-insert-side | 947 → 1000 (2/3 → 0/3; Δ +53 [+0, +150]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -970 [-1097, -884] | -424 [-434, -404] |
| peg-unplug-side | 337 → 1000 (3/3 → 0/3; Δ +663 [+640, +680]) | 767 → 1000 (3/3 → 0/3; Δ +233 [+160, +310]) | -1359 [-1467, -1281] | -703 [-713, -689] |
| pick-place | 900 → 1000 (3/3 → 0/3; Δ +100 [+30, +190]) | 1000 → 1000 (0/3 → 0/3; Δ +0 [+0, +0]) | -881 [-1021, -739] | -338 [-399, -248] |
| stick-push | 227 → 537 (3/3 → 3/3; Δ +310 [+140, +520]) | 327 → 877 (3/3 → 2/3; Δ +550 [+290, +700]) | -1273 [-1517, -860] | -834 [-961, -648] |

### E3 viewpoint: Δ(novel) − Δ(training), raw success units

| task | held-out last-5 | trajectory last-5 | held-out AUC | trajectory AUC |
|---|---|---|---|---|
| bin-picking | +0.631 [+0.312, +0.977] | +0.119 [-0.518, +0.603] | +0.276 [+0.191, +0.429] | +0.057 [-0.287, +0.268] |
| coffee-push | +0.617 [+0.553, +0.688] | -0.011 [-0.087, +0.103] | +0.325 [+0.305, +0.359] | -0.021 [-0.085, +0.052] |
| door-open | +0.589 [+0.407, +0.700] | -0.317 [-0.468, -0.217] | +0.592 [+0.480, +0.677] | -0.181 [-0.293, -0.100] |
| hammer | +0.500 [+0.112, +0.708] | +0.414 [+0.117, +0.603] | +0.358 [+0.138, +0.479] | +0.312 [+0.107, +0.429] |
| peg-insert-side | +0.484 [+0.408, +0.570] | -0.231 [-0.325, -0.065] | +0.194 [+0.182, +0.204] | -0.123 [-0.134, -0.116] |
| peg-unplug-side | +0.742 [+0.651, +0.809] | -0.138 [-0.212, -0.068] | +0.474 [+0.435, +0.503] | -0.141 [-0.179, -0.066] |
| pick-place | +0.531 [+0.478, +0.610] | -0.031 [-0.102, +0.055] | +0.194 [+0.140, +0.230] | +0.030 [-0.046, +0.089] |
| stick-push | +0.279 [+0.083, +0.476] | -0.224 [-0.382, +0.075] | +0.301 [+0.200, +0.434] | -0.125 [-0.219, -0.038] |
| **pooled (8 tasks)** | +0.547 [+0.485, +0.609] | -0.052 [-0.138, +0.028] | +0.339 [+0.306, +0.372] | -0.024 [-0.071, +0.018] |

### E3 aggregate, eight tasks

| metric | mean Δ over tasks [task-stratified 95% CI] | IQM no shift [CI] | IQM CNN [CI] |
|---|---|---|---|
| train/success/last5 | -0.543 [-0.606, -0.480] | 0.060 [0.044, 0.078] | 0.718 [0.666, 0.768] |
| heldout/success/last5 | +0.004 [-0.005, +0.013] | 0.009 [0.006, 0.016] | 0.008 [0.004, 0.014] |
| traj/success/last5 | -0.595 [-0.657, -0.529] | 0.043 [0.030, 0.060] | 0.771 [0.735, 0.809] |
| train/success/auc | -0.342 [-0.376, -0.306] | 0.040 [0.029, 0.051] | 0.409 [0.360, 0.454] |
| heldout/success/auc | -0.002 [-0.007, +0.003] | 0.012 [0.009, 0.015] | 0.012 [0.008, 0.018] |
| traj/success/auc | -0.365 [-0.405, -0.330] | 0.028 [0.024, 0.035] | 0.438 [0.400, 0.484] |
| train/success/early_auc | -0.126 [-0.138, -0.114] | 0.016 [0.013, 0.018] | 0.120 [0.103, 0.137] |
| heldout/success/early_auc | +0.001 [-0.006, +0.007] | 0.008 [0.006, 0.011] | 0.011 [0.008, 0.017] |
| traj/success/early_auc | -0.142 [-0.164, -0.118] | 0.019 [0.016, 0.023] | 0.116 [0.091, 0.142] |
| time_to_0.5/steps (k steps) | +373 [+313, +430] | | |
| time_to_0.8/steps (k steps) | +186 [+155, +216] | | |
| train/return/last5 | -1059 [-1162, -960] | | |
| train/return/auc | -581 [-633, -528] | | |

Novel/training ratio NA (training last-5 success 0): {'bin-picking': ['s4-drm-cnnnoshift-bin-picking-s2000', 's4-drm-cnnnoshift-bin-picking-s2001', 's4-drm-cnnnoshift-bin-picking-s2002'], 'peg-insert-side': ['s4-drm-cnnnoshift-peg-insert-side-s2001']}
