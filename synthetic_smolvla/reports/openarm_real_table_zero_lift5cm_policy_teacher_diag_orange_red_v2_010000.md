# Policy vs Teacher Action Diagnostic

Simulation only. This report uses stored clean dataset frames and does not run Isaac or real robot hardware.

- Dataset: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/datasets/openarm_real_table_zero_v1_lift5cm_routed_v2_extra_right`
- Checkpoint: `/home/chyanin/Desktop/realrobot/synthetic_smolvla/checkpoints/smolvla_openarm_real_table_zero_lift5cm_routed_v2_from020_lr3e5/checkpoints/010000/pretrained_model_typed`
- Selected episodes: `{"pick up the orange ball": [44, 91, 516, 654], "pick up the red cube": [798, 828, 837, 892]}`
- Policy `n_action_steps`: `10`
- Policy input features: `{"observation.images.camera1": {"shape": [3, 256, 256], "type": "VISUAL"}, "observation.images.camera2": {"shape": [3, 256, 256], "type": "VISUAL"}, "observation.images.camera3": {"shape": [3, 256, 256], "type": "VISUAL"}, "observation.state": {"shape": [6], "type": "STATE"}}`
- Policy output features: `{"action": {"shape": [8], "type": "ACTION"}}`

## Summary

| Group | Count | Mean MAE | P90 MAE | Mean Arm MAE | Mean Grip Abs | P90 Max Abs | Arm Limit Rows | Grip Too Closed Rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 968 | 2.9782 | 7.8948 | 2.8193 | 4.0904 | 21.8546 | 185 | 151 |
| mode=independent | 168 | 2.6850 | 4.7185 | 2.4957 | 4.0103 | 13.4228 | 25 | 18 |
| mode=independent target=orange_ball | 84 | 2.6252 | 3.8041 | 2.3794 | 4.3461 | 12.8267 | 4 | 10 |
| mode=independent target=red_cube | 84 | 2.7449 | 5.5754 | 2.6121 | 3.6745 | 13.5905 | 21 | 8 |
| mode=sequential_teacher_forced | 800 | 3.0398 | 8.4248 | 2.8873 | 4.1072 | 22.5208 | 160 | 133 |
| mode=sequential_teacher_forced target=orange_ball | 400 | 2.7748 | 6.3249 | 2.5400 | 4.4184 | 19.0055 | 44 | 82 |
| mode=sequential_teacher_forced target=red_cube | 400 | 3.3047 | 9.4136 | 3.2345 | 3.7960 | 27.4261 | 116 | 51 |

## Phase Breakdown

| Group | Count | Mean MAE | P90 MAE | Mean Arm MAE | Mean Grip Abs | P90 Max Abs |
|---|---:|---:|---:|---:|---:|---:|
| mode=independent phase=approach | 48 | 3.6245 | 6.9292 | 3.5623 | 4.0597 | 17.9467 |
| mode=independent phase=close | 24 | 3.6088 | 9.4196 | 2.9771 | 8.0308 | 25.9877 |
| mode=independent phase=descend | 40 | 2.7300 | 4.2089 | 2.5576 | 3.9369 | 15.3684 |
| mode=independent phase=hold | 16 | 1.3358 | 1.8760 | 1.1786 | 2.4360 | 5.4016 |
| mode=independent phase=lift | 40 | 1.4981 | 2.1788 | 1.3919 | 2.2419 | 5.8754 |
| mode=independent target=orange_ball phase=approach | 24 | 3.3182 | 5.4399 | 3.1720 | 4.3413 | 12.7016 |
| mode=independent target=orange_ball phase=close | 12 | 3.7638 | 10.6527 | 3.0050 | 9.0757 | 31.7930 |
| mode=independent target=orange_ball phase=descend | 20 | 2.7358 | 4.2089 | 2.5038 | 4.3598 | 13.0225 |
| mode=independent target=orange_ball phase=hold | 8 | 1.3114 | 1.7073 | 1.0916 | 2.8500 | 5.0525 |
| mode=independent target=orange_ball phase=lift | 20 | 1.5254 | 2.1236 | 1.4435 | 2.0990 | 5.9965 |
| mode=independent target=red_cube phase=approach | 24 | 3.9308 | 7.4689 | 3.9527 | 3.7780 | 20.0128 |
| mode=independent target=red_cube phase=close | 12 | 3.4539 | 7.5247 | 2.9493 | 6.9860 | 16.8389 |
| mode=independent target=red_cube phase=descend | 20 | 2.7242 | 3.7189 | 2.6114 | 3.5140 | 15.9947 |
| mode=independent target=red_cube phase=hold | 8 | 1.3602 | 2.0778 | 1.2656 | 2.0220 | 5.5745 |
| mode=independent target=red_cube phase=lift | 20 | 1.4709 | 2.1929 | 1.3403 | 2.3848 | 5.8502 |
| mode=sequential_teacher_forced phase=approach | 224 | 4.7182 | 10.3918 | 4.7629 | 4.4052 | 29.6465 |
| mode=sequential_teacher_forced phase=close | 128 | 3.3857 | 8.1837 | 2.7304 | 7.9728 | 27.5557 |
| mode=sequential_teacher_forced phase=descend | 192 | 2.9978 | 8.2952 | 2.8365 | 4.1272 | 21.5213 |
| mode=sequential_teacher_forced phase=hold | 64 | 1.2010 | 1.6604 | 1.0945 | 1.9465 | 4.5548 |
| mode=sequential_teacher_forced phase=lift | 192 | 1.5058 | 2.3072 | 1.4520 | 1.8827 | 7.0494 |
| mode=sequential_teacher_forced target=orange_ball phase=approach | 112 | 4.0359 | 9.1871 | 3.9649 | 4.5325 | 23.5711 |
| mode=sequential_teacher_forced target=orange_ball phase=close | 64 | 3.2935 | 8.0104 | 2.7817 | 6.8759 | 26.7282 |
| mode=sequential_teacher_forced target=orange_ball phase=descend | 96 | 2.6277 | 2.9600 | 2.2001 | 5.6212 | 11.5777 |
| mode=sequential_teacher_forced target=orange_ball phase=hold | 32 | 1.1546 | 1.6761 | 0.9602 | 2.5159 | 4.4423 |
| mode=sequential_teacher_forced target=orange_ball phase=lift | 96 | 1.6450 | 2.9565 | 1.5831 | 2.0782 | 8.5144 |
| mode=sequential_teacher_forced target=red_cube phase=approach | 112 | 5.4005 | 11.7895 | 5.5609 | 4.2779 | 33.8366 |
| mode=sequential_teacher_forced target=red_cube phase=close | 64 | 3.4778 | 9.2396 | 2.6790 | 9.0696 | 29.0783 |
| mode=sequential_teacher_forced target=red_cube phase=descend | 96 | 3.3679 | 9.5042 | 3.4728 | 2.6333 | 28.4666 |
| mode=sequential_teacher_forced target=red_cube phase=hold | 32 | 1.2474 | 1.5520 | 1.2288 | 1.3771 | 4.5738 |
| mode=sequential_teacher_forced target=red_cube phase=lift | 96 | 1.3666 | 1.8912 | 1.3209 | 1.6871 | 5.6621 |

## Largest Errors

| Mode | Target | Episode | Frame | Phase | MAE | Max Abs | Joint With Max Error | Teacher | Predicted |
|---|---|---:|---:|---|---:|---:|---|---|---|
| sequential_teacher_forced | red_cube | 837 | 66 | close | 18.2052 | 51.4280 | joint_7 | `[23.775623, -6.464297, -20.004438, 18.196081, 2.885256, -6.577711, -19.603662, -6.875]` | `[9.7063, -6.213087, -20.959051, 61.825966, -1.540014, -31.938225, -71.031624, -1.352261]` |
| independent | orange_ball | 91 | 0 | approach | 17.6024 | 43.0238 | joint_1 | `[8.423736, -7.0, -4.601048, 97.647087, -35.136604, -38.0, -78.0, -65.0]` | `[-34.600021, -12.004976, -1.848611, 117.260071, -42.962975, -31.070951, -37.196915, -50.13369]` |
| sequential_teacher_forced | red_cube | 892 | 62 | close | 17.3937 | 49.6942 | joint_7 | `[22.91172, -6.191181, -19.825438, 20.266966, 2.200251, -8.465029, -22.598932, -22.375]` | `[13.475752, -7.630374, -19.847342, 58.132156, -1.367541, -28.690329, -72.293159, -5.474834]` |
| sequential_teacher_forced | red_cube | 798 | 1 | approach | 16.1912 | 41.0579 | joint_1 | `[36.747913, -7.0, -16.243477, 48.924713, -12.091105, -38.0, -78.0, -65.0]` | `[-4.310026, -4.50485, -12.414434, 85.086716, -19.121656, -40.901436, -47.349724, -59.597122]` |
| sequential_teacher_forced | red_cube | 798 | 0 | approach | 16.1180 | 42.4694 | joint_1 | `[22.864182, -7.0, -15.703434, 80.178513, -12.832621, -38.0, -78.0, -65.0]` | `[-19.605194, -1.274953, -11.977404, 109.685837, -22.166126, -40.159637, -42.096031, -64.880531]` |
| sequential_teacher_forced | red_cube | 892 | 31 | descend | 15.6794 | 45.9025 | joint_7 | `[20.434652, -7.0, -20.061863, 26.631292, 2.485938, -14.044247, -30.592524, -65.0]` | `[30.247286, -8.827308, -20.479855, 53.23476, -14.365492, -36.567917, -76.495041, -66.495903]` |
| sequential_teacher_forced | red_cube | 798 | 27 | approach | 15.4518 | 44.3682 | joint_7 | `[19.669548, -7.0, -21.749176, 60.794754, -14.737829, -36.465012, -78.0, -65.0]` | `[21.51425, -6.780219, -19.971985, 24.959471, -1.334305, -11.834454, -33.631828, -63.464928]` |
| sequential_teacher_forced | orange_ball | 91 | 1 | approach | 15.0972 | 42.2250 | joint_1 | `[14.971668, -7.0, -5.073699, 71.657463, -38.196907, -38.0, -78.0, -65.0]` | `[-27.253355, -6.745552, -4.706555, 93.46431, -46.518349, -40.117378, -39.430622, -57.883865]` |
| independent | red_cube | 837 | 25 | approach | 15.0842 | 46.7484 | joint_7 | `[27.431046, -7.0, -20.049545, 51.262226, -10.772692, -35.781227, -78.0, -65.0]` | `[26.326965, -5.2604, -18.571255, 22.314899, -0.467201, -11.579047, -31.251627, -71.148224]` |
| sequential_teacher_forced | red_cube | 892 | 1 | approach | 15.0618 | 43.3952 | joint_4 | `[39.779758, -7.0, -16.499161, 44.705097, -10.638793, -38.0, -78.0, -65.0]` | `[-1.565508, -4.362894, -13.416025, 88.100334, -16.10704, -43.159073, -64.23922, -70.645615]` |
| independent | red_cube | 892 | 30 | descend | 14.9584 | 45.1126 | joint_7 | `[19.867777, -7.0, -20.137363, 27.472178, 2.812771, -16.466921, -31.78743, -65.0]` | `[26.716413, -6.784261, -21.473398, 55.570023, -11.053451, -33.077595, -76.900047, -72.579666]` |
| sequential_teacher_forced | red_cube | 837 | 27 | approach | 14.8384 | 49.3581 | joint_7 | `[26.973612, -7.0, -20.308184, 51.940716, -10.57778, -36.012302, -78.0, -65.0]` | `[22.592405, -7.070637, -18.709953, 24.893974, 2.531527, -13.031885, -28.641888, -64.837639]` |

