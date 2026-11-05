# Locked analysis (fixed before any ten-crop MACE outcome is seen)

Primary memory selection: release seed 0, K=4 images per disease, four retrieved captions. Never headline another seed, K or crop.

Primary contrasts (equal-crop mean accuracy, IID and held-out separately):
1. MACE retrieval minus SAGE.
2. MACE retrieval minus DINOv2 1-NN (same bank).
3. Labels-only minus MACE retrieval (ablation panel).
Secondary: macro disease recall; OpenCLIP 1-NN, majority and score-weighted votes; K=1/16; masked, shuffled, random; memory-selection SD; hit/miss (associative, oracle); agentic and image references (five crops only, all-attempted and common-completed accuracy).

Uncertainty: paired 95% bootstrap over unique query images, stratified by crop, 2,000 resamples, seed 0; paired methods and the three memory selections resampled together. SD across selections is selection sensitivity, not an image-sampling interval. No McNemar on concatenated repeated draws. Holm across the three primary contrasts x two splits; effect sizes emphasized.

Reporting: soybean pilot stays a separate PILOT / DIFFERENT PROTOCOL panel. No cell is filled without a matching protocol (model, prompts, KB, candidates, memory, retrieval).
