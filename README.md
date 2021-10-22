# Few Shot Temporal Action Localization using Query Adaptive Transformer

**Accepted as Poster in BMVC 2021**

This is an official implementation in PyTorch of FS-QAT. Our paper is available at [Arxiv](https://arxiv.org/abs/2110.10552)

![](img/fig1.png)

## Updates
- (October, 2021) We released FS-QAT training and inference code for ActivityNet dataset.
- (October, 2021) FS-QAT is accepted in BMVC2021.

## Abstract
Existing temporal action localization (TAL) works rely on a large number of training videos with exhaustive segment-level annotation, preventing them from scaling to new classes. As a solution to this problem, few-shot TAL (FS-TAL) aims to adapt a model to a new class represented by as few as a single video. Exiting FS-TAL methods assume trimmed training videos for new classes. However, this setting is not only unnatural – actions are typically captured in untrimmed videos, but also ignores background video segments containing vital contextual cues for foreground action segmentation. In this work, we first propose a new FS-TAL setting by proposing to use untrimmed training videos. Further, a novel FS-TAL model is proposed which maximizes the knowledge transfer from training classes whilst enabling the model to be dynamically adapted to both the new class and each video of that class simultaneously. This is achieved by introducing a query adaptive Transformer in the model. Extensive experiments on two action localization benchmarks demonstrate that our method can outperform all the stateof-the-art alternatives significantly in both single-domain and cross-domain scenarios.

## Summary
- First Few-Shot TAL setting to use Untrimmed Videos for both Support and Query 
- Unified Model can accomodate both Untrimmed and Trimmed Video without design change
- Instead of meta-learning the entire network, only Transformer is meta-learned hence faster adaptation.
- Intra-Class Variance is handled using this adaptation
- Promising performance in Cross-Domain/Dataset settings.

## Qualitative Performance

![](img/fig2.png)

## Training and Evaluation

Appologize for the messed up Code

Refactoring will be done soon ( delay due to CVPR workload )

> To Train

```train
python gtad_train_fs.py 
```

> To Test

```test
sh test_fs.sh
```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@misc{nag2021fewshot,
      title={Few-Shot Temporal Action Localization with Query Adaptive Transformer}, 
      author={Sauradip Nag and Xiatian Zhu and Tao Xiang},
      year={2021},
      eprint={2110.10552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
