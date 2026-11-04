# Prepared literature-sized dataset; quantum study not run

MNIST [3, 8]: 800 fit-training + 200 validation + 200 test images per seed.
Source: https://arxiv.org/html/2407.02553v1#S1.SS1.SSS1

The validation images are inside the training-pool count. Both algorithms use the same deterministic splits and training-only PCA.
Frozen prior-test exclusions: 2303; manifest SHA-256: fd5ef9a88426f75072151b70c56dda6aa83be054f9a4e3e0152c865a8fac1a72.
These files contain dataset preparation and planned resource bounds. They contain no new QRC/VQC accuracies or measured QPU energy.

Planned QRC fit/validation shots per seed: 1,400,000.
VQC fit/validation/objective shots per seed at the safety cap: 24,824,000.
Plateau stopping can consume less; every actual request will be accounted for by a completed run.

Repeated minibatch draws; not guaranteed dataset coverage. Linear-head checkpoints use all fit-training images. Unchanged optimizer safety cap is not proof of sufficient training on the larger dataset.
