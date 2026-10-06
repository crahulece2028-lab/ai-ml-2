# Task 3.1 summary — TensorFlow / Keras model

| Metric | Value |
|---|---|
| Framework | TensorFlow 2.21.0 / Keras 3.15.1 |
| Architecture | 2 x (Conv3x3 + BatchNorm + ReLU + MaxPool) -> Flatten -> Dropout -> Dense128 -> Dense10 |
| Parameters | 421,738 (PyTorch Week 2: 421,738) |
| Architectures identical | True |
| Optimizer | Adam 0.001, batch 64, 8 epochs, no schedule, no early stopping |
| Best validation accuracy | 0.9213 (epoch 8) |
| Test accuracy | 0.9140 |
| PyTorch Week 2 test accuracy | 0.9067 |
| Test difference | +0.73 points |
| Same-recipe best-val gap | 0.27 points |
| Worst class | Shirt (0.7270) |
| Best class | Bag (0.9810) |
