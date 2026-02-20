# Optimize Training Performance in PyTorch

This tutorial shows how to optimize training performance in PyTorch.

`MFU` (Model FLOPs Utilization) is a key metric to measure how efficiently a model utilizes the available computational resources during training. Higher `MFU` indicates better utilization of the hardware, leading to faster training times and improved performance.

$$
MFU = \frac{FLOPs_{actual}}{FLOPs_{max}}
$$

In general, `MFU > 0.5` is considered good, while `MFU < 0.3` indicates that there is significant room for optimization.

In the following examples, we will use a ResNet-18 model as an example.