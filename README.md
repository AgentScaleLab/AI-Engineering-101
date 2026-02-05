# AI Engineering 101
All you need to know to get started with AI Engineering. In this repository, I'm documenting my learning notes of AI Engineering.

## Table of Contents
- [AI Engineering 101](#ai-engineering-101)
  - [Table of Contents](#table-of-contents)
  - [Research](#research)
    - [Agent Infra](#agent-infra)
    - [Multimodal Infra](#multimodal-infra)
      - [Training](#training)
      - [Serving](#serving)
      - [GPU Kernels](#gpu-kernels)
  - [Engineering](#engineering)
    - [PyTorch Best Practices](#pytorch-best-practices)
    - [Multimodal Best Practices](#multimodal-best-practices)

## Research
_Research advances on agent infra and multimodal infra_

### Agent Infra
> Agent infra focuses on optimizing agent runtime performance instead of building agents. For more information, check out [Why agent infrastructure matters](https://blog.langchain.com/why-agent-infrastructure/) and [Agent Engineering: A New Discipline](https://blog.langchain.com/agent-engineering-a-new-discipline/).

1. [Identifying the Risks of LM Agents with an LM-Emulated Sandbox](https://arxiv.org/abs/2309.15817). ICLR 2024.
2. [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965). arXiv 2025.

### Multimodal Infra
> If you want to learn more about multimodal models, check out the [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms?utm_source=publication-search).

#### Training
1. [DistTrain: Addressing Model and Data Heterogeneity with Disaggregated Training for Multimodal Large Language Models](https://arxiv.org/abs/2408.04275). arXiv 2024.
2. [DISTMM: Accelerating Distributed Multimodal Model Training](https://www.usenix.org/system/files/nsdi24-huang.pdf). NSDI 2024.
3. [PipeWeaver: Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline](https://arxiv.org/abs/2504.14145). arXiv 2025.
#### Serving
1. [Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models](https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham). NSDI 2024.
2. [Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters](https://www.usenix.org/system/files/atc25-li-suyi-katz.pdf). ATC 2025.
3. [Understanding Diffusion Model Serving in Production: A Top-Down Analysis of Workload, Scheduling, and Resource Efficiency](https://dl.acm.org/doi/10.1145/3772052.3772206). SoCC 2025.
#### GPU Kernels
1. [Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](https://arxiv.org/abs/2510.08431). arXiv 2025. 
   - FlashAttention-2 JVP kernel for training
2. [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367). ICLR 2025.
   - 8-bit attention kernel for inference
3. [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://arxiv.org/abs/2509.24006). arXiv 2025.
   - Sparse attention kernel for inference

## Engineering
_Engineering best practices for building AI systems_

### PyTorch Best Practices
1. [Optimize Training Performance in PyTorch](./Engineering/PyTorch/training_optim.md) 
   - MFU, Performance Profiling, and Optimization Techniques
 
### Multimodal Best Practices

1. [Disaggregated Hybrid Parallelism with Ray](https://github.com/ray-project/multimodal-training) - A framework for training vision-language models using disaggregated hybrid parallelism, where each model component adopts its optimal parallelization strategy independently.

