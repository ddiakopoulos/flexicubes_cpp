# FlexiCubes C++

This repository is a C++17 header-only implementation of NVIDIA's FlexiCubes, a differentiable isosurface extraction method designed for gradient-based mesh optimization. The method represents a surface as an isosurface of a scalar field, introduces learnable parameters to flexibly adjust geometry and connectivity, and builds on Dual Marching Cubes for improved topology. It can optionally produce tetrahedral meshes and exposes gradients for optimization workflows. 

It is based on the reference NVIDIA's Python + PyTorch implementation (`flexicubes.py`). Versus Python, the main audience for this implementation is procedural pipeline folks (the SDF/modeling crowd) who may want to tinker with high-fidelity surfaces (with optional surface optimization) in-engine, rather than an isolated research environment. 

For optimization, this library exposes a `grad_func` for analytic surface gradients when using QEF. See the `optimization_loop.cpp` example. 

## What it can also be used for
- Gradient-based mesh optimization from signed distance fields (SDFs).
- Reconstruction and refinement pipelines in photogrammetry, generative modeling, and inverse physics workflows.
- Producing triangle or tetrahedral meshes from regular voxel grids.
- Research prototypes that need differentiable surface extraction (e.g., learning SDFs with geometry-aware losses, although you're probably just going to want the original Python implementation for that). 

## Dependencies
- Eigen 
- Adept-2

## Reference
- Tianchang Shen et al., "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization," ACM Trans. Graph. (SIGGRAPH 2023), 42(4), Article 37. arXiv:2308.05371.
- [NVIDIA Reference Implementation](https://github.com/nv-tlabs/FlexiCubes)

## License

Licensed under the Apache License 2.0 (same as the NVIDIA reference implementation).
