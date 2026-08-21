# Third-party components

VORaytracer uses the following external SDKs and libraries. They are not vendored in this repository and remain subject to their own licenses and redistribution terms:

- Khronos Vulkan SDK
- Slang
- Assimp (Open Asset Import Library)
- GLFW
- Dear ImGui
- meshoptimizer
- stb / stb_image
- NVIDIA CUDA Toolkit
- NVIDIA OptiX SDK

The build expects locally installed copies as described in `README.md`. Before distributing binaries, review the exact versions in use and include all notices, licenses, and redistributable runtime files required by their respective upstream packages. NVIDIA SDK components may impose additional redistribution conditions independent of the project's LGPL license.
