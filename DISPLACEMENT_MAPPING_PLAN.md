# Displacement Mapping Integration Plan

Status: Planned  
Target: VORaytracer, C++20, Visual Studio, Vulkan mesh-shader pipeline, OptiX 9.1, Slang, ImGui, Assimp, stb and meshoptimizer

## Objective

Implement true geometric displacement whose visible silhouette, Vulkan ray-query shadows and reflections, and OptiX intersections all use the same displaced geometry.

The correctness-first implementation will generate one crack-free displaced render mesh from immutable imported geometry. That mesh will then be consumed by:

- Vulkan task/mesh-shader rasterization
- Vulkan BLAS/TLAS ray queries
- OptiX GAS/IAS traversal

Displacement must not be applied only in the Vulkan mesh shader. Shader-only displacement would leave the Vulkan acceleration structures and OptiX geometry undisplaced, causing incorrect shadows, reflections, occlusion and intersections.

## Technology decision

The initial implementation will use explicitly tessellated triangle geometry shared by both render paths.

`VK_NV_displacement_micromap` will not be the primary implementation because it is provisional, not ratified, and deprecated in favor of `VK_NV_cluster_acceleration_structure`. The installed OptiX 9.1 SDK exposes cluster acceleration structures rather than the older displaced-micromesh API. A cluster-based NVIDIA-specific path can be evaluated after the portable path is correct and measured.

References:

- Vulkan displacement micromap extension: https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_displacement_micromap.html
- OptiX 9.1 Programming Guide: https://raytracing-docs.nvidia.com/optix9/guide/index.html
- NVIDIA RTX Mega Geometry: https://github.com/NVIDIA-RTX/RTXMG

## Milestone 1 — Material and texture representation

Add CPU-side displacement settings to the material representation:

```cpp
struct DisplacementSettings
{
    std::int32_t texture{-1};
    float scale{0.0f};
    float midpoint{0.5f};
    std::uint32_t subdivisionLevel{0};
    bool enabled{false};
};
```

Semantics:

```text
offset = (height - midpoint) * scale
position = basePosition + displacementDirection * offset
```

Tasks:

1. Import `aiTextureType_DISPLACEMENT` independently from normal and height/bump textures.
2. Allow the existing height texture to be selected as a displacement source when no dedicated displacement texture exists.
3. Keep bump mapping and geometric displacement as separate features.
4. Interpret displacement textures in linear space without sRGB conversion.
5. Add 16-bit and floating-point scalar image decoding to prevent visible displacement banding.
6. Preserve mip level zero scalar data for mesh generation and build linear scalar mipmaps when needed.
7. Define object/model-space units for `scale`; negative values must be supported.

The base implementation should use the red channel as scalar height. Grayscale RGB maps remain compatible, while single-channel formats can be supported without luminance conversion errors.

## Milestone 2 — Immutable source geometry

Separate imported source topology from generated render topology.

The source mesh must retain its original:

- positions
- normals
- tangents
- UVs
- indices
- material assignment

The displacement processor will create a derived render mesh. Repeated changes to scale, midpoint or quality must always regenerate from the immutable source and never accumulate displacement.

Recommended ownership:

- `SceneSource` or equivalent owns immutable imported meshes and decoded scalar textures.
- `Scene` owns the active generated render meshes used by both backends.
- Instances continue to reference meshes by index so one displaced geometry can still be instanced by multiple TLAS/IAS instances.

Changing displacement parameters must not re-read the model file or reload textures.

## Milestone 3 — Crack-free tessellation and displacement

Create a focused `DisplacementProcessor` module that converts one source mesh into one render mesh.

Initial algorithm:

1. Subdivide every source triangle uniformly.
2. Subdivision level `N` produces `4^N` triangles per source triangle.
3. Generate shared-edge vertices through a canonical edge cache.
4. Use the source vertex indices and canonical edge direction as the cache key.
5. Interpolate position, normal, tangent and UV barycentrically.
6. Bilinearly sample the scalar displacement texture using the same UV wrapping convention as rendering.
7. Displace along the normalized interpolated object-space normal.
8. Recalculate area-weighted geometric/shading normals from the displaced mesh.
9. Recalculate tangents and handedness from displaced positions and UVs.
10. Reject or repair degenerate triangles and any NaN/Inf values.

Uniform subdivision is the first milestone because it guarantees matching tessellation along shared edges. Adaptive tessellation will only be introduced after neighboring triangles propagate and agree on each shared edge subdivision level.

UV seams and intentional hard edges should remain split at the attribute level, but corresponding boundary positions must remain coincident so no geometric crack appears.

Add a configurable maximum generated triangle and memory budget. The processor must fail with an actionable diagnostic instead of exhausting system or GPU memory.

## Milestone 4 — meshoptimizer and derived geometry data

Run the existing mesh processing after displacement generation.

Mandatory operations:

- meshlet partitioning required by the Vulkan mesh-shader path
- rebuilding meshlet vertex and triangle streams
- recomputing displaced meshlet bounding spheres
- recomputing displaced normal cones

Optional operations controlled by the existing meshoptimizer UI toggle:

- vertex remapping
- vertex-cache optimization
- overdraw optimization
- vertex-fetch optimization
- simplified LOD generation

LOD geometry must be generated from the displaced base geometry. Bounds and culling data from the undisplaced source mesh must never be reused.

## Milestone 5 — Vulkan integration

Upload the generated displaced mesh through the existing device-local staging path.

Requirements:

1. The mesh shader reads already-displaced positions, normals, tangents and UVs.
2. The Vulkan BLAS uses the exact same position and index data.
3. Vulkan ray-query shadows and reflections therefore intersect the visible displaced silhouette.
4. Task-shader frustum and normal-cone culling uses displaced meshlet bounds.
5. A subdivision-level change rebuilds the affected BLAS because topology changed.
6. A scale or midpoint change with unchanged topology may use an acceleration-structure update/refit after resources are created with the appropriate update flag.
7. Unchanged, non-displaced BLAS resources should be reused.
8. The TLAS should only be updated after affected BLAS handles or bounds change.

No second geometric displacement will be applied in `MeshMain`.

## Milestone 6 — OptiX integration

Upload the same generated render mesh to OptiX.

Requirements:

1. Build one GAS per generated mesh and preserve existing IAS instancing.
2. Upload displaced position, normal, tangent, UV and index buffers together.
3. Rebuild the GAS when subdivision changes topology.
4. For position-only changes, create compatible GAS resources with `OPTIX_BUILD_FLAG_ALLOW_UPDATE` and evaluate `OPTIX_BUILD_OPERATION_UPDATE`.
5. Recalculate emissive triangle areas, power and the mesh-light CDF because displacement changes surface area.
6. Keep material IDs and instance IDs stable across rebuilds where possible.
7. Ensure OptiX hit interpolation, geometric normals and shading normals use the displaced buffers.

Vulkan and OptiX must report identical generated vertex and triangle counts for the same scene and displacement settings.

## Milestone 7 — UI and rebuild lifecycle

Add per-material ImGui controls:

- `Displacement enabled`
- displacement texture assignment
- `Scale`
- `Midpoint`
- `Subdivision level`
- maximum triangle budget
- estimated generated vertex/triangle count
- residual bump mapping toggle

Recommended user-facing modes:

- Bump only
- Displacement only
- Displacement plus normal map
- Displacement plus residual bump

When the same height texture drives both displacement and bump, bump contribution should default to disabled to avoid applying the same variation twice. A later residual-bump mode can preserve texture frequencies that are too fine for the selected displacement tessellation.

Rebuild behavior:

1. Changing shading-only material factors continues to update only the material buffer.
2. Changing displacement enablement, texture, scale, midpoint or subdivision schedules a geometry rebuild for affected meshes.
3. CPU mesh generation runs on a cancellable `std::jthread` with a generation ID.
4. Only the latest completed generation may replace the active scene.
5. CPU scene data and GPU resources are swapped at a frame boundary.
6. Vulkan fences/semaphores and CUDA events protect resource lifetime.
7. Do not use `vkDeviceWaitIdle()`, `vkQueueWaitIdle()`, `cuCtxSynchronize()` or full-frame CPU/GPU synchronization for routine displacement changes.
8. Display rebuild progress, generated triangle count, elapsed time and errors in the UI.

The default-plastic material override remains shading-only. It must not remove or regenerate displaced geometry.

## Milestone 8 — Performance controls

Implement the following safeguards before enabling high subdivision levels:

- per-mesh and whole-scene triangle budgets
- overflow-checked vertex/index count calculations
- derived-mesh cache keyed by source mesh, texture revision, scale, midpoint and subdivision settings
- reuse of unchanged GPU buffers and BLAS/GAS resources
- asynchronous preprocessing and upload
- timing statistics for tessellation, meshoptimizer processing, upload and acceleration-structure build
- memory statistics for source geometry, derived geometry and acceleration structures

Initial displacement quality should be fixed at scene/material level. Camera-distance adaptive tessellation is a later milestone because raster geometry and acceleration-structure geometry must switch consistently.

## Milestone 9 — Optional cluster acceleration path

After the explicit-triangle implementation is correct and profiled, evaluate an NVIDIA-specific adaptive path based on:

- `VK_NV_cluster_acceleration_structure` for Vulkan
- OptiX 9.1 cluster templates and cluster GAS
- structured grid templates for adaptive tessellation
- CUDA/Vulkan external-memory interop for shared generated vertex data where beneficial
- the NVIDIA RTX Mega Geometry reference implementation

This path is optional and must retain the explicit-triangle implementation as a fallback.

`RTXMG` is not currently present in `G:\CodingLibraries`. If this optional milestone is selected for implementation, the library must first be reviewed for license/build compatibility and added to the library directory with user approval.

## Testing plan

### CPU tests

- A constant midpoint height map leaves geometry unchanged.
- A constant non-midpoint map produces the expected uniform offset.
- A height ramp produces analytically expected displaced positions.
- Positive and negative scale work correctly.
- Midpoint values at 0, 0.5 and 1 work correctly.
- Missing or invalid displacement textures fail safely.
- Shared edges generate identical positions and contain no cracks.
- UV seams retain separate attributes while boundary positions coincide.
- Degenerate UVs and triangles do not produce NaN/Inf values.
- Recalculated normals and tangents are finite and normalized.
- Displaced bounds contain every generated vertex.
- meshoptimizer on/off preserves equivalent displaced geometry.
- Triangle-budget overflow is detected before allocation.
- FlightHelmet remains unchanged when displacement is disabled or absent.

### Vulkan tests

- Validation produces no VUID errors.
- Mesh-shader silhouettes use the displaced geometry.
- Ray-query shadows and reflections match the displaced silhouette.
- Meshlet frustum and normal-cone culling create no holes.
- Position-only updates/refits and topology rebuilds both complete without device-wide waits.

### OptiX tests

- OptiX validation reports no traversal or build errors.
- Primary, shadow and reflection rays hit the displaced geometry.
- GAS updates are used only when topology remains compatible.
- Emissive displaced meshes use recalculated areas and light PDFs.

### Cross-backend tests

- Vulkan and OptiX use identical displaced vertex and triangle counts.
- Known test rays hit equivalent positions and normals.
- Comparison renders have matching silhouettes, shadows and material response.
- Debug and Release configurations build successfully.
- Existing bump, normal-map, HDR, material-override, ground-plane, FBX and glTF regressions remain passing.

## Completion criteria

The displacement feature is complete when:

1. Displacement visibly changes the model silhouette.
2. Vulkan rasterization, Vulkan ray queries and OptiX trace the same displaced surface.
3. Shared triangle edges show no cracks.
4. Normal maps continue to work on top of displaced geometry.
5. Displacement settings rebuild geometry without reloading the source file.
6. Rebuilds do not introduce device-wide synchronization in the frame loop.
7. Meshlet bounds, BLAS/GAS bounds and camera framing include the displaced surface.
8. Debug/Release builds, CPU tests and Vulkan/OptiX GPU smoke tests pass.

