# VORaytracer

VORaytracer ist ein C++20-/Visual-Studio-Projekt für einen hybriden Vulkan-PBR-Renderer mit meshoptimizer-Meshlets, Slang-Shadern und einem optionalen NVIDIA-OptiX-Backend.

Der ausführliche technische Plan steht in [IMPLEMENTIERUNGSPLAN.md](IMPLEMENTIERUNGSPLAN.md).

## Aktueller Funktionsstand

- Native Visual-Studio-Solution mit x64 Debug und Release
- GLFW-Fenster und Dear ImGui UI
- Vulkan 1.3 mit Dynamic Rendering und Validation
- `VK_EXT_mesh_shader` mit `vkCmdDrawMeshTasksEXT`
- Assimp-Szenenimport und PBR-Materialmetadaten
- meshoptimizer 1.2 für Remap, Cache, Overdraw, Vertex Fetch, LODs, Meshlets und Bounds
- echte GPU-Storage-Buffer für Vertices, Meshlets, lokale Vertex-Indizes und gepackte Primitive
- Vulkan-BLAS/TLAS aus der geladenen Geometrie und aktive Ray Queries für PBR-Schatten im Fragmentshader
- Slang-zu-SPIR-V-Build für Mesh-, PBR-/Ray-Query-Fragment-, Compute-Ray-Query- und Tone-Mapping-Shader
- Slang-zu-PTX-Build mit OptiX-Raygen-, Miss- und Closest-Hit-Programmen
- OptiX-Triangle-GAS, SBT, progressiver Multi-Sample-Launch und gemeinsamer Metallic-Roughness-BRDF
- sichtbare OptiX-Ausgabe im Vulkan-Swapchain-Bild mit ImGui-Overlay
- UI-Schalter zwischen Vulkan und OptiX sowie Kamera-/PBR-/Raytracing-Parameter
- CPU-/Asset-Tests für Mat4, Scene-Statistiken, meshoptimizer und echten Assimp-Import

Der aktuelle vertikale Schnitt rendert im Vulkan-Backend das erste Mesh/LOD und baut dafür BLAS/TLAS. Das OptiX-Backend baut für dieselbe Geometrie ein GAS und führt echte Primärstrahlen mit direkter PBR-Beleuchtung aus. Die OptiX-Ausgabe wird zurzeit zur einfachen Diagnose über CUDA→CPU→Vulkan kopiert.

Noch nicht als Produktionsausbau umgesetzt sind Multi-Mesh-IAS/TLAS-Instanzierung, GPU-Textur-Uploads, rekursive OptiX-Bounces, Reflexions-/GI-Ray-Queries, Denoising sowie Vulkan/CUDA External-Memory- und Semaphore-Interop. Diese Punkte bleiben im Implementierungsplan als nächste Ausbaustufen erhalten.

Der installierte Slang-Compiler 2026.14.1 kompiliert Mesh-Shader korrekt, hängt aber reproduzierbar bei einem Amplification-/Task-Shader mit `out payload`. Deshalb verwendet der aktuelle lauffähige Pfad eine zulässige Mesh-Shader-Pipeline ohne die optionale Task-Stufe. Die isolierte Task-Shader-Quelle liegt unter `shaders/Vulkan/MeshletTask.slang`.

## Voraussetzungen

- Visual Studio 2026 mit C++-Toolset `v145`
- Windows SDK 10.0.26100.0 oder kompatibel
- Bibliotheken unter `G:\CodingLibraries`
- CUDA 13.3 unter `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`
- OptiX 9.1.0 unter `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0`
- NVIDIA-GPU mit `VK_EXT_mesh_shader`
- Vulkan-Unterstützung für `VK_KHR_acceleration_structure` und `VK_KHR_ray_query`

## Bauen

In PowerShell:

```powershell
.\scripts\Build.ps1 -Configuration Debug
.\scripts\Build.ps1 -Configuration Release
```

Das Skript baut Assimp aus `G:\CodingLibraries\assimp` in `build\assimp` und anschließend `VORaytracer.sln`.

Alternativ kann die Solution nach dem ersten Assimp-Build direkt in Visual Studio geöffnet werden.

## Starten

```powershell
.\bin\x64\Debug\VORaytracer.App.exe
```

Im Scene-Fenster kann ein von Assimp unterstütztes Modell über seinen Dateipfad geladen werden. Ohne Datei startet die Anwendung mit einem prozeduralen Würfel, der durch meshoptimizer verarbeitet und als Meshlets gerendert wird.

Für einen automatisierten Start mit vorgewähltem OptiX-Backend kann vor dem Aufruf `VOR_BACKEND=optix` als Umgebungsvariable gesetzt werden. Im normalen Betrieb erfolgt der Wechsel über die Radio-Buttons im ImGui-Fenster „Renderer“.

## Tests

```powershell
.\bin\x64\Debug\VORaytracer.Tests.exe
```

Debug-Builds aktivieren den Vulkan Validation Layer. Diagnoseausgaben erscheinen auf `stderr`.

Der Slang-Hinweis `E38040` beim OptiX-Build ist erwartet: Der Raygen-Parameter wird absichtlich als Uniform im SBT-Raygen-Record abgelegt.
