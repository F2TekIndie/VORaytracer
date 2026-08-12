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
- OptiX-Materialtabellen pro Dreieck mit baryzentrisch interpolierten, korrekt transformierten Vertexnormalen
- sichtbare OptiX-Ausgabe im Vulkan-Swapchain-Bild mit ImGui-Overlay
- UI-Schalter zwischen Vulkan und OptiX sowie Kamera-/PBR-/Raytracing-Parameter
- CPU-/Asset-Tests für Mat4, Scene-Statistiken, meshoptimizer und echten Assimp-Import

Der aktuelle vertikale Schnitt führt alle geladenen Mesh-Instanzen mit ihren Node-Transformationen zu einer GPU-Szene zusammen. Vulkan rendert deren Basis-LODs samt Materialdaten als Meshlets und baut dafür BLAS/TLAS. OptiX baut aus derselben vollständigen Geometrie ein GAS und führt echte Primärstrahlen mit direkter PBR-Beleuchtung aus. Die OptiX-Ausgabe wird zurzeit zur einfachen Diagnose über CUDA→CPU→Vulkan kopiert.

Der aktuelle Ausbau verwendet separate Vulkan-BLAS/TLAS-Instanzen, device-local Geometrie, Reflexions-/GI-Ray-Queries
sowie Vulkan/CUDA External-Memory- und Semaphore-Interop. Der OptiX-AI-Denoiser steht als optionaler
Vulkan-Postrender-Schritt zur Verfügung.

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

Das Skript baut `VORaytracer.sln`. Assimp wird als globale Debug-/Release-Bibliothek aus
`G:\CodingLibraries\assimp\out-v145` eingebunden; ein projektspezifischer Assimp-Build ist nicht erforderlich.

## Starten

```powershell
.\bin\x64\Debug\VORaytracer.App.exe
```

Im Scene-Fenster kann ein von Assimp unterstütztes Modell über seinen Dateipfad geladen werden. Ohne Datei startet die Anwendung mit einem prozeduralen Würfel, der durch meshoptimizer verarbeitet und als Meshlets gerendert wird.

Nach einem erfolgreichen Dateiimport wird die Kamera automatisch auf die Welt-Bounds aller geladenen Instanzen ausgerichtet. Die Navigation funktioniert außerhalb der ImGui-Fenster mit linker Maustaste zum Orbitieren, mittlerer Maustaste zum Verschieben und dem Mausrad zum Zoomen. Über „Frame model“ im Renderer-Fenster lässt sich die automatische Einpassung erneut ausführen.

Im Scene-Fenster schaltet der Button „meshoptimizer: ON/OFF“ Remapping, Cache-/Overdraw-/Vertex-Fetch-Optimierung, Meshlet-Bounds und LOD-Simplifizierung gemeinsam ein oder aus. Beim Umschalten wird das aktuelle Modell neu geladen. Nur die für die Meshshader-Ausgabe zwingende Meshlet-Partitionierung bleibt immer aktiv.

Der Button „Default plastic: ON/OFF“ ersetzt beim Laden sämtliche importierten Materialien und Texturreferenzen durch ein gemeinsames hellgraues Kunststoffmaterial (`Albedo 0.75`, `Metallic 0.0`, `Roughness 0.5`) und weist es jedem Mesh zu. Auch dieser Schalter lädt die aktuelle Szene unmittelbar neu.

Für einen automatisierten Start mit vorgewähltem OptiX-Backend kann vor dem Aufruf `VOR_BACKEND=optix` als Umgebungsvariable gesetzt werden. Im normalen Betrieb erfolgt der Wechsel über die Radio-Buttons im ImGui-Fenster „Renderer“.

Der OptiX-AI-Denoiser kann im Renderer-Fenster ausschließlich für das Vulkan-Backend als Postrender-Schritt
aktiviert werden. Für automatisierte Smoke-Tests lässt er sich mit `VOR_DENOISER=1` beim Start einschalten.
Vulkan rendert dafür lineares `RGBA32F` in ein Offscreen-Target; Übergabe, Denoising, Tone-Mapping und Präsentation
bleiben vollständig auf der GPU. Der OptiX-Pathtracer wird immer ohne Denoising ausgegeben.

„Ray-traced reflections“ aktiviert im Vulkan-Backend eine Inline-Ray-Query-Reflexion pro sichtbarem Fragment inklusive Material- und Schattenauswertung am Treffer. Im OptiX-Backend werden raue, Fresnel-gewichtete PBR-Reflexionspfade verfolgt; „Max bounces“ begrenzt dort die Pfadtiefe.

„Meshlet debug colors“ ersetzt im Vulkan-Backend die Materialausgabe durch eine stabile, pseudozufällige Farbe pro Meshlet. Dadurch werden Meshlet-Grenzen und die von `meshoptimizer` erzeugte Partitionierung direkt sichtbar. Für automatisierte Starts kann der Modus mit `VOR_MESHLET_DEBUG=1` aktiviert werden.

Für reproduzierbare Import-Smoke-Tests kann ein Startmodell über `VOR_SCENE=<absoluter Dateipfad>` vorgegeben werden.

## Tests

```powershell
.\bin\x64\Debug\VORaytracer.Tests.exe
```

Debug-Builds aktivieren den Vulkan Validation Layer. Diagnoseausgaben erscheinen auf `stderr`.

Der Slang-Hinweis `E38040` beim OptiX-Build ist erwartet: Der Raygen-Parameter wird absichtlich als Uniform im SBT-Raygen-Record abgelegt.
