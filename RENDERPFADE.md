# Renderpfade

Diese Übersicht zeigt, welche Daten beide Backends teilen und wo sich Vulkan und OptiX unterscheiden. Die Diagramme verwenden Mermaid und werden in GitHub, Visual Studio Code und anderen Mermaid-fähigen Markdown-Ansichten direkt gerendert.

## Gemeinsame Asset-Aufbereitung

```mermaid
flowchart LR
    Model["Modell über IFileOpenDialog"] --> Assimp["Assimp-Import"]
    Assimp --> Scene["Gemeinsame CPU-Szene<br/>Meshes, Instanzen, Materialien"]
    Scene --> Meshopt["meshoptimizer<br/>Remap, Cache, Meshlets, Bounds, LODs"]
    HDR["Radiance HDR über IFileOpenDialog"] --> STB["stb_image<br/>Float-Decodierung"]
    STB --> Mips["HDR-Mip-Pyramide<br/>Diffuse und spiegelnde IBL"]
    Meshopt --> VulkanData["Vulkan-Ressourcen"]
    Meshopt --> OptixData["CUDA-/OptiX-Ressourcen"]
    Mips --> VulkanData
    Mips --> OptixData
```

Die optionale meshoptimizer-Einstellung schaltet zusätzliche Remap-, Cache-, Overdraw-, Vertex-Fetch-, Bounds- und LOD-Verarbeitung. Die Meshlet-Partitionierung bleibt für den Vulkan-Mesh-Shader-Pfad immer aktiv.

## Vulkan: Mesh-Shader und Ray Queries

```mermaid
flowchart TD
    Scene["Gemeinsame Szene und Kamera"] --> Upload["Staging-Upload in device-local Buffer"]
    Upload --> Geometry["Vertex-, Index-, Meshlet- und Materialbuffer"]
    Geometry --> Task["Task-/Amplification-Shader<br/>32 Meshlets pro Workgroup"]
    Task --> Cull{"Frustum- und Normal-Cone-Culling<br/>bei vorhandenen Bounds"}
    Cull -->|sichtbar| Mesh["Mesh-Shader<br/>Vertex Pulling und Primitive-Ausgabe"]
    Cull -->|verworfen| Skip["Kein Mesh-Dispatch"]

    Geometry --> BLAS["Separate BLAS pro Geometrie"]
    BLAS --> TLAS["TLAS mit Szeneninstanzen"]

    Mesh --> Fragment["Fragment-Shader<br/>Metallic-Roughness-PBR"]
    TLAS --> Queries["Inline Ray Queries<br/>Schatten und optionale Reflexionen"]
    Light["Richtungslicht,<br/>prozeduraler Himmel oder HDR"] --> Fragment
    Queries --> Fragment
    Fragment --> Denoise{"Vulkan-Denoiser aktiv?"}
    Denoise -->|Nein| ToneMap["Tone Mapping im Fragmentshader<br/>direkt ins Swapchain-Image"]
    Denoise -->|Ja| HDRImage["Lineares Vulkan-Offscreenbild<br/>RGBA16F"]
    HDRImage --> ExtHalf["Vulkan Image → exportierter<br/>persistenter HALF4-Buffer"]
    ExtHalf --> WaitCuda["External Semaphore<br/>Vulkan signalisiert CUDA"]
    WaitCuda --> OptixDenoiser["OptiX AI Denoiser<br/>HALF4 → HALF4"]
    OptixDenoiser --> CudaTone["CUDA/OptiX Tone Mapping<br/>gepacktes RGBA8"]
    CudaTone --> WaitVk["External Semaphore<br/>CUDA signalisiert Vulkan"]
    WaitVk --> CopyVk["Vulkan Buffer → Swapchain"]
    ToneMap --> Swapchain["Swapchain-Image"]
    CopyVk --> Swapchain
    Swapchain --> ImGui["ImGui-Overlay"]
    ImGui --> Present["Präsentation"]
```

Wichtige Eigenschaften:

- Der Pfad erzeugt jeden Frame direkt und akkumuliert nicht progressiv.
- `Samples per frame` und `Max bounces` haben deshalb im Vulkan-Pfad keine Wirkung.
- HDR-Roughness wird über die Mip-Pyramide gefiltert; das Environment überschreibt den Materialwert nicht.
- Ohne Denoiser rendert und tonemappt der Fragmentshader direkt ins Swapchain-Image.
- Mit Denoiser bleiben Denoising, Tone Mapping und Bildübergabe vollständig auf der GPU.
- Persistente External-Memory-Zuordnung und Semaphoren vermeiden CPU-Wartepunkte und Map/Unmap pro Frame.

## OptiX: progressiver Pathtracer

```mermaid
flowchart TD
    Scene["Gemeinsame Szene, Kamera,<br/>Materialien und Beleuchtung"] --> Flatten["Aktuell: Instanzen zu<br/>Weltgeometrie zusammenführen"]
    Flatten --> CudaBuffers["CUDA Vertex-, Index-,<br/>Normalen- und Materialbuffer"]
    CudaBuffers --> GAS["OptiX Triangle GAS"]

    Shader["Slang → PTX<br/>Raygen, Miss, Closest Hit"] --> Pipeline["OptiX-Pipeline und SBT"]
    Pipeline --> Launch["optixLaunch"]
    GAS --> Launch
    Params["Kleine Launch-Parameter<br/>Kamera, Samples, Bounces, Licht"] --> Launch

    Launch --> Integrator["Progressiver GGX-Pathtracer<br/>direktes Licht, Sekundärpfade,<br/>Russian Roulette"]
    Integrator --> Accum["CUDA float4<br/>Akkumulationsbuffer"]
    Accum --> Tone["CUDA/OptiX Tone Mapping<br/>gepacktes RGBA8"]
    Tone --> External["Vulkan-kompatibler<br/>External-Memory-Buffer"]
    External --> Semaphore["External Semaphore<br/>CUDA signalisiert Vulkan"]
    Semaphore --> Copy["Vulkan Buffer → Swapchain"]
    Copy --> ImGui["ImGui-Overlay"]
    ImGui --> Present["Präsentation"]
```

Wichtige Eigenschaften:

- `Samples per frame` und `Max bounces` steuern nur diesen Pfad.
- Änderungen an Kamera, Szene, Licht oder Material setzen die Akkumulation zurück.
- SBT-Header werden beim Pipelineaufbau gepackt; pro Frame werden nur kleine Launch-Parameter aktualisiert.
- Der OptiX-Renderpfad verwendet keinen Denoiser.
- Die Ausgabe wird ohne synchrone Device-to-Host-Kopie oder CPU-Tone-Mapping direkt an Vulkan übergeben.
- Separate OptiX-GAS plus IAS-Instanzen sind als nächster Geometrieausbau vorgesehen.

## Backend-Wechsel

```mermaid
stateDiagram-v2
    [*] --> Vulkan
    Vulkan --> OptiX: NVIDIA OptiX auswählen
    OptiX --> Vulkan: Vulkan Mesh + Ray Query auswählen
    Vulkan --> Vulkan: Szene oder Renderparameter geändert
    OptiX --> OptiX: Änderung setzt Akkumulation zurück
```

Vulkan besitzt in beiden Fällen die Swapchain und rendert das ImGui-Overlay. Dadurch bleibt die Oberfläche beim Backend-Wechsel erhalten und OptiX benötigt kein eigenes Präsentationssystem.
