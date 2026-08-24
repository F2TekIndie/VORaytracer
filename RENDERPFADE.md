# Renderpfade

Diese Übersicht zeigt, welche Daten beide Backends teilen und wo sich Vulkan und OptiX unterscheiden. Die Diagramme verwenden Mermaid und werden in GitHub, Visual Studio Code und anderen Mermaid-fähigen Markdown-Ansichten direkt gerendert.

## Gemeinsame Asset-Aufbereitung

```mermaid
flowchart LR
    Model["Modell über IFileOpenDialog"] --> Assimp["Assimp-Import"]
    Assimp --> Scene["Gemeinsame CPU-Szene<br/>Meshes, Instanzen, Lichter, GpuMaterial"]
    Assimp --> Textures["CPU-Texturcache mit Mips,<br/>UV0/UV1, UV-Transform und Sampler"]
    Scene --> Meshopt["meshoptimizer<br/>Remap, Cache, Meshlets, Bounds, LODs"]
    HDR["Radiance HDR über IFileOpenDialog"] --> STB["stb_image<br/>Float-Decodierung"]
    STB --> Mips["HDR-Mips und Importance-CDF"]
    Mips --> IBL["Vulkan-IBL-Atlas<br/>Irradiance, GGX-Mips, BRDF-LUT"]
    Meshopt --> VulkanData["Vulkan-Ressourcen"]
    Meshopt --> OptixData["CUDA-/OptiX-Ressourcen"]
    Textures --> VulkanData
    Textures --> OptixData
    Mips --> VulkanData
    Mips --> OptixData
    IBL --> VulkanData
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

    Mesh --> Fragment["Fragment-Shader<br/>gemeinsame modulare PBR-Auswertung"]
    TLAS --> Queries["Inline Ray Queries<br/>Schatten und optionale Reflexionen"]
    Light["Richtungs-/Punkt-/Spot-/Flächenlicht,<br/>emissive Meshes, Sky oder HDR-IBL"] --> Fragment
    Queries --> Fragment
    Fragment --> Approx["Begrenzte Echtzeitapproximationen<br/>Transmission, SSS, Volumen"]
    Approx --> Denoise{"Vulkan-Denoiser aktiv?"}
    Denoise -->|Nein| ToneMap["Tone Mapping im Fragmentshader<br/>direkt ins Swapchain-Image"]
    Denoise -->|Ja| HDRImage["Lineares Vulkan-Offscreenbild<br/>RGBA16F"]
    HDRImage --> ExtHalf["Vulkan Image → exportierter<br/>persistenter HALF4-Buffer"]
    ExtHalf --> WaitCuda["External Semaphore<br/>Vulkan signalisiert CUDA"]
    WaitCuda --> OptixDenoiser["Temporaler OptiX AI Denoiser<br/>HALF4 + GPU-History → HALF4"]
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
- HDR-Roughness wählt die GGX-vorgefilterte Atlasstufe und die Split-Sum-BRDF-LUT; das Environment überschreibt den Materialwert nicht.
- Ohne Denoiser rendert und tonemappt der Fragmentshader direkt ins Swapchain-Image.
- Mit Denoiser bleiben Denoising, Tone Mapping und Bildübergabe vollständig auf der GPU.
- Der temporale Modus verwendet vorherige Beauty- und interne Guide-Layer; Änderungen an Kamera, Szene, Licht, Material oder Auflösung invalidieren die History.
- Persistente External-Memory-Zuordnung und Semaphoren vermeiden CPU-Wartepunkte und Map/Unmap pro Frame.
- Materialfaktoren werden bei UI-Änderungen als einzelner 240-Byte-Bereich in den device-local Materialbuffer übertragen.
- Debugansichten greifen vor der Beleuchtung auf dieselben aufgelösten `SurfaceData` wie das Beauty-Rendering zu.

## OptiX: progressiver Pathtracer

```mermaid
flowchart TD
    Scene["Gemeinsame Szene, Kamera,<br/>Materialien und Beleuchtung"] --> CudaBuffers["Einmalige objektlokale CUDA-Buffer<br/>Vertex, Index, Normal, Tangente, UV"]
    CudaBuffers --> GAS["Wiederverwendbares Triangle-GAS<br/>pro Mesh"]
    GAS --> IAS["IAS mit Szeneninstanzen,<br/>Transformationen und Material-IDs"]

    Shader["Slang → PTX<br/>Raygen, Miss, Closest Hit"] --> Pipeline["OptiX-Pipeline und SBT"]
    Pipeline --> Launch["optixLaunch"]
    IAS --> Launch
    Params["Kleine Launch-Parameter<br/>Kamera, Samples, Bounces, Licht"] --> Launch

    Launch --> Integrator["Progressiver PBR-Pathtracer<br/>VNDF, NEE/MIS, Mesh-Lichter,<br/>SSS, Medien, Russian Roulette"]
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
- HDR wird über eine zweistufige Luminanz-/Sinus-CDF importance-gesampelt; BSDF-, Environment- und emissive Dreieckslichter werden mit MIS kombiniert.
- Materialfaktoren aktualisieren nur einen GPU-Eintrag und setzen die Akkumulation zurück; Emissionsänderungen aktualisieren zusätzlich die Lichtverteilung.
- CUDA-Events messen den Launch asynchron, ohne den Renderpfad per `cuCtxSynchronize()` pro Frame zu blockieren.

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
