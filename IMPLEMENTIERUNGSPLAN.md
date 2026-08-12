# Implementierungsplan: Vulkan-/OptiX-PBR-Raytracer

## Umsetzungsstand (11. August 2026)

Der ausführbare vertikale Schnitt ist erstellt und in Debug sowie Release getestet. Umgesetzt sind Visual-Studio-/C++20-Projekte, Assimp-Import, meshoptimizer-LODs/Meshlets, Vulkan-Meshshader, Metallic-Roughness-PBR, Vulkan-BLAS/TLAS mit aktivem Ray-Query-Schatten, ImGui sowie ein umschaltbares OptiX-Backend mit Slang-PTX, Triangle-GAS, SBT, Raygen/Miss/Closest-Hit, progressiver Abtastung und sichtbarer Ausgabe im Vulkan-Fenster.

Der aktuelle Schnitt verarbeitet alle Mesh-Instanzen mit ihren Node-Transformationen und Basis-LODs als zusammengeführte Weltgeometrie. OptiX löst pro getroffenem Dreieck die Material-ID auf und interpoliert transformierte Vertexnormalen. Als weiterer Produktionsausbau bleiben insbesondere separate GPU-IAS-Instanzen, Textur-Residency, vollständige Reflexions-/GI-Pässe, rekursive OptiX-Bounces, Denoising und GPU-direktes Vulkan/CUDA-External-Memory-Interop offen. Die folgenden Abschnitte beschreiben weiterhin das vollständige Zielbild.

## 1. Ziel

Entwickelt wird eine native Windows-Anwendung in C++20 als Visual-Studio-Solution. Die Anwendung importiert Szenen über Assimp, bereitet Geometrie mit meshoptimizer als Meshlets auf und stellt sie über eine Vulkan-Task-/Mesh-Shader-Pipeline dar. Das Beleuchtungsmodell ist physikalisch basiert (PBR).

Der Vulkan-Renderer arbeitet hybrid:

1. Task- und Mesh-Shader bestimmen die primäre Sichtbarkeit und erzeugen einen G-Buffer.
2. Compute-Shader verwenden Vulkan Ray Queries für raytraced Schatten, Reflexionen und indirektes Licht.
3. Ein PBR-Resolve- und Postprocessing-Pass erzeugt das finale HDR-/LDR-Bild.

Über einen Schalter im ImGui-UI kann alternativ NVIDIA OptiX als vollständiger PBR-Pathtracing-Renderer verwendet werden. Beide Backends verwenden dieselbe Szene, Kamera, Materialien und Renderparameter.

Mesh-Shader sind compute-ähnliche Stufen der Vulkan-Grafikpipeline, führen aber selbst keine Ray-Traversierung aus. Deshalb werden sie im Vulkan-Backend mit Ray Queries in Compute-Shadern kombiniert.

## 2. Verifizierte Entwicklungsumgebung

### Hardware und Treiber

- NVIDIA GeForce RTX 4070 Laptop GPU
- NVIDIA-Treiber 610.62
- Compute Capability 8.9
- Unterstützung für `VK_EXT_mesh_shader`
- Unterstützung für `VK_KHR_acceleration_structure`
- Unterstützung für `VK_KHR_ray_query`

### Bibliotheken und SDKs

| Komponente | Pfad/Version |
|---|---|
| Vulkan SDK | `G:\CodingLibraries\VulkanSDK`, Header 1.4.357 |
| Slang | `G:\CodingLibraries\slang-14.1`, Version 2026.14.1 |
| Assimp | `G:\CodingLibraries\assimp` |
| GLFW | `G:\CodingLibraries\glfw-3.5.1`, Version 3.5.1 |
| Dear ImGui | `G:\CodingLibraries\imgui`, Version 1.93.0 WIP |
| meshoptimizer | `G:\CodingLibraries\meshoptimizer`, Version 1.2 |
| CUDA | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| OptiX | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |
| Bilddecoder | `G:\CodingLibraries\stb\stb_image.h` |

Assimp wird mit konsistenten x64-Debug- und x64-Release-Artefakten aus
`G:\CodingLibraries\assimp\out-v145` eingebunden. Ein lokaler Assimp-Build im Projekt ist nicht erforderlich.
GLFW und meshoptimizer liegen als Quellcode vor und werden als Solution-Projekte gebaut.

## 3. Solution- und Projektstruktur

```text
VORaytracer.sln
├── VORaytracer.App
├── VORaytracer.Core
├── VORaytracer.Assets
├── VORaytracer.Vulkan
├── VORaytracer.Optix
├── VORaytracer.Shaders
├── ThirdParty.GLFW
├── ThirdParty.Meshoptimizer
└── VORaytracer.Tests
```

### Verantwortlichkeiten

- `VORaytracer.App`: Programmeinstieg, Fenster, ImGui, Eingabe und Hauptschleife.
- `VORaytracer.Core`: Mathematik, Kamera, Szene, Materialien, Lichter, Renderparameter und Backend-Schnittstellen.
- `VORaytracer.Assets`: Assimp-Import, Texturen, meshoptimizer-Verarbeitung und Meshlet-Daten.
- `VORaytracer.Vulkan`: Vulkan-Kontext, Ressourcen, Mesh-Shader-Pipeline, Ray Queries und Postprocessing.
- `VORaytracer.Optix`: CUDA-/OptiX-Kontext, GAS/IAS, SBT, Launch und Vulkan-Interop.
- `VORaytracer.Shaders`: Slang-Quelldateien, gemeinsame PBR-Module und Build-Regeln.
- `VORaytracer.Tests`: CPU-Tests, GPU-Smoke-Tests und Bildvergleiche.

### Build-Konfiguration

- Plattform: ausschließlich x64.
- Sprachstandard: `/std:c++20`.
- Warnstufe: `/W4`, projektinterner Code zusätzlich `/WX` nach der Aufbauphase.
- Laufzeitbibliothek: `/MDd` für Debug und `/MD` für Release.
- Debug-Informationen auch in Release, damit GPU-Abstürze analysiert werden können.
- Zentrale Property Sheets:
  - `Dependencies.props`
  - `Compiler.props`
  - `Shaders.props`
- DLLs für Assimp und Slang werden per Post-Build-Schritt neben die EXE kopiert.
- Pfade werden in den Property Sheets definiert und nicht über globale Systemvariablen vorausgesetzt.

## 4. Grundarchitektur

```text
Assimp-Datei
    │
    ▼
CPU-Szene und Materialien
    │
    ├── meshoptimizer ──► LODs, Meshlets und Bounds
    │
    ├── Vulkan-Ressourcen ──► Mesh-Pipeline + Ray Queries
    │
    └── OptiX-Ressourcen ───► GAS/IAS + Pathtracing
                                  │
Vulkan HDR-Ausgabe ◄──────────────┘
    │
Tone Mapping / ImGui
    │
Swapchain
```

Die Schnittstelle `IRenderBackend` kapselt mindestens:

- Initialisierung und Feature-Prüfung
- Resize
- Szenen-Upload und Szenenfreigabe
- Kamera- und Renderparameter
- Frame-Rendering
- Akkumulations-Reset
- Statistiken und GPU-Zeiten
- Verfügbarkeit des Backends

Implementierungen:

- `VulkanHybridRenderer`
- `OptixPathTracer`

Backend-spezifische GPU-Ressourcen bleiben getrennt. CPU-Szenendaten und semantische Materialdaten werden gemeinsam genutzt.

## 5. Asset- und Texturpipeline

### Assimp-Import

Beim Import werden mindestens folgende Optionen beziehungsweise äquivalente Verarbeitungsschritte verwendet:

- Triangulierung
- Zusammenführen identischer Vertices
- Validierung der importierten Daten
- Generierung fehlender Normalen
- Generierung von Tangenten für Normal Mapping
- konsistente UV- und Koordinatensystembehandlung
- Übernahme der vollständigen Szenenhierarchie
- Instanzierung identischer Meshes ohne CPU-Duplikation

Importiert werden:

- Position, Normale, Tangente und Texturkoordinaten
- Indexdaten
- Node-Transformationen
- Kameras und Lichter, soweit im Format vorhanden
- Base Color/Albedo
- Metallic und Roughness
- Normal Map
- Ambient Occlusion
- Emissive
- Alpha Mode und Alpha Cutoff
- doppelseitige Materialien

Externe und eingebettete Texturen werden unterstützt. `stb_image` übernimmt PNG-, JPEG- und HDR-Dekodierung. Farbtexturen werden als sRGB, Daten- und Normaltexturen als linear behandelt.

### Einheitliches Datenformat

Die CPU-Szene enthält unter anderem:

- `Scene`
- `Node`
- `Mesh`
- `MeshLod`
- `Meshlet`
- `Material`
- `Texture`
- `Instance`
- `Camera`
- `Light`

Material- und Instanz-IDs bleiben zwischen Vulkan und OptiX stabil. GPU-Strukturen erhalten explizite Größen und Ausrichtungen, die sowohl zur Slang-Reflection als auch zum C++-Layout passen.

## 6. Geometrieaufbereitung mit meshoptimizer

Der zuvor erwogene eigene Meshlet-Builder entfällt. Die vollständige Geometrieaufbereitung verwendet meshoptimizer 1.2.

### Verarbeitung pro Mesh und LOD

1. Kanonisches Vertex-/Indexformat aus Assimp erzeugen.
2. Vertex-Duplikate mit `meshopt_generateVertexRemap` bestimmen.
3. Vertex- und Indexbuffer entsprechend der Remap-Tabelle neu anordnen.
4. Post-Transform-Cache mit `meshopt_optimizeVertexCache` optimieren.
5. Overdraw mit `meshopt_optimizeOverdraw` reduzieren.
6. Speicherzugriffe mit `meshopt_optimizeVertexFetch` optimieren.
7. Zusätzliche LOD-Stufen mit `meshopt_simplify` erzeugen.
8. Jede LOD-Stufe erneut für Cache und Fetch optimieren.
9. Meshlets mit `meshopt_buildMeshlets` erzeugen.
10. Bounding Sphere und Normal Cone mit `meshopt_computeMeshletBounds` berechnen.

Die Meshlet-Grenzen werden nicht blind fest codiert. Zuerst werden die Limits aus `VkPhysicalDeviceMeshShaderPropertiesEXT` gelesen. Darauf basierend wird ein kompatibles Profil gewählt, beispielsweise bis zu 64 Vertices und 124 beziehungsweise 126 Dreiecke pro Meshlet.

Gespeichert werden:

- Meshlet-Descriptor
- Offset und Anzahl lokaler Vertices
- Offset und Anzahl lokaler Dreiecke
- lokale Vertex-Indizes
- gepackte lokale Dreiecksindizes
- Bounding Sphere
- Normal Cone
- Mesh-, Material- und LOD-Referenz

Das Ergebnis wird optional als eigener Asset-Cache gespeichert, damit unveränderte Modelle bei späteren Starts nicht erneut verarbeitet werden müssen. Der Cache enthält eine Formatversion und Hashes der Quelldateien sowie der Importoptionen.

## 7. Vulkan-Basis

### Initialisierung

- GLFW-Fenster und Win32-Vulkan-Surface
- Vulkan-Instance mit Validation Layer in Debug
- Auswahl der diskreten NVIDIA-GPU
- Graphics-/Compute-/Transfer-Queues
- Swapchain mit Resize- und Out-of-Date-Behandlung
- Timeline Semaphores und Synchronization2
- Debug-Namen für Vulkan-Objekte

### Benötigte Vulkan-Funktionen

- Vulkan 1.3 als minimale Gerätebasis, soweit vom Zielgerät unterstützt
- Dynamic Rendering
- Synchronization2
- Buffer Device Address
- Descriptor Indexing beziehungsweise Bindless Resources
- `VK_EXT_mesh_shader`
- `VK_KHR_acceleration_structure`
- `VK_KHR_ray_query`
- `VK_KHR_deferred_host_operations`
- externe Win32-Memory- und Semaphore-Erweiterungen für CUDA-Interop

Alle Features und Limits werden zur Laufzeit abgefragt. Bei fehlenden Pflichtfeatures erscheint eine verständliche Fehlermeldung statt eines späteren Pipeline-Fehlers.

### Ressourcenverwaltung

Die erste Version verwendet einen eigenen, klar begrenzten Vulkan-Allokator mit großen Memory-Blöcken und Suballokation für Buffer und Images. Staging-Uploads werden über einen persistent gemappten Upload-Ring ausgeführt. Ressourcenzerstörung erfolgt verzögert, sobald die zugehörige GPU-Timeline abgeschlossen ist.

## 8. Slang-Shader-Workflow

Alle selbst entwickelten Renderer-Shader werden in Slang geschrieben. Die internen Vendor-Shader des unveränderten ImGui-Vulkan-Backends gelten nicht als Teil des Renderers.

### Gemeinsame Module

```text
Shaders/
├── Shared/
│   ├── Types.slang
│   ├── Camera.slang
│   ├── Materials.slang
│   ├── Pbr.slang
│   ├── Sampling.slang
│   ├── Lights.slang
│   └── Color.slang
├── Vulkan/
│   ├── MeshletTask.slang
│   ├── MeshletMesh.slang
│   ├── GBufferFragment.slang
│   ├── RayQueryLighting.slang
│   ├── PbrResolve.slang
│   └── ToneMap.slang
└── Optix/
    ├── RayGen.slang
    ├── Miss.slang
    ├── ClosestHit.slang
    └── AnyHit.slang
```

### Kompilierung

- Vulkan-Targets werden als SPIR-V erzeugt.
- OptiX-Targets werden über Slang und NVRTC als PTX erzeugt.
- Debug unterstützt Shader-Hot-Reload und aussagekräftige Diagnosen.
- Release verwendet vorab kompilierte Shader-Artefakte.
- Slang-Reflection erzeugt beziehungsweise validiert Descriptor-, Push-Constant- und Parameterlayouts.
- Ein Shaderwechsel wird erst aktiviert, wenn die neue Variante erfolgreich kompiliert wurde.

## 9. Vulkan-Task-/Mesh-Shader-Pipeline

### Task Shader

Der Task Shader verarbeitet Meshlet-Gruppen und übernimmt:

- Frustum Culling
- Normal-Cone-/Backface-Culling
- Distanz- und Bildschirmgrößen-basierte LOD-Auswahl
- optional Hi-Z-Occlusion-Culling
- Erzeugung des Payloads für sichtbare Meshlets
- Dispatch der erforderlichen Mesh-Shader-Workgroups

### Mesh Shader

Eine Mesh-Shader-Workgroup verarbeitet in der Regel genau ein Meshlet:

- Lesen der Meshlet- und Instanzdaten aus Storage Buffern
- Vertex Pulling über die lokalen Meshlet-Indizes
- Objekt-, Welt- und Clipraumtransformation
- Ausgabe von Vertices und Dreiecksprimitiven
- Ausgabe von Material-, Instanz- und Primitive-ID
- korrekte Transformation von Normalen und Tangenten
- Bewegungsvektoren aus aktueller und vorheriger Transformation

### Fragment Shader und G-Buffer

Der Fragment Shader schreibt mindestens:

- lineare oder rekonstruierbare Tiefe
- World-Space-Normale
- Base Color
- Metallic und Roughness
- Emissive
- Material-/Instanz-ID
- Bewegungsvektor

Alpha-Mask-Materialien verwerfen Fragmente anhand des Material-Cutoffs. Transparente Materialien werden zunächst separat und vorwärts gerendert; komplexe Brechung folgt in einer späteren Ausbaustufe.

## 10. PBR-Modell

Das Grundmodell entspricht einem Metallic-Roughness-Workflow:

- GGX/Trowbridge-Reitz-Normalverteilung
- Smith-Geometriefunktion
- Fresnel-Schlick
- energieerhaltende diffuse Komponente, zunächst Lambert, optional Burley
- Metallic-/Dielectric-F0-Behandlung
- Normal Mapping mit Tangent Space
- Emissive-Materialien
- Ambient Occlusion
- Punkt-, Spot- und Richtungslichter
- Flächenlichter für Raytracing
- Image-Based Lighting mit HDR-Environment

Die komplette Lichtberechnung erfolgt linear. Erst im Postprocessing folgen Exposure, ACES-Filmic-Tone-Mapping und die Umwandlung nach sRGB.

Vulkan und OptiX verwenden dieselben Slang-Funktionen für Materialauswertung, Fresnel, GGX und Sampling. Backend-spezifisch bleiben nur Traversierung, Ressourcenbindung und Entry Points.

## 11. Vulkan Ray Queries und Compute-Beleuchtung

### Acceleration Structures

- Eine BLAS pro eindeutiger Mesh-/LOD-Geometrie, soweit sinnvoll.
- TLAS-Instanzen übernehmen die Node-Transformationen.
- Statische BLAS werden nach dem Build kompaktiert.
- Transformationsänderungen aktualisieren bevorzugt nur die TLAS.
- Deformierte Geometrie verwendet Refit oder Rebuild abhängig von Änderungsumfang und Performance.

Die Acceleration Structures lesen die vollständigen optimierten Vertex- und Indexbuffer. Meshlet-Ausgaben sind kurzlebige Ergebnisse der Rasterpipeline und werden nicht direkt als BLAS-Eingabe verwendet.

### Compute-Pass

Der Ray-Query-Compute-Shader liest G-Buffer und TLAS und berechnet schrittweise:

1. harte raytraced Schatten
2. weiche Schatten für Flächenlichter
3. spiegelnde und raue Reflexionen
4. Ambient Occlusion
5. eine oder mehrere diffuse indirekte Lichtkomponenten
6. progressive Akkumulation

Primärstrahlen beziehungsweise primäre Sichtbarkeit kommen weiterhin aus der Mesh-Shader-Pipeline. Sekundärstrahlen verwenden `VK_KHR_ray_query`. Das Ergebnis ist ein hybrider raytraced PBR-Renderer.

Für stochastische Effekte werden reproduzierbare, pro Pixel und Sample variierte Zufallssequenzen verwendet. Kamera-, Szenen-, Licht- oder Materialänderungen setzen die Akkumulation zurück.

## 12. OptiX-PBR-Pathtracer

### Initialisierung

- CUDA Driver Context für die ausgewählte NVIDIA-GPU
- OptiX Device Context
- Validierungs- und Log-Callbacks in Debug
- Slang-zu-PTX-Kompilierung über CUDA 13.3/NVRTC
- OptiX-Modul- und Pipeline-Erzeugung

### Geometrie

- GAS pro eindeutigem Mesh
- IAS für Instanzen
- Compaction statischer GAS
- Any-Hit nur dort, wo Alpha-Mask oder spezielle Geometrie es erfordert
- Shader Binding Table mit Material-/Geometriereferenzen

### Shaderprogramme

- Ray Generation
- Miss für Environment/Hintergrund
- Closest Hit für PBR-Oberflächen
- Any Hit für Alpha-Mask
- später optional zusätzliche Ray-Typen oder Direct Callable Programs

### Integrator

Der OptiX-Renderer startet als progressiver Pathtracer mit:

- mehreren Bounces
- GGX-BSDF-Sampling
- Next Event Estimation
- Russian Roulette
- Environment Lighting
- Flächenlichtern
- Multiple Importance Sampling als geplanter Qualitätsausbau
- optionaler OptiX-AI-Denoiser als Vulkan-Postrender-Schritt; nicht im OptiX-Pathtracing-Pfad

Die Materialparameter und grundlegenden PBR-Funktionen bleiben mit dem Vulkan-Backend identisch. Da Vulkan hybrid und OptiX vollständig pfadbasiert arbeitet, wird keine bitgenaue Bildgleichheit erwartet; Materialreaktion, Farbmanagement und Kameramodell müssen jedoch übereinstimmen.

## 13. Vulkan-/CUDA-/OptiX-Interop

ImGui und Präsentation verbleiben immer auf Vulkan. OptiX schreibt in einen gemeinsam nutzbaren linearen HDR-Ausgabebuffer:

1. Vulkan erzeugt exportierbaren Speicher für einen `float4`-Buffer.
2. Der Win32-Memory-Handle wird in CUDA als External Memory importiert.
3. OptiX schreibt direkt in den importierten CUDA-Pointer.
4. Ein externes Timeline-/Binary-Semaphore synchronisiert CUDA und Vulkan.
5. Vulkan liest den Buffer im Tone-Mapping-Pass und schreibt in das Ausgabebild.
6. ImGui zeigt das resultierende Bild im Viewport an.

Damit entfällt eine CPU-Kopie des OptiX-Bildes. Die erste korrekte Implementierung darf zur Fehlereingrenzung noch einen einfachen Kopierpfad verwenden; das finale Abnahmekriterium ist die External-Memory-Variante.

## 14. ImGui-Oberfläche

### Hauptbereiche

- Render-Viewport
- Hauptmenü mit Dateiimport
- Szenenhierarchie
- Inspector für Nodes, Meshes, Materialien und Lichter
- Kameraeinstellungen
- Render-Einstellungen
- Performance-/Debug-Fenster

### Backend-Schalter

Auswahlmöglichkeiten:

- `Vulkan Mesh + Ray Query`
- `NVIDIA OptiX`

Der OptiX-Eintrag ist deaktiviert und zeigt einen Grund an, wenn CUDA, OptiX oder eine passende NVIDIA-GPU nicht verfügbar ist. Beim Wechsel werden die Kamera- und Szeneneinstellungen beibehalten, aber Akkumulation und backendabhängige temporale Historie zurückgesetzt.

### Renderparameter

- Auflösung und Render Scale
- Samples pro Frame
- maximale Bounce-Anzahl
- Exposure
- Tone Mapper
- Environment und Lichtstärken
- Shadow-, Reflection- und GI-Qualität
- Vulkan-Postrender-Denoiser ein/aus
- VSync

### Debugansichten

- Meshlets mit Zufallsfarben
- LOD-Stufe
- Bounding Spheres und Normal Cones
- sichtbare/verworfene Meshlets
- Base Color, Normalen, Roughness und Metallic
- Tiefe und Bewegungsvektoren
- direkte und indirekte Beleuchtung
- Sample Count
- Acceleration-Structure-Statistiken

## 15. Frameablauf des Vulkan-Backends

1. Eingabe und Kamera aktualisieren.
2. Transformationen und Animationen aktualisieren.
3. Geänderte GPU-Buffer hochladen.
4. Dynamische BLAS/TLAS aktualisieren.
5. Optional Hi-Z-Pyramide aus dem vorherigen Frame vorbereiten.
6. Task-/Mesh-Shader-G-Buffer-Pass ausführen.
7. Ray-Query-Compute-Pass ausführen.
8. PBR-Beiträge zusammenführen und akkumulieren.
9. Denoising beziehungsweise temporale Filterung ausführen.
10. Tone Mapping durchführen.
11. ImGui rendern.
12. Swapchain-Image präsentieren.

Synchronisation wird über explizite Ressourcenübergänge, Synchronization2 und Timeline Semaphores beschrieben. Jede Ressource besitzt einen nachvollziehbaren Besitzer und Zustand.

## 16. Fehlerbehandlung und Diagnose

- Vulkan Validation Layer in Debug
- Slang-Compilerdiagnosen mit Datei und Zeile
- OptiX-Log-Callback
- CUDA- und OptiX-Fehlercodes werden in Exceptions oder strukturierte Fehlerobjekte übersetzt
- Debug-Namen für Buffer, Images, Pipelines und Acceleration Structures
- GPU-Timestamps für alle großen Renderpässe
- robuste Behandlung von Fenster-Resize, minimiertem Fenster und Swapchain-Neuerzeugung
- verständliche Importfehler mit Dateipfad und Assimp-Fehlermeldung

## 17. Tests

### CPU-Tests

- Assimp-zu-Scene-Konvertierung
- Transformationshierarchie
- Materialkonvertierung
- Texturfarbraum-Zuordnung
- meshoptimizer-Remap
- LOD- und Meshlet-Grenzen
- Cache-Serialisierung
- PBR-Hilfsfunktionen, soweit CPU-seitig spiegelbar

### GPU-Tests

- Vulkan-Initialisierung
- Mesh-Shader-Featureprüfung
- Rendering eines einzelnen Meshlets
- BLAS-/TLAS-Build
- einfacher Ray Query
- OptiX-Kontext und Testlaunch
- External-Memory-Interop
- Shader-Reflection und Ressourcenlayouts

### Referenzszenen

- einzelnes Dreieck
- strukturierter Würfel
- Metallic-/Roughness-Materialkugeln
- Cornell Box
- texturiertes Assimp-Modell
- Szene mit vielen Instanzen und LODs
- Alpha-Mask-Geometrie

### Qualitätskriterien

- keine Vulkan-Validation-Fehler
- keine CUDA-/OptiX-Fehler
- keine NaN-/Inf-Pixel im HDR-Bild
- konsistentes Farbmanagement in beiden Backends
- stabiler Backend-Wechsel ohne Neustart
- keine stetig wachsenden GPU- oder CPU-Allokationen

## 18. Meilensteine

### M1: Solution und Fenster

- Visual-Studio-Projekte und Property Sheets
- GLFW-Fenster
- Vulkan-Instance, Device und Swapchain
- ImGui-Docking und leerer Viewport

### M2: Asset-Pipeline

- Assimp-Import
- Texturladen
- gemeinsame Szenendaten
- meshoptimizer-Optimierung
- LODs, Meshlets und Bounds

### M3: Mesh-Shader-Rasterisierung

- Task- und Mesh-Shader
- Meshlet-Culling
- G-Buffer
- Meshlet- und LOD-Debugansichten

### M4: PBR-Rasterbasis

- Metallic-Roughness-Materialien
- Normal Mapping
- direkte Lichter
- Environment Lighting
- HDR und Tone Mapping

### M5: Vulkan Ray Queries

- BLAS/TLAS
- raytraced Schatten
- Reflexionen und AO
- erste indirekte Beleuchtung
- Akkumulation

### M6: OptiX

- CUDA-/OptiX-Kontext
- Slang-PTX
- GAS/IAS und SBT
- erster PBR-Pathtracing-Pfad

### M7: Interop und UI-Wechsel

- gemeinsamer HDR-Ausgabebuffer
- externe Memory-/Semaphore-Synchronisation
- stabiler ImGui-Backend-Switch
- gemeinsame Renderparameter

### M8: Qualität und Performance

- Denoising
- Profiling
- Pipeline- und Asset-Caches
- Tests und Referenzbilder
- Speicher- und Synchronisationsbereinigung

## 19. Abnahmekriterien der ersten vollständigen Version

Die erste vollständige Version gilt als fertig, wenn:

1. die Solution in Visual Studio als x64 Debug und Release mit C++20 gebaut werden kann;
2. ein Modell über Assimp samt PBR-Texturen geladen werden kann;
3. meshoptimizer LODs, Meshlets und Bounds für die Mesh-Shader-Pipeline erzeugt;
4. der Vulkan-Renderer die Szene über Task-/Mesh-Shader und PBR darstellt;
5. Ray Queries mindestens Schatten und Reflexionen berechnen;
6. der OptiX-Renderer dieselbe Szene als PBR-Pathtracing-Bild rendert;
7. im ImGui-UI ohne Neustart zwischen Vulkan und OptiX gewechselt werden kann;
8. Resize, Szenenwechsel und Kameraänderungen stabil funktionieren;
9. beide Backends dasselbe Kameramodell, Materialmodell und Farbmanagement verwenden;
10. der normale Testlauf keine Vulkan-, CUDA- oder OptiX-Validierungsfehler erzeugt.

## 20. Empfohlene Umsetzungsreihenfolge

Die Entwicklung erfolgt strikt in vertikalen, jeweils lauffähigen Schritten:

1. Vulkan-Fenster und ImGui
2. einzelnes hart codiertes Meshlet
3. Assimp und meshoptimizer
4. vollständige Mesh-Shader-Szene
5. PBR ohne Raytracing
6. BLAS/TLAS und ein einzelner Schatten-Ray-Query
7. Reflexionen, GI und Akkumulation
8. OptiX-Testdreieck
9. gemeinsame OptiX-Szene und PBR
10. Vulkan-CUDA-Interop
11. Backend-Schalter
12. Tests, Denoising und Optimierung

Jeder Schritt endet mit einer sichtbaren Demo, einem GPU-Smoke-Test und einer Prüfung der Validation-Ausgabe. Dadurch werden Fehler in Import, Shaderlayouts, Synchronisation und Interop früh isoliert.
