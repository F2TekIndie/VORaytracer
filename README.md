# Testing ChatGPT Codex 5.6 Sol for generating a hybrid rasterizer/raytracer

All coding is done by Codex.
Project is setup for Windows and NVIDIA RTX graphic cards.
 
# VORaytracer

VORaytracer ist ein nativer C++20-PBR-Renderer für Windows mit zwei umschaltbaren Renderpfaden:

- Vulkan 1.3 mit Task-/Mesh-Shadern, Meshlet-Culling und Inline Ray Queries
- NVIDIA OptiX als progressiver Pathtracer

Dear ImGui stellt die Bedienoberfläche und Vulkan die gemeinsame Präsentation beider Backends bereit. Modelle werden mit Assimp geladen, mit meshoptimizer aufbereitet und von Slang-Shadern verarbeitet.

## Lizenz

VORaytracer ist unter der [GNU Lesser General Public License v3.0 oder später](LICENSE) (`LGPL-3.0-or-later`) veröffentlicht. Copyright © 2026 VORaytracer contributors. Eingebundene Bibliotheken und SDKs unterliegen ihren jeweiligen eigenen Lizenzen.

- [Grafische Darstellung der Renderpfade](RENDERPFADE.md)

## Funktionsumfang

- Visual-Studio-Solution für x64 Debug und Release mit C++20
- natives `IFileOpenDialog` für Modelle und Radiance-HDR-Dateien
- Assimp-Import mit Szeneninstanzen, Transformationen und PBR-Materialwerten
- automatische Kameraeinpassung nach dem Laden eines Modells
- Orbit-, Pan- und Zoom-Navigation mit der Maus
- gemeinsame Lichttransformation über dieselben Mausbewegungen mit gedrückter Umschalttaste
- meshoptimizer-Meshlets, Bounds und LOD-Erzeugung
- optional abschaltbare Remap-, Cache-, Overdraw-, Vertex-Fetch-, Bounds- und LOD-Optimierung; die für den Mesh-Shader notwendige Meshlet-Erzeugung bleibt aktiv
- Task-/Amplification- und Mesh-Shader-Pipeline mit Frustum- und Normal-Cone-Culling auf vorhandenen Meshlet-Bounds
- device-local Vulkan-Geometrie über Staging-Uploads
- separate Vulkan-BLAS pro Geometrie und TLAS-Instanzen ohne getrennte `vkQueueWaitIdle()`-Stopps
- gemeinsames modulares Slang-PBR-System für Metallic-Roughness, Clearcoat, Glas/Absorption, Anisotropie, Sheen, Emission, Subsurface und homogene Volumen
- bindless Vulkan-Materialtexturen und CUDA-Mipmapped-Texture-Objects für Base Color, Metallic-Roughness, Normal, AO, Emissive und spezialisierte Loben
- Vulkan-GGX-Auswertung mit Ray-Query-Schatten, Reflexionen sowie begrenzten Echtzeitapproximationen für Transmission, Subsurface und Volumen
- Richtungslicht, prozeduraler Himmel oder HDR-Environment als globale Beleuchtung
- roughnessabhängiges HDR-IBL mit Cosine-Irradiance- und GGX-Importance-Faltung im Vulkan-Shader
- optionaler OptiX-AI-Denoiser als Vulkan-Postrender-Schritt mit `HALF4`-Ein- und Ausgabe
- CUDA/Vulkan-External-Memory- und Semaphore-Interop ohne Bildkopie zur CPU
- progressiver OptiX-Pathtracer mit GGX-VNDF, NEE/MIS für HDR, Richtungs- und Mesh-Lichter, Medium-Stack, Samples pro Frame, maximaler Pfadtiefe und Russian Roulette
- wiederverwendbare OptiX-GAS pro Mesh und IAS-Szeneninstanzen ohne Geometrie-Flattening
- standardisiertes hellgraues Kunststoffmaterial als nichtdestruktive Laufzeitüberschreibung
- ein- und ausblendbare Bodenebene mit Standardkunststoffmaterial
- stabile, pseudozufällige Meshlet-Debugfarbe pro Meshlet
- PBR-Materialvergleichsszene, Materialeditor mit gezielten 224-Byte-GPU-Updates und gemeinsame Material-Debugansichten
- asynchrone Vulkan-Timestamps und CUDA-Events sowie GPU-Speicher-/Descriptorstatistiken im Performance-Fenster

## Voraussetzungen

| Komponente | Erwarteter Pfad |
|---|---|
| Vulkan SDK | `G:\CodingLibraries\VulkanSDK` |
| Slang | `G:\CodingLibraries\slang-14.1` |
| Assimp | `G:\CodingLibraries\assimp\out-v145` |
| GLFW | `G:\CodingLibraries\glfw-3.5.1` |
| Dear ImGui | `G:\CodingLibraries\imgui` |
| meshoptimizer | `G:\CodingLibraries\meshoptimizer` |
| stb_image | `G:\CodingLibraries\stb\stb_image.h` |
| CUDA | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| OptiX | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |

Zusätzlich erforderlich sind Visual Studio mit C++-Toolset `v145`, ein kompatibles Windows SDK und eine NVIDIA-GPU mit den verwendeten Vulkan-Mesh-Shader-, Acceleration-Structure- und Ray-Query-Funktionen.

## Bauen und starten

```powershell
.\scripts\Build.ps1 -Configuration Debug
.\scripts\Build.ps1 -Configuration Release
```

Anschließend beispielsweise:

```powershell
.\bin\x64\Debug\VORaytracer.App.exe
.\bin\x64\Debug\VORaytracer.Tests.exe
```

Ohne explizit geladene Datei startet die Anwendung mit einer prozeduralen Testszenengeometrie.

## Bedienung

Die Kamerasteuerung reagiert außerhalb von ImGui-Fenstern:

| Eingabe | Kamera | Mit Umschalttaste |
|---|---|---|
| linke Maustaste ziehen | Orbit | Licht drehen |
| mittlere Maustaste ziehen | Pan | Licht verschieben, soweit auf den Lichttyp anwendbar |
| Mausrad | Zoom | Lichtabstand bzw. lichttypspezifischen Transformationsanteil ändern |

`Frame model` passt die Kamera erneut an die Bounds der geladenen Szene an. Die Lichtmodi Richtungslicht, prozeduraler Himmel und HDR teilen sich die anwendbaren Transformationsanteile, sodass die Umschaltung die Orientierung beibehält.

Im Menü `File` und über die jeweiligen Browse-Schaltflächen stehen native Windows-Dateidialoge für Modelle und HDR-Environments bereit.
`PBR material comparison` lädt ein Raster mit Referenzmaterialien für sämtliche implementierten Loben.

## Szenen- und Materialoptionen

- `meshoptimizer`: Lädt das aktuelle Modell beim Umschalten neu. Meshlet-Erzeugung bleibt immer aktiv; zusätzliche Optimierungen folgen dem Schalter.
- `Default plastic`: Setzt `materialOverrideId` auf das residente Material mit `Albedo 0.75`, `Metallic 0.0`, `Roughness 0.5`. Originalmaterialien und Texturreferenzen bleiben erhalten; Geometrie und Acceleration Structures werden nicht neu geladen.
- `Ground plane`: Blendet eine Bodenebene mit dem Standardkunststoffmaterial ein oder aus.
- `Meshlet debug colors`: Zeigt im Vulkan-Pfad eine stabile Zufallsfarbe pro Meshlet.
- `Ray-traced reflections`: Aktiviert Vulkan-Reflexions-Ray-Queries; im OptiX-Pfad steuert die Option die reflektierenden Sekundärpfade.
- `Indirect lighting`: Schaltet die globale Beleuchtung durch HDR, prozeduralen Himmel oder Richtungslicht entsprechend der aktiven Konfiguration.
- `Material editor`: Wählt Mesh und Material, zeigt Texturzuweisungen und bearbeitet sämtliche PBR-Faktoren. Faktoränderungen übertragen nur den betroffenen GPU-Materialeintrag; Emissionsänderungen bauen zusätzlich die Mesh-Licht-CDF neu auf.
- `Debug view`: Zeigt Base Color, Metallic, Roughness, Normalen, Tangenten, AO, Emission, einzelne Loben, PDF, IDs, Medium und OptiX-Pfadtiefe.

## Unterschiede der Backends

### Vulkan

Vulkan ist kein progressiver Pathtracer. Jeder Frame wird direkt über Task-/Mesh-Shader, PBR-Fragmentauswertung und optionale Inline Ray Queries erzeugt. Deshalb gelten `Samples per frame` und `Max bounces` nicht für diesen Pfad. Der optionale Denoiser wird ausschließlich nach dem Vulkan-Rendering ausgeführt.

Das HDR-Environment beeinflusst diffuse und spiegelnde Beleuchtung. Der Fragmentshader faltet das Environment über deterministische Cosine- beziehungsweise GGX-Importance-Samples; Roughness verändert dabei die GGX-Verteilung, nicht den Materialparameter. Alpha-Blend-Materialien verwenden sortierunabhängige dithered Coverage mit regulärem Depth-Write. Ohne Denoiser führt der Fragmentshader Tone Mapping aus und rendert direkt ins Swapchain-Image. Nur der Denoiser-Pfad verwendet vorher ein lineares `RGBA16F`-Offscreenbild und `HALF4`-Interop.

### OptiX

OptiX akkumuliert progressiv in einem CUDA-`float4`-Buffer. `Samples per frame` und `Max bounces` steuern den Integrator. Kamera-, Material-, Licht- oder Szenenänderungen setzen die Akkumulation zurück. Ray-Cone-Footprints wählen die Mip-Level der CUDA-Texturen. Reflexions- und indirekte Transportloben werden entsprechend ihrer UI-Schalter mit jeweils konsistenter Evaluation und PDF gesampelt. Subsurface verwendet einen instanzgebundenen, begrenzten Random Walk. Der OptiX-Pfad besitzt keinen eigenen Denoiser; sein Bild wird auf der GPU tonemapped und über Vulkan präsentiert.
Geometrien bleiben objektlokal in je einem wiederverwendbaren GAS; ein IAS enthält die Szeneninstanzen und ihre Transformationen.

## Umgebungsvariablen für Smoke-Tests

| Variable | Wirkung |
|---|---|
| `VOR_BACKEND=optix` | startet mit OptiX statt Vulkan |
| `VOR_DENOISER=1` | aktiviert den Vulkan-Postrender-Denoiser |
| `VOR_SCENE=<absoluter Pfad>` | lädt beim Start ein Modell |
| `VOR_MESHLET_DEBUG=1` | aktiviert die Meshlet-Debugfarben |
| `VOR_REFLECTIONS=0` | deaktiviert reflektierende Sekundärpfade für automatisierte Tests |
| `VOR_INDIRECT_LIGHTING=0` | deaktiviert diffuse/transmissive Sekundärpfade für automatisierte Tests |
| `VOR_DEFAULT_PLASTIC=1` | startet mit der nichtdestruktiven Default-Plastic-Materialüberschreibung |
| `VOR_GLOBAL_LIGHT=sky` | startet mit prozeduralem Himmel |
| `VOR_GLOBAL_LIGHT=hdr` | startet im HDR-Modus |
| `VOR_HDR=<absoluter Pfad>` | lädt beim Start ein Radiance-HDR-Environment |
| `VOR_PBR_COMPARISON=1` | startet mit der vollständigen PBR-Materialvergleichsszene |
| `VOR_DEBUG_VIEW=0..19` | wählt Beauty oder eine Material-/Pfaddiagnose |
| `VOR_TEST_FRAMES=<Anzahl>` | beendet einen automatischen GPU-Smoke-Test nach der angegebenen Framezahl |

## Tests und Diagnose

Die Tests decken unter anderem Layouts, Materialkonvertierung, Texturfarbräume und -deduplizierung, Szenenstatistiken, meshoptimizer-Verarbeitung, OBJ-/FBX-Import, HDR-Importance-Verteilungen, Fresnel/TIR, Beer-Lambert, Henyey-Greenstein-Normalisierung, White-Furnace-Energiegrenzen und NaN/Inf-Sweeps ab. Debug-Builds aktivieren Vulkan Validation; CUDA-, OptiX- und Vulkan-Diagnosen erscheinen auf `stderr`.

Der Slang-Hinweis `E38040` beim OptiX-Build ist erwartet: Der Raygen-Parameter wird absichtlich als Uniform im SBT-Raygen-Record abgelegt.

## Bewusste Grenzen

- LOD-Stufen werden erzeugt, derzeit wird aber nur das Basis-LOD hochgeladen und gerendert.
- Vulkan bleibt nicht-progressiv: Transmission, Subsurface und Volumen verwenden klar gekennzeichnete Echtzeitapproximationen; der vollständige stochastische Transport liegt im OptiX-Pfad.
- Volumen sind homogen; heterogene Dichtefelder und mehrschichtige Hautmodelle sind nicht Bestandteil dieses Updates.
- Das Farbmanagement verwendet ein kompaktes Tone Mapping mit sRGB-Ausgabe; ein auswählbarer Filmic-Tone-Mapper bleibt ein möglicher Ausbau.
