# Implementierungsplan und technischer Stand

Stand: 12. August 2026

Dieses Dokument beschreibt den implementierten Stand des Projekts und grenzt ihn von den noch offenen Ausbauschritten ab. Die detaillierten Frameabläufe sind in [RENDERPFADE.md](RENDERPFADE.md) grafisch dargestellt. Der spätere Ausbau des Material- und BSDF-Systems ist in [PBR_SHADER_UPDATE.md](PBR_SHADER_UPDATE.md) festgehalten.

## 1. Zielbild

VORaytracer ist eine native Windows-Anwendung in C++20 mit Dear ImGui und zwei zur Laufzeit umschaltbaren PBR-Renderpfaden:

1. Vulkan rendert in Echtzeit über Task-/Mesh-Shader und ergänzt Schatten sowie Reflexionen mit Inline Ray Queries.
2. NVIDIA OptiX rendert dieselbe Szene als progressiven Pathtracer.

Beide Backends teilen Kamera, Szenendaten, Materialwerte, Lichtkonfiguration und grundsätzliche GGX-BRDF-Logik. Vulkan besitzt immer Fenster, Swapchain, ImGui und Präsentation.

## 2. Entwicklungsumgebung

| Komponente | Konfiguration |
|---|---|
| Plattform | Windows x64 |
| Sprache | C++20 |
| Projekt | Visual-Studio-Solution, Debug und Release |
| Vulkan SDK | `G:\CodingLibraries\VulkanSDK` |
| Slang | `G:\CodingLibraries\slang-14.1` |
| Assimp | `G:\CodingLibraries\assimp\out-v145` |
| GLFW | `G:\CodingLibraries\glfw-3.5.1` |
| Dear ImGui | `G:\CodingLibraries\imgui` |
| meshoptimizer | `G:\CodingLibraries\meshoptimizer` |
| stb_image | `G:\CodingLibraries\stb\stb_image.h` |
| CUDA | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| OptiX | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |

Assimp wird aus dem globalen Bibliotheksverzeichnis eingebunden. Die frühere lokale Assimp-Kopie ist nicht mehr erforderlich. Release-Shader werden mit Optimierung, Debug-Shader ohne Release-Optimierung kompiliert.

## 3. Solution-Struktur

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

- `App`: Hauptschleife, ImGui, Eingabe, Dateidialoge und Backend-Wechsel
- `Core`: Mathematik, Szene, Kamera, Licht, Material- und Renderparameter
- `Assets`: Assimp-Import, HDR-Laden und meshoptimizer-Aufbereitung
- `Vulkan`: Vulkan-Kontext, Mesh-Shader, Ray Queries, Acceleration Structures, Interop und Präsentation
- `Optix`: CUDA-/OptiX-Kontext, Pathtracer, Denoiser-Dienst und External-Memory-Interop
- `Shaders`: Slang-Quellen und SPIR-V-/PTX-Buildregeln
- `Tests`: CPU- und Importtests

## 4. Asset- und Materialpipeline

### Implementiert

1. Ein Modell wird über `IFileOpenDialog`, einen direkten Pfad oder `VOR_SCENE` ausgewählt.
2. Assimp importiert Meshes, Node-Transformationen, Instanzen und PBR-Materialwerte.
3. Fehlende oder ungültige Basisdaten werden in ein kanonisches Vertex-/Indexformat überführt.
4. `Default Plastic` wird zusätzlich zu den importierten Materialien resident gehalten. Der UI-Schalter setzt eine allgemeine `materialOverrideId` auf dieses Material:
   - Albedo: `0.75, 0.75, 0.75`
   - Metallic: `0.0`
   - Roughness: `0.5`
   Die Originalmaterialien, Texturreferenzen und Mesh-Material-IDs bleiben unverändert. Beim Umschalten werden weder das Modell neu importiert noch Geometrie oder Acceleration Structures neu aufgebaut.
5. meshoptimizer erzeugt das Vulkan-Meshlet-Layout. Bei aktivierter Optimierung werden zusätzlich Remap, Vertex Cache, Overdraw, Vertex Fetch, Bounds und LODs berechnet.
6. Nach erfolgreichem Laden wird die Kamera anhand der Welt-Bounds so positioniert, dass das vollständige Modell sichtbar ist.
7. Der Szenenupload erneuert beide Backends und setzt gegebenenfalls die OptiX-Akkumulation zurück.

Die Meshlet-Erzeugung ist zwingend und bleibt auch bei deaktiviertem meshoptimizer-Schalter aktiv. Das Umschalten des Schalters oder des Standardmaterials lädt das aktuelle Modell neu.

### HDR-Environments

- Radiance-`.hdr`-Dateien werden über `stb_image` als lineare Float-Daten geladen.
- Eine vollständige Mip-Pyramide wird erzeugt und für diffuse sowie spiegelnde Image-Based Lighting verwendet.
- Material-Roughness wählt den Filtergrad beziehungsweise Mip-Level; die HDR-Datei verändert Roughness nicht.
- Richtungslicht ist der Standard beim Programmstart. Alternativ stehen prozeduraler Himmel und HDR-Environment bereit.
- Die drei Lichtmodi verwenden soweit anwendbar dieselbe Maus-gesteuerte Orientierung beziehungsweise Transformation.

### Noch offen

- Importierte Texturpfade und Materialmetadaten sind vorhanden, aber die vollständige Base-Color-, Roughness-, Metallic-, Normal-, AO- und Emissive-Texturabtastung muss in beiden Backends vereinheitlicht werden.
- LOD-Stufen werden generiert, aktuell wird nur das Basis-LOD hochgeladen und ausgewählt.
- Ein persistenter Asset-Cache ist noch nicht implementiert.

## 5. Vulkan-Renderpfad

Vulkan ist absichtlich kein progressiver Pathtracer. Der Renderer produziert pro Frame ein vollständiges Bild. `Samples per frame` und `Max bounces` gelten nur für OptiX.

### Geometrie und Mesh-Shader

- Geometrie wird über Staging in device-local Buffer übertragen.
- Der Task-/Amplification-Shader verarbeitet Gruppen von Meshlets und kompaktiert die sichtbaren Einträge.
- Frustum-Culling verwendet die vorhandenen Meshlet-Bounds.
- Normal-Cone-Culling verwirft abgewandte Meshlets. Beide Culling-Verfahren greifen, wenn die optionale Bounds-Berechnung aktiv war.
- Der Mesh-Shader liest die kompaktierten Meshlets, führt Vertex Pulling aus und gibt Vertices sowie Primitive aus.
- `Meshlet debug colors` erzeugt eine stabile pseudozufällige Farbe pro Meshlet.

### PBR und Ray Queries

- Der Fragmentshader verwendet einen Metallic-Roughness-Workflow mit GGX, Smith-Geometrie und Fresnel-Schlick.
- Eine TLAS steht dem Fragmentshader für Inline Ray Queries zur Verfügung.
- Schatten werden gegen die Vulkan-Acceleration-Structure getestet.
- `Ray-traced reflections` aktiviert eine zusätzliche Reflexionsabfrage und wertet Material und Schatten am Treffer aus.
- Globale Beleuchtung kommt je nach UI-Auswahl von Richtungslicht, prozeduralem Himmel oder HDR-Environment.
- Das HDR-Environment wird für die Lichtberechnung selbst verwendet und nicht nur als sichtbarer Hintergrund.

### Acceleration Structures

- Eine BLAS wird pro eindeutiger Vulkan-Geometrie angelegt.
- Die TLAS bildet Szeneninstanzen mit ihren Transformationen ab.
- BLAS- und TLAS-Builds werden gebündelt; getrennte `vkQueueWaitIdle()`-Stopps zwischen den Stufen wurden entfernt.
- Größenvergleiche in den Resize-Pfaden vermeiden unnötige Ressourcen-Neuanlagen.

### Ausgabe und optionales Denoising

1. Ohne Denoiser führt der Fragmentshader Tone Mapping aus und rendert direkt ins Swapchain-Image.
2. Mit Denoiser rendert Vulkan zunächst linear in ein `RGBA16F`-Offscreenbild.
3. Dieses Bild wird in einen dauerhaft bereitgestellten, exportierbaren `HALF4`-Buffer kopiert.
4. Ein externes Semaphore übergibt den Buffer an CUDA.
5. Der OptiX AI Denoiser arbeitet `HALF4` nach `HALF4`.
6. CUDA/OptiX führt Tone Mapping in einen gepackten `RGBA8`-Interop-Buffer aus.
7. Ein zweites Semaphore gibt die Ausgabe an Vulkan zurück.
8. Vulkan kopiert zur Swapchain, rendert ImGui und präsentiert.

Es gibt dabei weder synchrones `cuMemcpyDtoH` noch CPU-Tone-Mapping oder Map/Unmap des vollständigen Bildes pro Frame.

## 6. OptiX-Renderpfad

### Pipeline

- CUDA Driver Context und OptiX Device Context werden einmalig initialisiert.
- Slang erzeugt PTX für Raygen-, Miss- und Closest-Hit-Programme.
- Program Groups, Pipeline und Shader Binding Table werden beim Pipelineaufbau erstellt.
- SBT-Header werden nicht pro Frame neu gepackt oder übertragen.
- Pro Frame werden nur kleine Launch-Parameter wie Kamera, Licht, Samplezahl und maximale Pfadtiefe aktualisiert.

### Integrator

- progressives Multi-Sample-Pathtracing
- Metallic-Roughness-GGX
- direktes Licht und Environment Lighting
- Fresnel-gewichtete reflektierende Sekundärpfade
- maximale Pfadtiefe über `Max bounces`
- Russian Roulette für längere Pfade
- baryzentrisch interpolierte und korrekt transformierte Vertexnormalen
- Materialauflösung pro getroffenem Dreieck

Kamera-, Szenen-, Licht- und Materialänderungen setzen den Akkumulationszähler zurück. Der OptiX-Pfad besitzt keinen Denoiser.

### Ausgabe

OptiX akkumuliert in einem CUDA-`float4`-Buffer. Eine CUDA-/OptiX-Stufe führt Tone Mapping auf der GPU aus und schreibt direkt in einen Vulkan-kompatiblen External-Memory-Buffer. Externe Semaphoren synchronisieren den Vulkan-Kopiervorgang zur Swapchain. Ein vollständiges `cuCtxSynchronize()` im normalen Framepfad und synchrone CPU-Bildkopien sind entfernt.

### Noch offen

- Derzeit werden Szeneninstanzen für OptiX zu einem Triangle-GAS in Weltkoordinaten zusammengeführt.
- Nächster Geometrieschritt ist ein GAS pro eindeutiger Geometrie plus IAS-Instanzen. Das reduziert bei mehrfach instanzierten Modellen Speicherbedarf und Wiederaufbauzeit.
- Next Event Estimation und Multiple Importance Sampling können die Varianz bei HDR- und Flächenbeleuchtung weiter reduzieren.

## 7. CUDA/Vulkan-Interop und Synchronisation

### Implementiert

- Win32 External Memory für Vulkan-/CUDA-kompatible Ausgabebuffer
- External Semaphores für die Besitzübergabe zwischen den APIs
- dauerhafte CUDA-Importe und persistente Vulkan-Ressourcen
- keine vollständige Bildkopie über die CPU
- keine unnötige vollständige CUDA-Kontextsynchronisation pro Frame
- kleine Launch-Parameter-Updates statt vollständigem SBT-Neuaufbau
- Resize nur bei tatsächlich geänderter Bildgröße

### Weiteres Performance-Potenzial

- GPU-Timestamps für Mesh-Culling, Raster/PBR, Ray Queries, Denoising, OptiX-Launch, Interop-Kopie und Präsentation ergänzen.
- Frame-in-flight- und Timeline-Semaphore-Auslastung mit realen Messwerten prüfen.
- OptiX-GAS/IAS-Instanzierung umsetzen.
- GPU-selektiertes LOD und optionales Hi-Z-Occlusion-Culling ergänzen.
- Descriptor- und Pipeline-Caches anhand gemessener Hotspots optimieren.

## 8. ImGui und Interaktion

### Implementiert

- Backend-Auswahl: `Vulkan Mesh + Ray Query` oder `NVIDIA OptiX`
- native Dateidialoge für Modell und HDR
- automatische Szeneneinpassung sowie `Frame model`
- Orbit, Pan und Zoom
- Lichtbewegung mit denselben Gesten plus Umschalttaste
- Lichttyp und Lichtparameter
- Exposure, Samples pro Frame und Max Bounces
- Reflexionen und indirekte Beleuchtung
- Vulkan-Postrender-Denoiser
- meshoptimizer-Schalter mit Neuladen
- Default-Plastic-Schalter über allgemeine `materialOverrideId`, ohne Neuladen
- Bodenebene
- Meshlet-Debugfarben
- Szenen- und Rendererstatistiken

OptiX-spezifische Parameter werden im Vulkan-Pfad nicht fälschlich zur progressiven Integration verwendet.

## 9. Tests und Verifikation

### Vorhandene Tests

- Matrix- und Transformationsgrundlagen
- Szenenstatistiken und Bounds
- meshoptimizer-Meshlet-Erzeugung und Bounds
- OBJ-Import
- FBX-Import aus den Testassets
- Standardkunststoffmaterial
- Bodenebene
- HDR-Decodierung und erwartete Mip-Anzahl

### Build- und Laufzeitprüfung

- Debug- und Release-Solution bauen
- Testprogramm ausführen
- Debug-Anwendung mit Vulkan Validation starten
- Vulkan ohne und mit Denoiser prüfen
- OptiX starten und auf CUDA-/OptiX-Fehler kontrollieren
- Resize, Backend-Wechsel, Modellwechsel und HDR-Wechsel prüfen
- Materialvergleich zwischen Vulkan und OptiX mit identischer Kamera und Beleuchtung durchführen

## 10. Nächste Meilensteine

### M1: OptiX-Instanzierung

- Geometrien nach Mesh beziehungsweise eindeutigem Vertex-/Indexbestand deduplizieren.
- GAS pro eindeutiger Geometrie erstellen und nach Möglichkeit kompaktieren.
- IAS-Instanzen mit Node-Transformation und Materialzuordnung erzeugen.
- Aktualisierungspfad für reine Transformationsänderungen hinzufügen.
- Speicher- und Buildzeiten gegen den aktuellen Flattening-Pfad messen.

### M2: Vollständige PBR-Texturen

- Textur-Cache und eindeutige Residency pro Bild einführen.
- Base Color als sRGB, Datentexturen linear behandeln.
- Metallic/Roughness-Kanalbelegung aus Assimp zuverlässig übernehmen.
- Normal Mapping und Tangentenbasis in Vulkan und OptiX angleichen.
- Referenzszene für Materialparität aufbauen.

### M3: Sichtbarkeit und LOD

- vorhandene LODs vollständig hochladen.
- LOD-Auswahl im Task-Shader anhand projizierter Größe ergänzen.
- optional Hi-Z-Occlusion-Culling hinzufügen.
- sichtbare, verworfene und nach LOD sortierte Meshlets in GPU-Statistiken erfassen.

### M4: Beleuchtungsqualität

- OptiX Next Event Estimation für relevante Lichtquellen ergänzen.
- Multiple Importance Sampling für BSDF- und Environment-Sampling hinzufügen.
- Energie- und Farbmanagement beider Pfade anhand von Referenzmaterialien angleichen.
- optional auswählbares Filmic-Tone-Mapping ergänzen.

### M5: Messbare Performance

- GPU-Timestamp-Queries pro Renderpass integrieren.
- CPU- und GPU-Framezeiten getrennt anzeigen.
- Interop-Wartezeiten und Auslastung mehrerer Frames in flight messen.
- Optimierungen ausschließlich gegen reproduzierbare Szenen und Messreihen bewerten.

## 11. Abnahmekriterien

Der aktuelle Kernumfang gilt als erfüllt, wenn:

1. Debug und Release als x64/C++20 gebaut werden;
2. OBJ-, FBX- und weitere Assimp-Modelle über einen nativen Dateidialog geladen werden;
3. die Kamera das Modell nach dem Laden vollständig einpasst;
4. Task-/Mesh-Shader mit Meshlet-Culling die Szene rendern;
5. Vulkan-Schatten und optionale Reflexionen per Ray Query berechnet;
6. HDR, prozeduraler Himmel und Richtungslicht korrekt zur Beleuchtung beitragen;
7. OptiX dieselbe Szene progressiv mit PBR-Materialwerten rendert;
8. der Backend-Wechsel ohne Neustart funktioniert;
9. der optionale Denoiser ausschließlich als Vulkan-Postrender-Schritt läuft;
10. Interop und Tone Mapping ohne vollständigen CPU-Bildpfad auskommen;
11. Resize, Szenenwechsel und Licht-/Kameranavigation stabil bleiben;
12. Tests sowie Debug-/Release-Builds ohne neue Fehler durchlaufen.

Die noch offenen Meilensteine erweitern Qualität und Performance, ohne diese bereits funktionierende Architektur grundsätzlich zu ändern.
