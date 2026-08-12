# PBR-Shader-Updateplan

Stand: 12. August 2026

Dieses Dokument hält den vollständigen Ausbauplan für das gemeinsame PBR-Material- und BSDF-System von Vulkan und NVIDIA OptiX fest. Es ist als Arbeitsgrundlage für eine spätere, zusammenhängende Implementierung gedacht. Die bestehenden Renderpfade sind in [RENDERPFADE.md](RENDERPFADE.md) beschrieben.

## 1. Ziel

Das bestehende Metallic-Roughness-Modell wird zu einem modularen, energieerhaltenden Materialsystem erweitert. Vulkan und OptiX sollen dieselben Slang-Funktionen für Materialdekodierung, BSDF-Auswertung und – soweit für den jeweiligen Renderpfad sinnvoll – BSDF-Sampling verwenden.

Geplante Materialeigenschaften:

- Metallic-Roughness
- Dielectric und Conductor
- Clearcoat
- Transmission, Glas und Absorption
- Anisotropie
- Sheen und Cloth
- Emission und emissive Mesh-Lichter
- Subsurface Scattering
- volumetrische Medien

Vulkan bleibt ein nicht-progressiver Echtzeitpfad mit Mesh-Shadern und wenigen gezielten Ray Queries. OptiX bleibt der progressive Pathtracer und übernimmt die vollständige stochastische Abtastung komplexer BSDFs.

## 2. Architekturprinzipien

### Ein gemeinsames Materialsystem

Es werden keine vollständig getrennten Shader pro Materialtyp angelegt. Ein Material kombiniert stattdessen Loben und Merkmale über Parameter und Flags. Beispielsweise kann ein Material gleichzeitig eine dielektrische Basisschicht, Clearcoat, Sheen und Emission besitzen.

Die bereits vorhandene `materialOverrideId` bleibt die allgemeine, nichtdestruktive Laufzeitüberschreibung. Default Plastic ist nur eines der residenten Materialien, auf das diese ID zeigen kann; Originalzuweisungen werden beim Deaktivieren ohne Szenenreload wieder wirksam.

### Gemeinsame Slang-BSDF-Schicht

- Vulkan verwendet die gemeinsame BSDF-Auswertung für bekannte Blick- und Lichtrichtungen.
- OptiX verwendet zusätzlich BSDF-Sampling und PDFs für indirekte Pfade, Next Event Estimation und Multiple Importance Sampling.
- Backend-spezifisch bleiben Traversierung, Ressourcenbindung, Texturzugriff und Entry Points.

### Keine Materialpipeline pro Kombination

Die meisten Eigenschaften werden über Materialdaten gesteuert. Separate Vulkan-Pipelines sind nur für grundlegende Rasterzustände vorgesehen:

- undurchsichtig
- Alpha Mask
- Transmission/transparent
- doppelseitig

### Physikalische Konsistenz

- Beleuchtung und BSDF-Berechnung erfolgen linear.
- Alle Loben müssen zusammen energieerhaltend gewichtet werden.
- Geometrische Normale und Shading-Normale werden getrennt behandelt.
- PDFs müssen zur jeweiligen Sampling-Verteilung passen.
- Materialparameter dürfen nicht durch HDR- oder Denoiser-Einstellungen verändert werden.

## 3. Gemeinsames GPU-Materialformat

C++ und Slang erhalten binär kompatible Materialstrukturen. Die konkrete Ausrichtung wird über `static_assert`, Slang-Reflection oder explizite Layouttests abgesichert.

```cpp
enum class MaterialFlags : std::uint32_t
{
    None         = 0,
    DoubleSided  = 1 << 0,
    AlphaMask    = 1 << 1,
    Transmission = 1 << 2,
    Clearcoat    = 1 << 3,
    Sheen        = 1 << 4,
    Anisotropy   = 1 << 5,
    Emissive     = 1 << 6,
    Subsurface   = 1 << 7,
    Volume       = 1 << 8,
};

struct GpuMaterial
{
    Vec4 baseColorFactor;

    Vec3 emissiveFactor;
    float metallic;

    float roughness;
    float normalScale;
    float occlusionStrength;
    float alphaCutoff;

    float transmission;
    float indexOfRefraction;
    float clearcoat;
    float clearcoatRoughness;

    float anisotropy;
    float anisotropyRotation;
    float sheenRoughness;
    std::uint32_t flags;

    Vec3 sheenColor;
    float absorptionDistance;

    Vec3 absorptionColor;
    std::uint32_t baseColorTexture;

    std::uint32_t metallicRoughnessTexture;
    std::uint32_t normalTexture;
    std::uint32_t emissiveTexture;
    std::uint32_t occlusionTexture;

    std::uint32_t transmissionTexture;
    std::uint32_t clearcoatTexture;
    std::uint32_t clearcoatRoughnessTexture;
    std::uint32_t clearcoatNormalTexture;
};
```

Nicht vorhandene Texturen verwenden einen definierten ungültigen Index. Faktoren bleiben auch ohne Textur vollständig funktionsfähig.

## 4. Einheitliche Oberflächendaten

Nach Material- und Texturauflösung wird eine backendunabhängige Oberflächenstruktur erzeugt:

```slang
struct SurfaceData
{
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    float3 tangent;
    float3 bitangent;

    float3 baseColor;
    float3 emissive;
    float alpha;
    float metallic;
    float roughness;
    float occlusion;

    float transmission;
    float ior;
    float clearcoat;
    float clearcoatRoughness;

    float anisotropy;
    float anisotropyRotation;
    float3 sheenColor;
    float sheenRoughness;

    float3 absorptionColor;
    float absorptionDistance;
    uint flags;
};
```

Der Assimp-Importer konvertiert importierte Materialmodelle in dieses interne Format. Vulkan und OptiX dürfen danach keine voneinander abweichende Interpretation derselben Materialwerte besitzen.

## 5. Textursystem

### Import und CPU-Cache

- externe und eingebettete Assimp-Texturen unterstützen
- Pfade normalisieren und identische Bilder deduplizieren
- sRGB für Base Color und Emissive verwenden
- lineare Dekodierung für Metallic, Roughness, AO, Normalen und sonstige Datenmaps verwenden
- Kanalbelegung, UV-Set, UV-Transformation und Samplerzustand übernehmen
- fehlende Texturen durch definierte Fallbackwerte ersetzen

### Vulkan

- Descriptor Indexing beziehungsweise bindless Sampled Images verwenden
- Sampler deduplizieren
- Texturen über Staging in device-local Images übertragen
- Mipmaps erzeugen
- Deskriptoren nur bei Residency-Änderungen aktualisieren

### OptiX/CUDA

- CUDA Arrays, Mipmapped Arrays und Texture Objects verwenden
- dieselben Address-, Filter- und Farbraumregeln wie Vulkan verwenden
- Texture Objects pro eindeutigem Bild-/Samplerpaar cachen
- Texturindizes zwischen CPU-Material, Vulkan und OptiX stabil halten

### Erste Texturgruppe

1. Base Color
2. Metallic-Roughness
3. Normal
4. Ambient Occlusion
5. Emissive

Spezialisierte Clearcoat-, Transmission-, Sheen- und Volumentexturen folgen mit den jeweiligen Materialloben.

## 6. Slang-Modulstruktur

```text
shaders/Shared/
├── MaterialTypes.slang
├── MaterialSampling.slang
├── Fresnel.slang
├── Microfacet.slang
├── BsdfTypes.slang
├── BsdfDiffuse.slang
├── BsdfConductor.slang
├── BsdfDielectric.slang
├── BsdfClearcoat.slang
├── BsdfTransmission.slang
├── BsdfAnisotropic.slang
├── BsdfSheen.slang
├── BsdfSubsurface.slang
├── PhaseFunctions.slang
└── Pbr.slang
```

`Pbr.slang` bleibt die stabile öffentliche Schnittstelle für beide Backends und kombiniert die internen Module.

## 7. Zentrale BSDF-Schnittstelle

```slang
enum BsdfLobe : uint
{
    DiffuseLobe,
    SpecularReflectionLobe,
    SpecularTransmissionLobe,
    ClearcoatLobe,
    SheenLobe,
    SubsurfaceLobe,
};

struct BsdfSample
{
    float3 direction;
    float3 weight;
    float pdf;
    uint lobe;
    bool transmission;
    bool delta;
};

float3 evaluateBsdf(
    SurfaceData surface,
    float3 viewDirection,
    float3 lightDirection);

float evaluateBsdfPdf(
    SurfaceData surface,
    float3 viewDirection,
    float3 lightDirection);

BsdfSample sampleBsdf(
    SurfaceData surface,
    float3 viewDirection,
    float2 directionSample,
    float lobeSample);
```

Anforderungen:

- `evaluateBsdf()` und `evaluateBsdfPdf()` müssen dieselbe Mischung von Loben verwenden.
- `sampleBsdf()` muss das ausgewählte Loben-PDF und das Gesamt-PDF korrekt berücksichtigen.
- Delta-Loben werden kenntlich gemacht und bei MIS entsprechend behandelt.
- Ungültige Parameter werden beim Import normalisiert, nicht versteckt pro Shaderaufruf repariert.

## 8. Metallic-Roughness, Dielectric und Conductor

Das bestehende Modell wird als verifizierte Basis konsolidiert:

- GGX/Trowbridge-Reitz-Normalverteilung
- Smith-Masking und -Shadowing
- Fresnel-Schlick für die Echtzeitbasis
- exaktes dielektrisches Fresnel dort, wo Transmission oder IOR erforderlich ist
- diffuse Lambert- oder optional Burley-Komponente
- energieerhaltende diffuse/specular Gewichtung

Interpretation:

- `metallic = 0`: Dielectric
- `metallic = 1`: Conductor im Metallic-Roughness-Workflow
- Zwischenwerte: Texturfilterung beziehungsweise Materialmischung

Für höhere Leitergenauigkeit kann später optional komplexes IOR mit Eta und K ergänzt werden. Zunächst bleibt die glTF-kompatible gefärbte F0-Interpretation maßgeblich.

## 9. Clearcoat

Clearcoat wird als zusätzliche dielektrische Mikrofacetten-Lobe über der Basis implementiert:

- eigener Gewichtungsfaktor
- eigene Roughness
- fester Standard-IOR um 1,5
- später eigene Normal Map
- Abschwächung der Basisschicht durch Clearcoat-Fresnel

Vulkan wertet Basis und Clearcoat deterministisch aus. OptiX nimmt Clearcoat in Loben-Auswahl, Sampling und Gesamt-PDF auf.

Abnahmekriterium: Eine lackierte dielektrische und eine lackierte metallische Referenzkugel reagieren in beiden Backends plausibel und ohne Energiegewinn.

## 10. Transmission, Glas und Absorption

### Oberflächenphysik

- exaktes dielektrisches Fresnel
- Reflexion und Brechung nach Snellius
- Erkennung vollständiger interner Reflexion
- korrekte Eta-Skalierung beim Radiancetransport
- raue Transmission über eine Mikrofacetten-BTDF

### Mediumzustand

OptiX erhält einen expliziten Medium- beziehungsweise IOR-Stack für verschachtelte transparente Objekte. Mindestens Luft und ein geschlossenes Objekt müssen korrekt unterstützt werden.

### Absorption

Beer-Lambert:

```slang
float3 transmittance = exp(-absorptionCoefficient * distanceInMedium);
```

`absorptionColor` und `absorptionDistance` werden beim Import in einen Absorptionskoeffizienten umgerechnet.

### Vulkan-Strategie

Der Echtzeitpfad beginnt mit einer begrenzten Ray-Query-Transmission für Eintritt und nächsten Austritt. Komplexe verschachtelte Brechung, Mehrfachpfade und Dispersion bleiben OptiX vorbehalten. Eine UI- oder Debugkennzeichnung soll sichtbar machen, wenn Vulkan eine vereinfachte Transmission verwendet.

## 11. Anisotropie

- anisotropes GGX mit `alphaX` und `alphaY`
- Auswertung und VNDF-Sampling in Tangent-/Bitangent-Basis
- Rotationsparameter um die Shading-Normale
- zuverlässiger Assimp-Tangentenimport beziehungsweise MikkTSpace-kompatible Erzeugung
- Entartungen bei fehlenden oder gespiegelten UVs robust behandeln

Referenzmaterialien: gebürstetes Aluminium, Kupfer und eine Schallplattenoberfläche.

## 12. Sheen und Cloth

- Charlie-NDF oder ein kompatibles glTF-Sheen-Modell
- Sheen Color und Sheen Roughness
- energieerhaltende Reduktion der Basisschicht
- konsistente Evaluation, Sampling und PDF in OptiX
- Vulkan-Auswertung für direktes Licht und Environment

Ein komplexeres Faser-/Mehrstreumodell kann später ergänzt werden; der erste Ausbau soll glTF-kompatibel und zwischen den Backends vergleichbar sein.

## 13. Emission und Mesh-Lichter

Emission wird zunächst bei Oberflächentreffern zur Radiance addiert. Für wirksame indirekte Beleuchtung werden emissive Dreiecke zusätzlich als explizite Lichtquellen erfasst:

1. emissive Dreiecke beim Szenenupload sammeln
2. Leistung aus Fläche, Emissionsfarbe und Intensität berechnen
3. diskrete Verteilung beziehungsweise Alias-Tabelle aufbauen
4. Dreieck nach Leistung und Punkt auf der Fläche abtasten
5. Geometrie-PDF in Solid-Angle-PDF umrechnen
6. Sichtbarkeit per Shadow Ray prüfen
7. BSDF- und Licht-Sampling mit MIS kombinieren

Transformations- oder Materialänderungen aktualisieren die Lichtverteilung nur bei Bedarf.

## 14. OptiX Next Event Estimation und MIS

Der OptiX-Integrator wird vor den komplexeren Loben erweitert:

- direkte Abtastung von Richtungslicht
- importance-gewichtete HDR-Environment-Abtastung
- Abtastung emissiver Mesh-Lichter
- BSDF-Sampling
- Balance- oder Power-Heuristic für MIS
- korrekte Behandlung von Delta-Lichtern und Delta-BSDFs
- Russian Roulette auf Basis des Pfaddurchsatzes

Diese Stufe ist Voraussetzung für belastbare Materialvergleiche, weil sie Varianz besonders bei rauen Materialien, HDR-Beleuchtung und kleinen Leuchtflächen reduziert.

## 15. Subsurface Scattering

Die Implementierung erfolgt in zwei Stufen:

### Stufe A: Echtzeit/Vulkan

- screen- oder thickness-basierte Approximation
- Materialparameter für Streufarbe und charakteristische Distanz
- klar als Approximation abgegrenzter Beitrag

### Stufe B: OptiX

- zunächst Random-Walk-Subsurface oder ein begrenztes Diffusionsmodell
- Eintritt, interner Transport und Austritt mit konsistentem Durchsatz
- robustes Ray Origin Offset und Selbstschnitt-Vermeidung
- Russian Roulette für lange interne Wege

Referenzmaterialien: Wachs, Marmor und Haut-Testmaterial. Ein vollständiges Hautmodell mit mehreren Schichten ist ein späterer Spezialausbau.

## 16. Volumetrische Medien

### Datenmodell

- Absorptionskoeffizient `sigmaA`
- Streukoeffizient `sigmaS`
- Extinktion `sigmaT`
- Emission
- Phasenfunktion und Asymmetrieparameter
- homogene Medien zuerst, heterogene Medien später

### OptiX

- freie Weglänge sampeln
- Medium-Interaktion gegen nächsten Oberflächentreffer vergleichen
- Henyey-Greenstein-Phasenfunktion auswerten und sampeln
- direktes Licht am Volumenereignis mit MIS berücksichtigen
- Medium-Stack mit Oberflächentransmission verbinden

### Vulkan

Der erste Vulkan-Ausbau bleibt auf einfache homogene Fog-/Aerial-Perspective-Approximationen begrenzt. Vollständiges volumetrisches Pathtracing gehört in den OptiX-Pfad.

## 17. Einbindung in Vulkan

```text
Task-/Mesh-Shader
    ↓
Material- und Texturindizes
    ↓
Fragmentshader
    ↓
SurfaceData aus Faktoren und Texturen
    ↓
evaluateBsdf()
    ├── Richtungslicht
    ├── HDR-/Sky-Environment
    ├── Ray-Query-Schatten
    ├── Ray-Query-Reflexion
    └── begrenzte Ray-Query-Transmission
```

Vulkan-spezifische Anforderungen:

- keine progressive Akkumulation einführen
- Materialflags möglichst kohärent verarbeiten
- komplexe Loben anhand gemessener Kosten optimieren
- Texturdeskriptoren bindless und stabil halten
- Denoiser ausschließlich als optionalen Postrender-Schritt beibehalten
- Debugansichten für einzelne Loben und Texturkanäle ergänzen

## 18. Einbindung in OptiX

Der Closest-Hit-Code liefert Geometrie- und Materialdaten. Der Raygen-Integrator führt anschließend die gemeinsame Materialauswertung aus:

```text
Treffer
    ↓
SurfaceData laden
    ↓
Emission addieren
    ↓
Direktes Licht über NEE/MIS
    ↓
BSDF-Lobe wählen und sampeln
    ↓
Durchsatz, Mediumzustand und Ray aktualisieren
    ↓
Russian Roulette
```

OptiX-spezifische Anforderungen:

- separate GAS/IAS-Instanzen vor oder zusammen mit dem Texturausbau umsetzen
- Primitive-, Material- und Instanz-IDs stabil halten
- PDFs, Eta-Skalierung und Mediumübergänge testen
- Akkumulation bei jeder relevanten Material-/Texturänderung zurücksetzen
- kein Denoiser im OptiX-eigenen Renderpfad

## 19. UI und Debugansichten

### Materialeditor

- Materialauswahl pro Mesh
- Faktoren und Flags bearbeiten
- Texturzuweisungen anzeigen
- IOR, Clearcoat, Transmission, Anisotropie, Sheen und Absorption bearbeiten
- Änderungen ohne vollständigen Szenenimport an GPU-Daten übertragen
- OptiX-Akkumulation gezielt zurücksetzen

### Debugansichten

- Base Color
- Metallic
- Roughness
- Shading- und geometrische Normale
- Tangent-/Bitangent-Richtung
- AO und Emissive
- Diffuse-, Specular-, Clearcoat-, Sheen- und Transmission-Lobe
- BSDF-PDF
- Material- und Texturindex
- Medium-ID und Pfadtiefe im OptiX-Diagnosemodus

## 20. Tests

### CPU- und Layouttests

- C++-/Slang-Strukturgrößen und Offsets
- Assimp-Materialkonvertierung
- sRGB-/Linear-Zuordnung
- Kanalbelegung und Fallbacktexturen
- Texturdeduplizierung
- Absorptionsparameter

### Mathematische Shadertests

- Fresnel bei senkrechtem und streifendem Einfall
- GGX-Auswertung und Grenzfälle
- Evaluation/PDF/Sampling-Konsistenz
- Energieerhaltung über numerische hemisphärische Integration
- anisotrope Verteilung
- Totalreflexion
- Beer-Lambert
- Phasenfunktionen
- keine NaN-/Inf-Ergebnisse bei zulässigen Eingaben

### Bild- und GPU-Tests

- Materialkugelraster für alle Loben
- Vulkan-/OptiX-Vergleich mit identischer Kamera und Beleuchtung
- weißes Furnace-Test-Environment zur Energieprüfung
- HDR-Environment mit mehreren Roughness-Stufen
- Clearcoat auf Metall und Dielectric
- dünnes und massives Glas mit Absorption
- anisotropes Metall
- Stoff/Sheen
- emissive Fläche in einer geschlossenen Referenzszene
- Subsurface- und Volumenreferenzen

Vergleiche sollen nicht bitgenau sein. Erwartet werden gleiche Materialparameter, plausibel übereinstimmende Mittelwerte und keine systematischen Energie- oder Farbabweichungen.

## 21. Performance und Speicher

- Texturen nach Inhalt und Samplerzustand deduplizieren
- Uploads bündeln und device-local beziehungsweise CUDA-resident halten
- keine Textur- oder Materialkopien pro Frame
- Materialbuffer nur in geänderten Bereichen aktualisieren
- Shader-Loben-Auslastung und Divergenz mit GPU-Timestamps beziehungsweise NVIDIA-Profilern messen
- emissive Lichtverteilungen nur bei relevanten Änderungen neu aufbauen
- OptiX-GAS/IAS-Instanzierung zur Vermeidung duplizierter Geometrie umsetzen
- Vulkan-Shader-Varianten nur bei nachweisbarem Gewinn einführen

## 22. Meilensteine

### M1: Gemeinsames Fundament

- `GpuMaterial`, Flags und `SurfaceData`
- Layouttests
- modulare Slang-Struktur
- bestehendes Metallic-Roughness-Modell migrieren
- Debug-/Release-Build und vorhandene Bildparität erhalten

### M2: Vollständige Basistexuren

- CPU-Texturcache
- Vulkan bindless Images
- CUDA Texture Objects
- Base Color, Metallic-Roughness, Normal, AO und Emissive
- Import- und Referenztests

### M3: BSDF-API und OptiX-Lichtsampling

- `evaluateBsdf`, `sampleBsdf`, `evaluateBsdfPdf`
- GGX-VNDF-Sampling
- Next Event Estimation
- HDR-Importance-Sampling
- MIS

### M4: Clearcoat

- Evaluation, Sampling und PDF
- Texturen und Normal Map
- Vulkan-/OptiX-Referenzvergleich

### M5: Transmission und Glas

- Reflexion, Brechung und Totalreflexion
- raue BTDF
- Medium-Stack
- Beer-Lambert-Absorption
- begrenzte Vulkan-Transmission

### M6: Anisotropie

- Tangentenpipeline
- anisotropes GGX
- VNDF-Sampling
- Referenzmaterialien

### M7: Sheen und Cloth

- Charlie-/glTF-Sheen
- Energiekompensation
- gemeinsame Backendtests

### M8: Emissive Mesh-Lichter

- Dreieckslichtliste
- Leistungsverteilung
- Flächensampling
- MIS mit BSDF und Environment

### M9: Subsurface Scattering

- Vulkan-Approximation
- OptiX Random Walk beziehungsweise Diffusionsmodell
- Referenztests

### M10: Volumen

- Mediumparameter und Medium-Stack
- homogene Absorption und Streuung
- Phasenfunktion und direktes Licht
- einfache Vulkan-Fog-Approximation

### M11: Qualität und Performance

- vollständige Materialvergleichsszene
- Energie- und NaN-Prüfungen
- GPU-Profiling
- Speicher- und Descriptorprüfung
- Dokumentation und UI-Hilfe aktualisieren

## 23. Abnahmekriterien

Der vollständige PBR-Ausbau gilt als abgeschlossen, wenn:

1. Vulkan und OptiX dasselbe CPU- und GPU-Materialmodell verwenden;
2. alle Basismaterialtexturen in beiden Backends korrekt geladen und abgetastet werden;
3. Metallic-Roughness, Clearcoat, Transmission, Anisotropie und Sheen über gemeinsame Slang-Funktionen ausgewertet werden;
4. OptiX für alle nicht-deltaförmigen Loben konsistente Evaluation, Sampling und PDFs besitzt;
5. Next Event Estimation und MIS Richtungslicht, HDR und emissive Mesh-Lichter berücksichtigen;
6. Glas IOR, Totalreflexion, Mediumübergänge und Absorption korrekt behandelt;
7. Subsurface Scattering und homogene Volumen im OptiX-Pfad funktionsfähig sind;
8. Vulkan dokumentierte Echtzeitapproximationen für Transmission, Subsurface und Volumen besitzt;
9. Materialänderungen beide Backends aktualisieren und nur notwendige Akkumulationen beziehungsweise GPU-Daten zurücksetzen;
10. Debug- und Release-Build sowie automatische Tests erfolgreich sind;
11. keine neuen Vulkan-, CUDA- oder OptiX-Validierungsfehler auftreten;
12. Furnace-, Referenzbild- und NaN-/Inf-Tests keine systematischen Energiefehler zeigen;
13. Texturen und Materialien nicht pro Frame neu alloziert oder zur CPU zurückkopiert werden;
14. Dokumentation, UI-Beschriftungen und Renderpfaddiagramme dem implementierten Stand entsprechen.

## 24. Empfohlene Ausführung

Der Plan soll in der angegebenen Meilensteinreihenfolge umgesetzt werden. Jeder Meilenstein endet mit:

1. Debug-Build
2. Release-Build
3. automatischen Tests
4. mindestens einer visuellen Referenzprüfung in Vulkan und OptiX
5. Prüfung auf Validation-, CUDA- und OptiX-Fehler
6. Aktualisierung dieses Dokuments mit dem tatsächlichen Umsetzungsstand

Die Arbeit darf über mehrere automatische Fortsetzungen oder Kontextkompaktierungen laufen. Der Zustand wird über dieses Dokument, Tests und den Quellbestand festgehalten; bereits verifizierte Meilensteine werden nicht ohne konkreten Grund erneut umgesetzt.
