#include "Core/Math.h"
#include "Core/Scene.h"
#include "Assets/AssetLoader.h"

#include <meshoptimizer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

int main()
{
    using namespace vor;

    int failures = 0;
    const auto require = [&](bool condition, std::string_view message) {
        if (!condition)
        {
            std::cerr << "FAILED: " << message << '\n';
            ++failures;
        }
    };

    const Mat4 identity = Mat4::identity();
    const Mat4 product = identity * identity;
    require(product.m == identity.m, "identity matrix multiplication");

    require(sizeof(GpuMaterial) == 224, "shared GPU material size");
    require(alignof(GpuMaterial) == 16, "shared GPU material alignment");
    require(offsetof(GpuMaterial, materialFlags) == 160, "shared GPU material flag offset");
    require(offsetof(GpuMaterial, textureIndices0) == 176, "shared GPU material texture offset");
    Material layoutMaterial{};
    layoutMaterial.baseColor = {0.2f, 0.3f, 0.4f, 0.8f};
    layoutMaterial.metallic = 0.6f;
    layoutMaterial.roughness = 0.35f;
    layoutMaterial.clearcoat = 0.7f;
    layoutMaterial.transmission = 0.25f;
    layoutMaterial.baseColorTexture = 3;
    layoutMaterial.clearcoatTexture = 4;
    layoutMaterial.clearcoatRoughnessTexture = 5;
    layoutMaterial.clearcoatNormalTexture = 6;
    layoutMaterial.absorptionColor = {0.7f, 0.8f, 0.9f};
    layoutMaterial.absorptionDistance = 2.5f;
    layoutMaterial.anisotropy = 0.7f;
    layoutMaterial.anisotropyRotation = 0.2f;
    layoutMaterial.anisotropyTexture = 7;
    const GpuMaterial layoutGpu = layoutMaterial.toGpu();
    require(layoutGpu.baseColorFactor.x == 0.2f && layoutGpu.emissiveAndMetallic.w == 0.6f,
            "shared GPU material factors");
    require(layoutGpu.surfaceParameters.x == 0.35f && layoutGpu.transmissionClearcoat.z == 0.7f,
            "shared GPU material lobes");
    require(layoutGpu.transmissionClearcoat.x == 0.25f && layoutGpu.transmissionClearcoat.y == 1.5f &&
                layoutGpu.absorptionColorSubsurface.x == 0.7f &&
                layoutGpu.sheenColorAbsorptionDistance.w == 2.5f,
            "transmission, IOR, and absorption GPU factors");
    require(layoutGpu.anisotropySheen.x == 0.7f && layoutGpu.anisotropySheen.y == 0.2f &&
                layoutGpu.textureIndices2[3] == 7u,
            "anisotropy factors and GPU texture ID");
    require(layoutGpu.textureIndices0[0] == 3u && layoutGpu.textureIndices0[2] == kInvalidTextureId,
            "shared GPU material texture IDs");
    require(layoutGpu.textureIndices1[2] == 4u && layoutGpu.textureIndices1[3] == 5u &&
                layoutGpu.textureIndices2[0] == 6u,
            "clearcoat GPU texture IDs");
    require(hasFlag(layoutMaterial.flags(), MaterialFlags::Clearcoat) &&
                hasFlag(layoutMaterial.flags(), MaterialFlags::Transmission),
            "shared GPU material flags");

    Scene scene{};
    Mesh mesh{};
    mesh.lods.emplace_back();
    mesh.lods.front().indices = {0, 1, 2, 0, 2, 3};
    mesh.lods.front().meshlets.emplace_back();
    scene.meshes.push_back(mesh);
    require(scene.triangleCount() == 2, "scene triangle count");
    require(scene.meshletCount() == 1, "scene meshlet count");

    constexpr std::array<float, 12> positions{
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
    };
    constexpr std::array<std::uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    constexpr std::size_t maxVertices = 64;
    constexpr std::size_t maxTriangles = 124;
    const std::size_t bound = meshopt_buildMeshletsBound(indices.size(), maxVertices, maxTriangles);
    std::vector<meshopt_Meshlet> meshlets(bound);
    std::vector<std::uint32_t> meshletVertices(bound * maxVertices);
    std::vector<std::uint8_t> meshletTriangles(bound * maxTriangles * 3);
    const std::size_t count = meshopt_buildMeshlets(
        meshlets.data(), meshletVertices.data(), meshletTriangles.data(), indices.data(), indices.size(),
        positions.data(), 4, sizeof(float) * 3, maxVertices, maxTriangles, 0.5f);
    require(count == 1, "meshoptimizer meshlet count");
    require(meshlets.front().vertex_count == 4, "meshoptimizer vertex count");
    require(meshlets.front().triangle_count == 2, "meshoptimizer triangle count");

    const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
        meshletVertices.data(), meshletTriangles.data(), meshlets.front().triangle_count,
        positions.data(), 4, sizeof(float) * 3);
    require(bounds.radius > 1.0f, "meshoptimizer bounds");

    const std::filesystem::path sourceFile = std::filesystem::path(__FILE__);
    const std::filesystem::path projectRoot = sourceFile.parent_path().parent_path().parent_path();
    AssetLoader assetLoader;
    AssetLoadResult imported = assetLoader.load(projectRoot / "assets" / "SampleTriangle.obj");
    require(static_cast<bool>(imported), "Assimp sample import");
    if (imported)
    {
        require(imported.scene.meshes.size() == 1, "imported mesh count");
        require(imported.scene.triangleCount() == 1, "imported triangle count");
        require(imported.scene.meshletCount() == 1, "imported meshlet count");
        require(!imported.scene.materials.empty(), "imported material count");
    }

    AssetLoadResult importedWithoutOptionalPasses = assetLoader.load(
        projectRoot / "assets" / "SampleTriangle.obj",
        {.enableOptionalMeshoptimizerPasses = false});
    require(static_cast<bool>(importedWithoutOptionalPasses), "Assimp import without optional meshoptimizer passes");
    if (importedWithoutOptionalPasses)
    {
        require(importedWithoutOptionalPasses.scene.meshletCount() == 1,
                "mandatory meshlet generation when optional passes are disabled");
        require(importedWithoutOptionalPasses.scene.meshes.front().lods.size() == 1,
                "optional LOD generation disabled");
        require(importedWithoutOptionalPasses.scene.meshes.front().lods.front().meshlets.front().boundingSphere.w == 0.0f,
                "optional meshlet bounds disabled");
    }

    AssetLoadResult texturedImport = assetLoader.load(projectRoot / "assets" / "TexturedTriangle.obj");
    require(static_cast<bool>(texturedImport), "textured Assimp sample import");
    if (texturedImport)
    {
        require(!texturedImport.scene.textures.empty(), "CPU texture cache populated");
        const TextureReference& texture = texturedImport.scene.textures.front();
        require(texture.valid(), "decoded texture and mip chain validation");
        require(texture.width == 2 && texture.height == 2 && texture.mipCount == 2,
                "texture dimensions and mip generation");
        require(texture.mipOffsets.size() == 2 && texture.mipOffsets[0] == 0 && texture.mipOffsets[1] == 16,
                "texture mip offsets");
        require(texture.rgba8Pixels.size() == 20, "texture mip byte count");
        const Material& texturedMaterial = texturedImport.scene.materials[
            texturedImport.scene.meshes.front().materialIndex];
        require(texturedMaterial.baseColorTexture >= 0, "base-color texture reference");
        require(texturedMaterial.emissiveTexture == texturedMaterial.baseColorTexture,
                "sRGB texture cache deduplication");
        require(texturedMaterial.normalTexture >= 0 && texturedMaterial.metallicRoughnessTexture >= 0 &&
                    texturedMaterial.occlusionTexture >= 0,
                "normal, metallic-roughness, and AO texture references");
        require(texturedMaterial.normalTexture == texturedMaterial.metallicRoughnessTexture &&
                    texturedMaterial.normalTexture == texturedMaterial.occlusionTexture,
                "linear texture cache deduplication");
        require(texturedImport.scene.textures.size() == 2, "color-space-aware texture cache entries");
        const GpuMaterial texturedGpu = texturedMaterial.toGpu();
        require(texturedGpu.textureIndices0[0] == static_cast<std::uint32_t>(texturedMaterial.baseColorTexture) &&
                    texturedGpu.textureIndices0[3] == static_cast<std::uint32_t>(texturedMaterial.emissiveTexture),
                "stable CPU/GPU texture indices");
        const Vec4 tangent = texturedImport.scene.meshes.front().vertices.front().tangent;
        require(std::abs(length(Vec3{tangent.x, tangent.y, tangent.z}) - 1.0f) < 1.0e-4f &&
                    std::abs(std::abs(tangent.w) - 1.0f) < 1.0e-4f,
                "normalized tangent and handedness import");
    }

    AssetLoadResult importedWithMaterialOverride = assetLoader.load(projectRoot / "assets" / "SampleTriangle.obj");
    require(static_cast<bool>(importedWithMaterialOverride), "Assimp import for material override");
    if (importedWithMaterialOverride)
    {
        Scene& overrideScene = importedWithMaterialOverride.scene;
        const std::size_t materialCountBeforeOverride = overrideScene.materials.size();
        const std::size_t textureCountBeforeOverride = overrideScene.textures.size();
        const std::size_t triangleCountBeforeOverride = overrideScene.triangleCount();
        const std::size_t meshletCountBeforeOverride = overrideScene.meshletCount();
        const std::uint32_t originalMaterialId = overrideScene.meshes.front().materialIndex;
        AssetLoader::setDefaultPlasticOverride(overrideScene, true);
        require(overrideScene.materialOverrideId != kInvalidMaterialId, "default plastic material override enabled");
        require(overrideScene.materialOverrideId < overrideScene.materials.size(), "material override ID in range");
        const Material& plastic = overrideScene.materials[overrideScene.materialOverrideId];
        require(plastic.name == "Default Plastic", "default plastic name");
        require(plastic.baseColor.x == 0.75f && plastic.baseColor.y == 0.75f && plastic.baseColor.z == 0.75f,
                "default plastic light-gray albedo");
        require(plastic.metallic == 0.0f, "default plastic metallic");
        require(plastic.roughness == 0.5f, "default plastic roughness");
        require(overrideScene.materials.size() == materialCountBeforeOverride,
                "default plastic already resident before override");
        require(overrideScene.meshes.front().materialIndex == originalMaterialId,
                "material override preserves original mesh material ID");
        require(overrideScene.textures.size() == textureCountBeforeOverride,
                "material override preserves imported texture references");
        require(overrideScene.triangleCount() == triangleCountBeforeOverride &&
                    overrideScene.meshletCount() == meshletCountBeforeOverride,
                "material override preserves processed geometry");
        AssetLoader::setDefaultPlasticOverride(overrideScene, false);
        require(overrideScene.materialOverrideId == kInvalidMaterialId, "material override disabled");
        require(overrideScene.meshes.front().materialIndex == originalMaterialId,
                "original material restored without reload");
    }

    AssetLoadResult importedWithGroundPlane = assetLoader.load(
        projectRoot / "assets" / "SampleTriangle.obj",
        {.enableOptionalMeshoptimizerPasses = true, .addGroundPlane = true});
    require(static_cast<bool>(importedWithGroundPlane), "Assimp import with ground plane");
    if (importedWithGroundPlane)
    {
        require(importedWithGroundPlane.scene.meshes.size() == 2, "ground plane mesh added");
        require(importedWithGroundPlane.scene.instances.size() == 2, "ground plane instance added");
        require(importedWithGroundPlane.scene.triangleCount() == 3, "ground plane triangle count");
        const Mesh& ground = importedWithGroundPlane.scene.meshes.back();
        require(ground.isGroundPlane, "ground plane marker");
        require(ground.lods.front().meshlets.size() == 1, "ground plane mandatory meshlet");
        const Material& groundMaterial = importedWithGroundPlane.scene.materials[ground.materialIndex];
        require(groundMaterial.name == "Default Plastic", "ground plane default plastic material");
        require(groundMaterial.baseColor.x == 0.75f && groundMaterial.metallic == 0.0f && groundMaterial.roughness == 0.5f,
                "ground plane PBR values");
    }

    const std::filesystem::path fbxPath = projectRoot / "assets" / "SampleObject.fbx";
    if (std::filesystem::exists(fbxPath))
    {
        AssetLoadResult importedFbx = assetLoader.load(fbxPath);
        require(static_cast<bool>(importedFbx), "Assimp FBX sample import");
        if (importedFbx)
        {
            require(!importedFbx.scene.meshes.empty(), "FBX mesh count");
            require(!importedFbx.scene.instances.empty(), "FBX instance count");
            require(importedFbx.scene.triangleCount() > 0, "FBX triangle count");
            require(importedFbx.scene.meshletCount() > 0, "FBX meshlet count");
            std::cout << "FBX diagnostic: " << importedFbx.scene.meshes.size() << " meshes, "
                      << importedFbx.scene.instances.size() << " instances, "
                      << importedFbx.scene.triangleCount() << " triangles, "
                      << importedFbx.scene.meshletCount() << " meshlets\n";
        }
    }

    const std::filesystem::path hdrPath = projectRoot / "assets" / "SampleHDR.hdr";
    if (std::filesystem::exists(hdrPath))
    {
        Environment environment;
        std::string error;
        require(assetLoader.loadHdrEnvironment(hdrPath, environment, error), "Radiance HDR sample import");
        require(environment.hdrWidth == 4096 && environment.hdrHeight == 2048, "HDR sample dimensions");
        require(environment.hdrMipCount == 13, "HDR mip count");
        require(environment.hasHdr(), "HDR mip pyramid validation");
        require(environment.hdrPixels.size() > static_cast<std::size_t>(environment.hdrWidth) *
                                                   environment.hdrHeight,
                "HDR mip pixels generated");
        require(environment.hdrConditionalCdf.size() ==
                    static_cast<std::size_t>(environment.hdrHeight) * (environment.hdrWidth + 1u),
                "HDR conditional importance CDF dimensions");
        require(environment.hdrMarginalCdf.size() == environment.hdrHeight + 1u &&
                    environment.hdrMarginalCdf.front() == 0.0f &&
                    std::abs(environment.hdrMarginalCdf.back() - 1.0f) < 1.0e-5f,
                "HDR marginal importance CDF normalization");
        require(environment.hdrImportanceTotal > 0.0f, "HDR importance distribution weight");
    }

    const Scene comparison = assetLoader.createPbrMaterialComparisonScene(
        {.enableOptionalMeshoptimizerPasses = true, .addGroundPlane = true});
    require(comparison.name == "PBR Material Comparison", "PBR comparison scene name");
    require(comparison.materials.size() >= 11 && comparison.instances.size() >= 11,
            "complete PBR comparison material grid");
    const auto hasMaterialFlag = [&](MaterialFlags flag) {
        return std::any_of(comparison.materials.begin(), comparison.materials.end(),
                           [&](const Material& material) { return hasFlag(material.flags(), flag); });
    };
    require(hasMaterialFlag(MaterialFlags::Clearcoat) && hasMaterialFlag(MaterialFlags::Transmission) &&
                hasMaterialFlag(MaterialFlags::Anisotropy) && hasMaterialFlag(MaterialFlags::Sheen) &&
                hasMaterialFlag(MaterialFlags::Emissive) && hasMaterialFlag(MaterialFlags::Subsurface) &&
                hasMaterialFlag(MaterialFlags::Volume),
            "PBR comparison scene covers every advanced material lobe");
    for (const Material& material : comparison.materials)
    {
        const GpuMaterial gpu = material.toGpu();
        const float* values = reinterpret_cast<const float*>(&gpu);
        bool finite = true;
        for (std::size_t index = 0; index < 40; ++index)
            finite = finite && std::isfinite(values[index]);
        require(finite, "comparison material GPU factors contain no NaN/Inf");
    }

    const auto fresnelDielectricReference = [](double cosine, double etaI, double etaT) {
        cosine = std::clamp(cosine, -1.0, 1.0);
        if (cosine < 0.0)
        {
            cosine = -cosine;
            std::swap(etaI, etaT);
        }
        const double sinTransmitted = etaI / etaT * std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
        if (sinTransmitted >= 1.0)
            return 1.0;
        const double cosTransmitted = std::sqrt(std::max(0.0, 1.0 - sinTransmitted * sinTransmitted));
        const double parallel = (etaT * cosine - etaI * cosTransmitted) /
                                (etaT * cosine + etaI * cosTransmitted);
        const double perpendicular = (etaI * cosine - etaT * cosTransmitted) /
                                     (etaI * cosine + etaT * cosTransmitted);
        return 0.5 * (parallel * parallel + perpendicular * perpendicular);
    };
    require(std::abs(fresnelDielectricReference(1.0, 1.0, 1.5) - 0.04) < 1.0e-8,
            "dielectric Fresnel normal incidence");
    require(fresnelDielectricReference(1.0e-6, 1.0, 1.5) > 0.999,
            "dielectric Fresnel grazing incidence");
    require(fresnelDielectricReference(0.5, 1.5, 1.0) == 1.0,
            "dielectric total internal reflection");

    const Vec3 absorptionColor{0.25f, 0.5f, 0.8f};
    const float absorptionDistance = 2.0f;
    const Vec3 sigmaA{-std::log(absorptionColor.x) / absorptionDistance,
                      -std::log(absorptionColor.y) / absorptionDistance,
                      -std::log(absorptionColor.z) / absorptionDistance};
    const Vec3 recoveredColor{std::exp(-sigmaA.x * absorptionDistance),
                              std::exp(-sigmaA.y * absorptionDistance),
                              std::exp(-sigmaA.z * absorptionDistance)};
    require(std::abs(recoveredColor.x - absorptionColor.x) < 1.0e-6f &&
                std::abs(recoveredColor.y - absorptionColor.y) < 1.0e-6f &&
                std::abs(recoveredColor.z - absorptionColor.z) < 1.0e-6f,
            "Beer-Lambert absorption round trip");

    const auto henyeyGreenstein = [](double cosine, double g) {
        const double denominator = std::max(1.0 + g * g - 2.0 * g * cosine, 1.0e-12);
        return (1.0 - g * g) / (4.0 * 3.14159265358979323846 * denominator * std::sqrt(denominator));
    };
    for (const double anisotropy : {-0.8, 0.0, 0.8})
    {
        constexpr int thetaSteps = 1024;
        double integral = 0.0;
        for (int thetaIndex = 0; thetaIndex < thetaSteps; ++thetaIndex)
        {
            const double theta = (thetaIndex + 0.5) * 3.14159265358979323846 / thetaSteps;
            integral += henyeyGreenstein(std::cos(theta), anisotropy) * std::sin(theta) *
                        2.0 * 3.14159265358979323846 * 3.14159265358979323846 / thetaSteps;
        }
        require(std::abs(integral - 1.0) < 2.0e-4, "Henyey-Greenstein phase normalization");
    }

    const auto furnaceReflectance = [](double roughness, double metallic, double baseColor,
                                       double viewCosine) {
        constexpr int thetaSteps = 128;
        constexpr int phiSteps = 256;
        const double alpha = std::max(roughness * roughness, 0.0004);
        const double sinView = std::sqrt(std::max(0.0, 1.0 - viewCosine * viewCosine));
        const std::array<double, 3> view{sinView, 0.0, viewCosine};
        double integral = 0.0;
        for (int thetaIndex = 0; thetaIndex < thetaSteps; ++thetaIndex)
        {
            const double theta = (thetaIndex + 0.5) * 0.5 * 3.14159265358979323846 / thetaSteps;
            const double nDotL = std::cos(theta);
            const double sinTheta = std::sin(theta);
            for (int phiIndex = 0; phiIndex < phiSteps; ++phiIndex)
            {
                const double phi = (phiIndex + 0.5) * 2.0 * 3.14159265358979323846 / phiSteps;
                const std::array<double, 3> light{sinTheta * std::cos(phi), sinTheta * std::sin(phi), nDotL};
                std::array<double, 3> halfway{view[0] + light[0], view[1] + light[1], view[2] + light[2]};
                const double halfwayLength = std::sqrt(halfway[0] * halfway[0] + halfway[1] * halfway[1] +
                                                       halfway[2] * halfway[2]);
                for (double& component : halfway)
                    component /= halfwayLength;
                const double nDotH = std::max(halfway[2], 0.0);
                const double vDotH = std::clamp(view[0] * halfway[0] + view[1] * halfway[1] +
                                                view[2] * halfway[2], 0.0, 1.0);
                const double denominator = nDotH * nDotH * (alpha * alpha - 1.0) + 1.0;
                const double distribution = alpha * alpha /
                    (3.14159265358979323846 * denominator * denominator);
                const auto smithG1 = [&](double cosine) {
                    const double cosine2 = cosine * cosine;
                    return 2.0 * cosine /
                           std::max(cosine + std::sqrt(alpha * alpha + (1.0 - alpha * alpha) * cosine2), 1.0e-9);
                };
                const double geometry = smithG1(viewCosine) * smithG1(nDotL);
                const double f0 = 0.04 * (1.0 - metallic) + baseColor * metallic;
                const double fresnel = f0 + (1.0 - f0) * std::pow(1.0 - vDotH, 5.0);
                const double specular = distribution * geometry * fresnel /
                                        std::max(4.0 * viewCosine * nDotL, 1.0e-9);
                const double diffuse = (1.0 - fresnel) * (1.0 - metallic) * baseColor /
                                       3.14159265358979323846;
                const double solidAngle = sinTheta * (0.5 * 3.14159265358979323846 / thetaSteps) *
                                          (2.0 * 3.14159265358979323846 / phiSteps);
                integral += (diffuse + specular) * nDotL * solidAngle;
            }
        }
        return integral;
    };
    for (const double roughness : {0.02, 0.1, 0.35, 0.7, 1.0})
    {
        for (const double metallic : {0.0, 0.5, 1.0})
        {
            for (const double viewCosine : {0.1, 0.35, 0.7, 1.0})
            {
                const double energy = furnaceReflectance(roughness, metallic, 0.8, viewCosine);
                require(std::isfinite(energy) && energy >= 0.0 && energy <= 1.05,
                        "GGX white-furnace energy and NaN/Inf bound");
            }
        }
    }

    if (failures == 0)
        std::cout << "VORaytracer CPU and asset tests passed\n";
    return failures == 0 ? 0 : 1;
}
