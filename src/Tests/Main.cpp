#include "Core/Math.h"
#include "Core/Scene.h"
#include "Assets/AssetLoader.h"

#include <meshoptimizer.h>

#include <array>
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
    }

    if (failures == 0)
        std::cout << "VORaytracer CPU and asset tests passed\n";
    return failures == 0 ? 0 : 1;
}
