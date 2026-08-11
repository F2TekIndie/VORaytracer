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

    if (failures == 0)
        std::cout << "VORaytracer CPU and asset tests passed\n";
    return failures == 0 ? 0 : 1;
}
