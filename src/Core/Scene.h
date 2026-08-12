#pragma once

#include "Core/Math.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vor
{
struct Vertex
{
    Vec3 position{};
    Vec3 normal{0.0f, 1.0f, 0.0f};
    Vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    Vec2 uv{};
};

struct Meshlet
{
    std::uint32_t vertexOffset{};
    std::uint32_t triangleOffset{};
    std::uint32_t vertexCount{};
    std::uint32_t triangleCount{};
    Vec4 boundingSphere{};
    Vec4 normalCone{};
};

struct MeshLod
{
    std::vector<std::uint32_t> indices;
    std::vector<Meshlet> meshlets;
    std::vector<std::uint32_t> meshletVertices;
    std::vector<std::uint8_t> meshletTriangles;
    float simplificationError{};
};

struct Mesh
{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<MeshLod> lods;
    std::uint32_t materialIndex{};
    bool isGroundPlane{};
};

enum class AlphaMode : std::uint32_t
{
    Opaque,
    Mask,
    Blend,
};

struct TextureReference
{
    std::filesystem::path path;
    bool srgb{};
};

struct Material
{
    std::string name;
    Vec4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    Vec3 emissive{};
    float metallic{};
    float roughness{1.0f};
    float normalScale{1.0f};
    float occlusionStrength{1.0f};
    float alphaCutoff{0.5f};
    AlphaMode alphaMode{AlphaMode::Opaque};
    bool doubleSided{};
    std::int32_t baseColorTexture{-1};
    std::int32_t normalTexture{-1};
    std::int32_t metallicRoughnessTexture{-1};
    std::int32_t occlusionTexture{-1};
    std::int32_t emissiveTexture{-1};
};

struct Instance
{
    std::string name;
    std::uint32_t meshIndex{};
    Mat4 transform{Mat4::identity()};
    Mat4 previousTransform{Mat4::identity()};
};

enum class LightType : std::uint32_t
{
    Directional,
    Point,
    Spot,
    Area,
};

struct Light
{
    std::string name;
    LightType type{LightType::Point};
    Vec3 color{1.0f, 0.96f, 0.88f};
    float intensity{5.0f};
    Vec3 position{};
    float range{100.0f};
    Vec3 direction{0.45f, -0.85f, -0.3f};
    float innerCone{0.5f};
    float outerCone{0.7f};
};

enum class GlobalLightMode : std::uint32_t
{
    Directional,
    HdrEnvironment,
    ProceduralSky,
};

struct Environment
{
    GlobalLightMode mode{GlobalLightMode::Directional};
    std::filesystem::path hdrPath;
    std::vector<Vec4> hdrPixels;
    std::uint32_t hdrWidth{};
    std::uint32_t hdrHeight{};
    std::uint32_t hdrMipCount{};
    float intensity{1.0f};
    float rotationRadians{};
    Vec3 zenithColor{0.12f, 0.32f, 0.75f};
    Vec3 horizonColor{0.65f, 0.75f, 0.90f};
    Vec3 groundColor{0.035f, 0.035f, 0.04f};
    bool visibleBackground{true};

    [[nodiscard]] bool hasHdr() const
    {
        if (hdrWidth == 0 || hdrHeight == 0 || hdrMipCount == 0)
            return false;
        std::size_t expectedPixels = 0;
        std::uint32_t width = hdrWidth;
        std::uint32_t height = hdrHeight;
        for (std::uint32_t mip = 0; mip < hdrMipCount; ++mip)
        {
            expectedPixels += static_cast<std::size_t>(width) * height;
            width = std::max(width / 2, 1u);
            height = std::max(height / 2, 1u);
        }
        return hdrPixels.size() == expectedPixels;
    }
};

struct Camera
{
    Vec3 position{0.0f, 1.5f, 4.0f};
    Vec3 target{0.0f, 0.5f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float verticalFovDegrees{60.0f};
    float nearPlane{0.05f};
    float farPlane{1000.0f};
};

struct Scene
{
    std::string name{"Untitled"};
    std::filesystem::path sourcePath;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<TextureReference> textures;
    std::vector<Instance> instances;
    std::vector<Light> lights;
    Environment environment{};
    Camera camera{};

    [[nodiscard]] std::size_t triangleCount() const;
    [[nodiscard]] std::size_t meshletCount() const;
    [[nodiscard]] bool empty() const { return meshes.empty(); }
};

Scene makeDefaultScene();
} // namespace vor
