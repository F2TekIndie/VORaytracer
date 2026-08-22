#pragma once

#include "Core/Material.h"
#include "Core/Math.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace vor
{
inline constexpr std::uint32_t kInvalidMaterialId = std::numeric_limits<std::uint32_t>::max();

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
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mipCount{};
    std::vector<std::uint32_t> mipOffsets;
    std::vector<std::uint8_t> rgba8Pixels;

    [[nodiscard]] bool valid() const
    {
        return width > 0 && height > 0 && mipCount > 0 && mipOffsets.size() == mipCount &&
               !rgba8Pixels.empty();
    }
};

struct Material
{
    std::string name;
    Vec4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    Vec3 emissive{};
    float metallic{};
    float roughness{1.0f};
    float normalScale{1.0f};
    float bumpScale{1.0f};
    float occlusionStrength{1.0f};
    float alphaCutoff{0.5f};
    float transmission{};
    float indexOfRefraction{1.5f};
    float clearcoat{};
    float clearcoatRoughness{};
    float anisotropy{};
    float anisotropyRotation{};
    Vec3 sheenColor{};
    float sheenRoughness{};
    Vec3 absorptionColor{1.0f, 1.0f, 1.0f};
    float absorptionDistance{1.0e30f};
    float subsurface{};
    Vec3 subsurfaceColor{1.0f, 1.0f, 1.0f};
    float subsurfaceRadius{1.0f};
    Vec3 volumeAbsorption{};
    float volumeDensity{};
    Vec3 volumeScattering{};
    float volumeAnisotropy{};
    AlphaMode alphaMode{AlphaMode::Opaque};
    bool doubleSided{};
    std::int32_t baseColorTexture{-1};
    std::int32_t opacityTexture{-1};
    std::int32_t normalTexture{-1};
    std::int32_t heightTexture{-1};
    std::uint32_t heightTextureWidth{};
    std::uint32_t heightTextureHeight{};
    std::int32_t metallicRoughnessTexture{-1};
    std::int32_t occlusionTexture{-1};
    std::int32_t emissiveTexture{-1};
    std::int32_t transmissionTexture{-1};
    std::int32_t clearcoatTexture{-1};
    std::int32_t clearcoatRoughnessTexture{-1};
    std::int32_t clearcoatNormalTexture{-1};
    std::int32_t sheenColorTexture{-1};
    std::int32_t sheenRoughnessTexture{-1};
    std::int32_t anisotropyTexture{-1};

    [[nodiscard]] MaterialFlags flags() const;
    [[nodiscard]] GpuMaterial toGpu() const;
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
    std::vector<float> hdrConditionalCdf;
    std::vector<float> hdrMarginalCdf;
    float hdrImportanceTotal{};
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
    std::uint32_t materialOverrideId{kInvalidMaterialId};

    [[nodiscard]] std::size_t triangleCount() const;
    [[nodiscard]] std::size_t meshletCount() const;
    [[nodiscard]] bool empty() const { return meshes.empty(); }
};

Scene makeDefaultScene();
} // namespace vor
