#include "Core/Scene.h"

namespace vor
{
namespace
{
std::uint32_t textureId(std::int32_t value)
{
    return value >= 0 ? static_cast<std::uint32_t>(value) : kInvalidTextureId;
}
} // namespace

MaterialFlags Material::flags() const
{
    MaterialFlags result = MaterialFlags::None;
    if (doubleSided)
        result |= MaterialFlags::DoubleSided;
    if (alphaMode == AlphaMode::Mask)
        result |= MaterialFlags::AlphaMask;
    if (transmission > 0.0f)
        result |= MaterialFlags::Transmission;
    if (clearcoat > 0.0f)
        result |= MaterialFlags::Clearcoat;
    if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f)
        result |= MaterialFlags::Sheen;
    if (std::abs(anisotropy) > 0.0f)
        result |= MaterialFlags::Anisotropy;
    if (emissive.x > 0.0f || emissive.y > 0.0f || emissive.z > 0.0f)
        result |= MaterialFlags::Emissive;
    if (subsurface > 0.0f)
        result |= MaterialFlags::Subsurface;
    if (volumeDensity > 0.0f || volumeAbsorption.x > 0.0f || volumeAbsorption.y > 0.0f ||
        volumeAbsorption.z > 0.0f || volumeScattering.x > 0.0f || volumeScattering.y > 0.0f ||
        volumeScattering.z > 0.0f)
        result |= MaterialFlags::Volume;
    return result;
}

GpuMaterial Material::toGpu() const
{
    GpuMaterial result{};
    result.baseColorFactor = baseColor;
    result.emissiveAndMetallic = {emissive.x, emissive.y, emissive.z, metallic};
    result.surfaceParameters = {roughness, normalScale, occlusionStrength, alphaCutoff};
    result.transmissionClearcoat = {transmission, indexOfRefraction, clearcoat, clearcoatRoughness};
    result.anisotropySheen = {anisotropy, anisotropyRotation, sheenRoughness, 0.0f};
    result.sheenColorAbsorptionDistance = {sheenColor.x, sheenColor.y, sheenColor.z, absorptionDistance};
    result.absorptionColorSubsurface = {absorptionColor.x, absorptionColor.y, absorptionColor.z, subsurface};
    result.subsurfaceColorRadius = {subsurfaceColor.x, subsurfaceColor.y, subsurfaceColor.z, subsurfaceRadius};
    result.volumeAbsorptionDensity = {volumeAbsorption.x, volumeAbsorption.y, volumeAbsorption.z, volumeDensity};
    result.volumeScatteringAnisotropy = {volumeScattering.x, volumeScattering.y, volumeScattering.z, volumeAnisotropy};
    result.materialFlags = {static_cast<std::uint32_t>(flags()), static_cast<std::uint32_t>(alphaMode), 0u, 0u};
    result.textureIndices0 = {textureId(baseColorTexture), textureId(metallicRoughnessTexture),
                              textureId(normalTexture), textureId(emissiveTexture)};
    result.textureIndices1 = {textureId(occlusionTexture), textureId(transmissionTexture),
                              textureId(clearcoatTexture), textureId(clearcoatRoughnessTexture)};
    result.textureIndices2 = {textureId(clearcoatNormalTexture), textureId(sheenColorTexture),
                              textureId(sheenRoughnessTexture), textureId(anisotropyTexture)};
    return result;
}

std::size_t Scene::triangleCount() const
{
    std::size_t count = 0;
    for (const Mesh& mesh : meshes)
    {
        if (!mesh.lods.empty())
            count += mesh.lods.front().indices.size() / 3;
    }
    return count;
}

std::size_t Scene::meshletCount() const
{
    std::size_t count = 0;
    for (const Mesh& mesh : meshes)
    {
        if (!mesh.lods.empty())
            count += mesh.lods.front().meshlets.size();
    }
    return count;
}

Scene makeDefaultScene()
{
    Scene scene{};
    scene.name = "Procedural Cube";
    scene.materials.push_back(Material{.name = "Default", .baseColor = {0.72f, 0.34f, 0.12f, 1.0f}, .metallic = 0.1f, .roughness = 0.35f});
    scene.lights.push_back(Light{.name = "Sun", .type = LightType::Directional});
    return scene;
}
} // namespace vor
