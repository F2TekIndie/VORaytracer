#include "Assets/AssetLoader.h"

#include "Core/Log.h"
#include "Core/TextureCompression.h"

#include <assimp/Importer.hpp>
#include <assimp/GltfMaterial.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <meshoptimizer.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <span>
#include <unordered_map>

namespace vor
{
namespace
{
Mat4 toMat4(const aiMatrix4x4& source)
{
    Mat4 result{};
    result.m = {source.a1, source.b1, source.c1, source.d1,
                source.a2, source.b2, source.c2, source.d2,
                source.a3, source.b3, source.c3, source.d3,
                source.a4, source.b4, source.c4, source.d4};
    return result;
}

Vec3 transformPoint(const Mat4& transform, Vec3 point)
{
    return {transform.m[0] * point.x + transform.m[4] * point.y + transform.m[8] * point.z + transform.m[12],
            transform.m[1] * point.x + transform.m[5] * point.y + transform.m[9] * point.z + transform.m[13],
            transform.m[2] * point.x + transform.m[6] * point.y + transform.m[10] * point.z + transform.m[14]};
}

Vec3 transformVector(const Mat4& transform, Vec3 vector)
{
    return normalize({transform.m[0] * vector.x + transform.m[4] * vector.y + transform.m[8] * vector.z,
                      transform.m[1] * vector.x + transform.m[5] * vector.y + transform.m[9] * vector.z,
                      transform.m[2] * vector.x + transform.m[6] * vector.y + transform.m[10] * vector.z});
}

Vec4 toColor(const aiColor4D& color)
{
    return {color.r, color.g, color.b, color.a};
}

Material makeDefaultPlasticMaterial()
{
    Material material{};
    material.name = "Default Plastic";
    material.baseColor = {0.75f, 0.75f, 0.75f, 1.0f};
    material.metallic = 0.0f;
    material.roughness = 0.5f;
    return material;
}

void normalizeMaterial(Material& material)
{
    const auto finiteOr = [](float value, float fallback) { return std::isfinite(value) ? value : fallback; };
    material.baseColor.x = std::max(finiteOr(material.baseColor.x, 1.0f), 0.0f);
    material.baseColor.y = std::max(finiteOr(material.baseColor.y, 1.0f), 0.0f);
    material.baseColor.z = std::max(finiteOr(material.baseColor.z, 1.0f), 0.0f);
    material.baseColor.w = std::clamp(finiteOr(material.baseColor.w, 1.0f), 0.0f, 1.0f);
    material.metallic = std::clamp(finiteOr(material.metallic, 0.0f), 0.0f, 1.0f);
    material.roughness = std::clamp(finiteOr(material.roughness, 1.0f), 0.02f, 1.0f);
    material.normalScale = std::max(finiteOr(material.normalScale, 1.0f), 0.0f);
    material.bumpScale = std::max(finiteOr(material.bumpScale, 1.0f), 0.0f);
    material.occlusionStrength = std::clamp(finiteOr(material.occlusionStrength, 1.0f), 0.0f, 1.0f);
    material.alphaCutoff = std::clamp(finiteOr(material.alphaCutoff, 0.5f), 0.0f, 1.0f);
    material.transmission = std::clamp(finiteOr(material.transmission, 0.0f), 0.0f, 1.0f);
    material.indexOfRefraction = std::clamp(finiteOr(material.indexOfRefraction, 1.5f), 1.0001f, 3.0f);
    material.clearcoat = std::clamp(finiteOr(material.clearcoat, 0.0f), 0.0f, 1.0f);
    material.clearcoatRoughness = std::clamp(finiteOr(material.clearcoatRoughness, 0.0f), 0.02f, 1.0f);
    material.anisotropy = std::clamp(finiteOr(material.anisotropy, 0.0f), -0.99f, 0.99f);
    material.anisotropyRotation = finiteOr(material.anisotropyRotation, 0.0f);
    material.sheenRoughness = std::clamp(finiteOr(material.sheenRoughness, 0.0f), 0.0f, 1.0f);
    material.absorptionDistance = std::clamp(finiteOr(material.absorptionDistance, 1.0e30f), 1.0e-4f, 1.0e30f);
    material.subsurface = std::clamp(finiteOr(material.subsurface, 0.0f), 0.0f, 1.0f);
    material.subsurfaceRadius = std::max(finiteOr(material.subsurfaceRadius, 1.0f), 1.0e-4f);
    material.volumeDensity = std::max(finiteOr(material.volumeDensity, 0.0f), 0.0f);
    material.volumeAnisotropy = std::clamp(finiteOr(material.volumeAnisotropy, 0.0f), -0.99f, 0.99f);
}

float srgbToLinear(float value)
{
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb(float value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

std::uint8_t toByte(float value)
{
    return static_cast<std::uint8_t>(std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
}

float radicalInverse(std::uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

Vec2 hammersley(std::uint32_t index, std::uint32_t count)
{
    return {static_cast<float>(index) / static_cast<float>(count), radicalInverse(index)};
}

Vec3 equirectDirection(float u, float v)
{
    const float phi = (u - 0.5f) * 2.0f * kPi;
    const float theta = v * kPi;
    const float sineTheta = std::sin(theta);
    return {std::cos(phi) * sineTheta, std::cos(theta), std::sin(phi) * sineTheta};
}

Vec3 sampleHdrBilinear(std::span<const Vec4> pixels, std::uint32_t width, std::uint32_t height, Vec3 direction)
{
    direction = normalize(direction);
    float u = std::atan2(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
    u -= std::floor(u);
    const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / kPi;
    const float x = u * static_cast<float>(width) - 0.5f;
    const float y = v * static_cast<float>(height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, static_cast<int>(height) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(height) - 1);
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto at = [&](int sourceX, int sourceY) {
        sourceX %= static_cast<int>(width);
        if (sourceX < 0)
            sourceX += static_cast<int>(width);
        const Vec4& value = pixels[static_cast<std::size_t>(sourceY) * width + sourceX];
        return Vec3{value.x, value.y, value.z};
    };
    const Vec3 top = at(x0, y0) * (1.0f - tx) + at(x0 + 1, y0) * tx;
    const Vec3 bottom = at(x0, y1) * (1.0f - tx) + at(x0 + 1, y1) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

Vec3 toWorld(Vec3 local, Vec3 normal)
{
    const Vec3 helper = std::abs(normal.y) < 0.999f ? Vec3{0.0f, 1.0f, 0.0f}
                                                     : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(helper, normal));
    const Vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + normal * local.y + bitangent * local.z);
}

Vec3 importanceSampleGgx(Vec2 sample, float roughness, Vec3 normal)
{
    const float alpha = std::max(roughness * roughness, 1.0e-4f);
    const float alphaSquared = alpha * alpha;
    const float phi = 2.0f * kPi * sample.x;
    const float cosineTheta = std::sqrt((1.0f - sample.y) /
                                        (1.0f + (alphaSquared - 1.0f) * sample.y));
    const float sineTheta = std::sqrt(std::max(1.0f - cosineTheta * cosineTheta, 0.0f));
    return toWorld({std::cos(phi) * sineTheta, cosineTheta, std::sin(phi) * sineTheta}, normal);
}

float geometrySchlickGgxIbl(float nDotDirection, float roughness)
{
    const float k = roughness * roughness * 0.5f;
    return nDotDirection / std::max(nDotDirection * (1.0f - k) + k, 1.0e-6f);
}

void generateIblAtlas(Environment& environment)
{
    constexpr std::uint32_t atlasWidth = Environment::kIblAtlasWidth;
    constexpr std::uint32_t atlasHeight = Environment::kIblAtlasHeight;
    constexpr std::uint32_t specularWidth = 256;
    constexpr std::uint32_t specularHeight = 128;
    constexpr std::uint32_t specularMipCount = 9;
    constexpr std::uint32_t irradianceWidth = 64;
    constexpr std::uint32_t irradianceHeight = 32;
    constexpr std::uint32_t brdfSize = 128;
    constexpr std::uint32_t diffuseSamples = 128;
    constexpr std::uint32_t specularSamples = 96;
    constexpr std::uint32_t brdfSamples = 128;

    const std::span<const Vec4> source(environment.hdrPixels.data(),
                                       static_cast<std::size_t>(environment.hdrWidth) * environment.hdrHeight);
    environment.iblPixels.assign(static_cast<std::size_t>(atlasWidth) * atlasHeight, Vec4{});
    const auto store = [&](std::uint32_t x, std::uint32_t y, Vec3 value) {
        environment.iblPixels[static_cast<std::size_t>(y) * atlasWidth + x] =
            {std::max(value.x, 0.0f), std::max(value.y, 0.0f), std::max(value.z, 0.0f), 1.0f};
    };

    // Cosine-weighted samples directly estimate irradiance divided by PI, matching Lambert albedo below.
    for (std::uint32_t y = 0; y < irradianceHeight; ++y)
    {
        for (std::uint32_t x = 0; x < irradianceWidth; ++x)
        {
            const Vec3 normal = equirectDirection((static_cast<float>(x) + 0.5f) / irradianceWidth,
                                                   (static_cast<float>(y) + 0.5f) / irradianceHeight);
            Vec3 irradiance{};
            for (std::uint32_t sampleIndex = 0; sampleIndex < diffuseSamples; ++sampleIndex)
            {
                const Vec2 sample = hammersley(sampleIndex, diffuseSamples);
                const float radius = std::sqrt(sample.y);
                const float phi = 2.0f * kPi * sample.x;
                const Vec3 direction = toWorld({radius * std::cos(phi), std::sqrt(1.0f - sample.y),
                                                radius * std::sin(phi)}, normal);
                irradiance = irradiance + sampleHdrBilinear(source, environment.hdrWidth,
                                                            environment.hdrHeight, direction);
            }
            store(256 + x, 64 + y, irradiance / static_cast<float>(diffuseSamples));
        }
    }

    // GGX-prefiltered specular levels are packed left-to-right across the atlas top.
    for (std::uint32_t mip = 0; mip < specularMipCount; ++mip)
    {
        const std::uint32_t width = std::max(specularWidth >> mip, 1u);
        const std::uint32_t height = std::max(specularHeight >> mip, 1u);
        const std::uint32_t offsetX = mip == 0 ? 0u : atlasWidth - (atlasWidth >> mip);
        const float roughness = static_cast<float>(mip) / static_cast<float>(specularMipCount - 1);
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                const Vec3 reflection = equirectDirection((static_cast<float>(x) + 0.5f) / width,
                                                           (static_cast<float>(y) + 0.5f) / height);
                if (mip == 0)
                {
                    store(offsetX + x, y, sampleHdrBilinear(source, environment.hdrWidth,
                                                            environment.hdrHeight, reflection));
                    continue;
                }
                Vec3 radiance{};
                float totalWeight = 0.0f;
                for (std::uint32_t sampleIndex = 0; sampleIndex < specularSamples; ++sampleIndex)
                {
                    const Vec3 halfVector = importanceSampleGgx(hammersley(sampleIndex, specularSamples),
                                                               roughness, reflection);
                    const Vec3 light = normalize(halfVector * (2.0f * dot(reflection, halfVector)) - reflection);
                    const float nDotLight = std::max(dot(reflection, light), 0.0f);
                    if (nDotLight > 0.0f)
                    {
                        radiance = radiance + sampleHdrBilinear(source, environment.hdrWidth,
                                                                environment.hdrHeight, light) * nDotLight;
                        totalWeight += nDotLight;
                    }
                }
                store(offsetX + x, y, radiance / std::max(totalWeight, 1.0e-6f));
            }
        }
    }

    // Split-sum environment BRDF integration (R=scale, G=bias).
    for (std::uint32_t y = 0; y < brdfSize; ++y)
    {
        const float roughness = (static_cast<float>(y) + 0.5f) / brdfSize;
        for (std::uint32_t x = 0; x < brdfSize; ++x)
        {
            const float nDotView = (static_cast<float>(x) + 0.5f) / brdfSize;
            const Vec3 view{std::sqrt(std::max(1.0f - nDotView * nDotView, 0.0f)), nDotView, 0.0f};
            float scale = 0.0f;
            float bias = 0.0f;
            for (std::uint32_t sampleIndex = 0; sampleIndex < brdfSamples; ++sampleIndex)
            {
                const Vec3 halfVector = importanceSampleGgx(hammersley(sampleIndex, brdfSamples), roughness,
                                                            {0.0f, 1.0f, 0.0f});
                const Vec3 light = normalize(halfVector * (2.0f * dot(view, halfVector)) - view);
                const float nDotLight = std::max(light.y, 0.0f);
                const float nDotHalf = std::max(halfVector.y, 0.0f);
                const float viewDotHalf = std::max(dot(view, halfVector), 0.0f);
                if (nDotLight <= 0.0f)
                    continue;
                const float geometry = geometrySchlickGgxIbl(nDotView, roughness) *
                                       geometrySchlickGgxIbl(nDotLight, roughness);
                const float visibility = geometry * viewDotHalf /
                                         std::max(nDotHalf * nDotView, 1.0e-6f);
                const float fresnel = std::pow(1.0f - viewDotHalf, 5.0f);
                scale += (1.0f - fresnel) * visibility;
                bias += fresnel * visibility;
            }
            environment.iblPixels[static_cast<std::size_t>(128 + y) * atlasWidth + 256 + x] =
                {scale / brdfSamples, bias / brdfSamples, 0.0f, 1.0f};
        }
    }
}

void generateTextureMips(TextureReference& texture, std::vector<std::uint8_t> basePixels)
{
    texture.rgba8Pixels.clear();
    texture.mipOffsets.clear();
    std::uint32_t width = texture.width;
    std::uint32_t height = texture.height;
    std::vector<std::uint8_t> level = std::move(basePixels);
    for (;;)
    {
        texture.mipOffsets.push_back(static_cast<std::uint32_t>(texture.rgba8Pixels.size()));
        texture.rgba8Pixels.insert(texture.rgba8Pixels.end(), level.begin(), level.end());
        if (width == 1 && height == 1)
            break;

        const std::uint32_t nextWidth = std::max(width / 2, 1u);
        const std::uint32_t nextHeight = std::max(height / 2, 1u);
        std::vector<std::uint8_t> next(static_cast<std::size_t>(nextWidth) * nextHeight * 4);
        for (std::uint32_t y = 0; y < nextHeight; ++y)
        {
            for (std::uint32_t x = 0; x < nextWidth; ++x)
            {
                float sum[4]{};
                std::uint32_t sampleCount = 0;
                for (std::uint32_t dy = 0; dy < 2; ++dy)
                {
                    for (std::uint32_t dx = 0; dx < 2; ++dx)
                    {
                        const std::uint32_t sourceX = std::min(x * 2 + dx, width - 1);
                        const std::uint32_t sourceY = std::min(y * 2 + dy, height - 1);
                        const std::size_t sourceOffset = (static_cast<std::size_t>(sourceY) * width + sourceX) * 4;
                        for (std::uint32_t channel = 0; channel < 4; ++channel)
                        {
                            float value = static_cast<float>(level[sourceOffset + channel]) / 255.0f;
                            if (texture.srgb && channel < 3)
                                value = srgbToLinear(value);
                            sum[channel] += value;
                        }
                        ++sampleCount;
                    }
                }
                const std::size_t destinationOffset = (static_cast<std::size_t>(y) * nextWidth + x) * 4;
                for (std::uint32_t channel = 0; channel < 4; ++channel)
                {
                    float value = sum[channel] / static_cast<float>(sampleCount);
                    if (texture.srgb && channel < 3)
                        value = linearToSrgb(value);
                    next[destinationOffset + channel] = toByte(value);
                }
            }
        }
        level = std::move(next);
        width = nextWidth;
        height = nextHeight;
    }
    texture.mipCount = static_cast<std::uint32_t>(texture.mipOffsets.size());
    compressTextureBc3(texture);
}

bool decodeTexture(const aiScene* imported, const std::filesystem::path& resolved,
                   std::string_view assimpPath, TextureReference& texture)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded = nullptr;
    const aiTexture* embedded = imported ? imported->GetEmbeddedTexture(std::string(assimpPath).c_str()) : nullptr;
    std::vector<std::uint8_t> uncompressed;
    if (embedded && embedded->mHeight > 0)
    {
        width = static_cast<int>(embedded->mWidth);
        height = static_cast<int>(embedded->mHeight);
        uncompressed.resize(static_cast<std::size_t>(width) * height * 4);
        for (std::size_t index = 0; index < static_cast<std::size_t>(width) * height; ++index)
        {
            const aiTexel& texel = embedded->pcData[index];
            uncompressed[index * 4 + 0] = texel.r;
            uncompressed[index * 4 + 1] = texel.g;
            uncompressed[index * 4 + 2] = texel.b;
            uncompressed[index * 4 + 3] = texel.a;
        }
    }
    else
    {
        std::vector<std::uint8_t> encoded;
        if (embedded)
        {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(embedded->pcData);
            encoded.assign(bytes, bytes + embedded->mWidth);
        }
        else
        {
            std::ifstream stream(resolved, std::ios::binary | std::ios::ate);
            if (!stream)
                return false;
            const std::streamsize size = stream.tellg();
            if (size <= 0)
                return false;
            encoded.resize(static_cast<std::size_t>(size));
            stream.seekg(0);
            if (!stream.read(reinterpret_cast<char*>(encoded.data()), size))
                return false;
        }
        decoded = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels, 4);
        if (!decoded)
            return false;
        uncompressed.assign(decoded, decoded + static_cast<std::size_t>(width) * height * 4);
        stbi_image_free(decoded);
    }

    if (width <= 0 || height <= 0)
        return false;
    texture.width = static_cast<std::uint32_t>(width);
    texture.height = static_cast<std::uint32_t>(height);
    generateTextureMips(texture, std::move(uncompressed));
    return texture.valid();
}

std::int32_t addTexture(Scene& scene, const aiScene* imported, const std::filesystem::path& basePath,
                        aiMaterial* material, aiTextureType type, bool srgb, unsigned textureIndex = 0)
{
    aiString texturePath;
    aiTextureMapping mapping = aiTextureMapping_UV;
    unsigned uvIndex = 0;
    aiTextureMapMode mapModes[3]{aiTextureMapMode_Wrap, aiTextureMapMode_Wrap, aiTextureMapMode_Wrap};
    if (material->GetTextureCount(type) <= textureIndex ||
        material->GetTexture(type, textureIndex, &texturePath, &mapping, &uvIndex, nullptr, nullptr, mapModes) !=
            AI_SUCCESS)
        return -1;
    if (mapping != aiTextureMapping_UV)
        uvIndex = 0;
    aiUVTransform uvTransform{};
    material->Get(AI_MATKEY_UVTRANSFORM(type, textureIndex), uvTransform);
    const auto addressMode = [](aiTextureMapMode mode) {
        return mode == aiTextureMapMode_Clamp || mode == aiTextureMapMode_Decal
                   ? TextureAddressMode::Clamp
               : mode == aiTextureMapMode_Mirror ? TextureAddressMode::Mirror
                                                  : TextureAddressMode::Repeat;
    };

    const std::string assimpPath = texturePath.C_Str();
    std::filesystem::path resolved = std::filesystem::path(assimpPath);
    if (!resolved.is_absolute() && !resolved.native().starts_with(L"*"))
        resolved = basePath / resolved;
    resolved = resolved.lexically_normal();

    const auto bindingMatches = [&](const TextureReference& value) {
        return value.srgb == srgb && value.uvSet == std::min(uvIndex, 1u) &&
               value.uvScale.x == uvTransform.mScaling.x && value.uvScale.y == uvTransform.mScaling.y &&
               value.uvOffset.x == uvTransform.mTranslation.x && value.uvOffset.y == uvTransform.mTranslation.y &&
               value.uvRotation == uvTransform.mRotation && value.addressU == addressMode(mapModes[0]) &&
               value.addressV == addressMode(mapModes[1]);
    };
    const auto found = std::find_if(scene.textures.begin(), scene.textures.end(), [&](const TextureReference& value) {
        return value.path == resolved && bindingMatches(value);
    });
    if (found != scene.textures.end())
        return static_cast<std::int32_t>(std::distance(scene.textures.begin(), found));

    TextureReference texture{};
    texture.path = resolved;
    texture.srgb = srgb;
    texture.uvSet = std::min(uvIndex, 1u);
    texture.uvScale = {uvTransform.mScaling.x, uvTransform.mScaling.y};
    texture.uvOffset = {uvTransform.mTranslation.x, uvTransform.mTranslation.y};
    texture.uvRotation = uvTransform.mRotation;
    texture.addressU = addressMode(mapModes[0]);
    texture.addressV = addressMode(mapModes[1]);
    if (!decodeTexture(imported, resolved, assimpPath, texture))
    {
        log(LogLevel::Warning, "Could not decode material texture '" + resolved.string() + "'");
        return -1;
    }
    const auto duplicateContent = std::find_if(scene.textures.begin(), scene.textures.end(),
                                               [&](const TextureReference& value) {
        return bindingMatches(value) && value.width == texture.width &&
               value.height == texture.height && value.mipCount == texture.mipCount &&
               value.rgba8Pixels == texture.rgba8Pixels;
    });
    if (duplicateContent != scene.textures.end())
        return static_cast<std::int32_t>(std::distance(scene.textures.begin(), duplicateContent));
    scene.textures.push_back(std::move(texture));
    return static_cast<std::int32_t>(scene.textures.size() - 1);
}
} // namespace

AssetLoadResult AssetLoader::load(const std::filesystem::path& path, AssetLoadOptions options) const
{
    Assimp::Importer importer;
    const unsigned flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality |
                           aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | aiProcess_ValidateDataStructure |
                           aiProcess_SortByPType | aiProcess_FlipUVs;
    const aiScene* imported = importer.ReadFile(path.string(), flags);
    if (!imported || !imported->mRootNode)
        return {.error = importer.GetErrorString()};

    Scene scene{};
    scene.name = path.filename().string();
    scene.sourcePath = path;
    const std::filesystem::path basePath = path.parent_path();

    scene.materials.reserve(imported->mNumMaterials);
    for (unsigned materialIndex = 0; materialIndex < imported->mNumMaterials; ++materialIndex)
    {
        aiMaterial* source = imported->mMaterials[materialIndex];
        Material material{};
        aiString name;
        if (source->Get(AI_MATKEY_NAME, name) == AI_SUCCESS)
            material.name = name.C_Str();

        aiColor4D baseColor(1.0f, 1.0f, 1.0f, 1.0f);
        if (source->Get(AI_MATKEY_BASE_COLOR, baseColor) != AI_SUCCESS)
            source->Get(AI_MATKEY_COLOR_DIFFUSE, baseColor);
        material.baseColor = toColor(baseColor);

        aiColor3D emissive(0.0f, 0.0f, 0.0f);
        source->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
        material.emissive = {emissive.r, emissive.g, emissive.b};
        source->Get(AI_MATKEY_METALLIC_FACTOR, material.metallic);
        source->Get(AI_MATKEY_ROUGHNESS_FACTOR, material.roughness);
        source->Get(AI_MATKEY_BUMPSCALING, material.bumpScale);
        source->Get(AI_MATKEY_GLTF_ALPHACUTOFF, material.alphaCutoff);
        source->Get(AI_MATKEY_REFRACTI, material.indexOfRefraction);
        source->Get(AI_MATKEY_TRANSMISSION_FACTOR, material.transmission);
        source->Get(AI_MATKEY_CLEARCOAT_FACTOR, material.clearcoat);
        source->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, material.clearcoatRoughness);
        source->Get(AI_MATKEY_ANISOTROPY_FACTOR, material.anisotropy);
        source->Get(AI_MATKEY_ANISOTROPY_ROTATION, material.anisotropyRotation);
        source->Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, material.sheenRoughness);
        source->Get(AI_MATKEY_VOLUME_ATTENUATION_DISTANCE, material.absorptionDistance);
        aiColor3D sheenColor(0.0f, 0.0f, 0.0f);
        source->Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor);
        material.sheenColor = {sheenColor.r, sheenColor.g, sheenColor.b};
        aiColor3D absorptionColor(1.0f, 1.0f, 1.0f);
        source->Get(AI_MATKEY_VOLUME_ATTENUATION_COLOR, absorptionColor);
        material.absorptionColor = {absorptionColor.r, absorptionColor.g, absorptionColor.b};
        float emissiveIntensity = 1.0f;
        source->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissiveIntensity);
        material.emissive = material.emissive * std::max(emissiveIntensity, 0.0f);

        aiString alphaMode;
        const bool hasExplicitAlphaMode = source->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS;
        if (hasExplicitAlphaMode)
        {
            if (std::strcmp(alphaMode.C_Str(), "MASK") == 0)
                material.alphaMode = AlphaMode::Mask;
            else if (std::strcmp(alphaMode.C_Str(), "BLEND") == 0)
                material.alphaMode = AlphaMode::Blend;
        }
        else
        {
            float opacity = 1.0f;
            if (source->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
                material.baseColor.w *= std::clamp(opacity, 0.0f, 1.0f);
            if (material.baseColor.w < 1.0f)
                material.alphaMode = AlphaMode::Blend;
        }

        int doubleSided = 0;
        source->Get(AI_MATKEY_TWOSIDED, doubleSided);
        material.doubleSided = doubleSided != 0;
        material.baseColorTexture = addTexture(scene, imported, basePath, source, aiTextureType_BASE_COLOR, true);
        if (material.baseColorTexture < 0)
            material.baseColorTexture = addTexture(scene, imported, basePath, source, aiTextureType_DIFFUSE, true);
        material.opacityTexture = addTexture(scene, imported, basePath, source, aiTextureType_OPACITY, false);
        if (!hasExplicitAlphaMode && material.opacityTexture >= 0)
            material.alphaMode = AlphaMode::Blend;
        material.normalTexture = addTexture(scene, imported, basePath, source, aiTextureType_NORMALS, false);
        material.heightTexture = addTexture(scene, imported, basePath, source, aiTextureType_HEIGHT, false);
        if (material.heightTexture >= 0)
        {
            const TextureReference& heightTexture = scene.textures[static_cast<std::size_t>(material.heightTexture)];
            material.heightTextureWidth = heightTexture.width;
            material.heightTextureHeight = heightTexture.height;
        }
        material.metallicRoughnessTexture = addTexture(scene, imported, basePath, source, aiTextureType_METALNESS, false);
        if (material.metallicRoughnessTexture < 0)
            material.metallicRoughnessTexture = addTexture(scene, imported, basePath, source,
                                                           aiTextureType_DIFFUSE_ROUGHNESS, false);
        material.occlusionTexture = addTexture(scene, imported, basePath, source, aiTextureType_AMBIENT_OCCLUSION, false);
        if (material.occlusionTexture < 0)
            material.occlusionTexture = addTexture(scene, imported, basePath, source, aiTextureType_LIGHTMAP, false);
        if (material.occlusionTexture < 0)
            material.occlusionTexture = addTexture(scene, imported, basePath, source, aiTextureType_AMBIENT, false);
        material.emissiveTexture = addTexture(scene, imported, basePath, source, aiTextureType_EMISSIVE, true);
        material.transmissionTexture = addTexture(scene, imported, basePath, source,
                                                  aiTextureType_TRANSMISSION, false);
        material.clearcoatTexture = addTexture(scene, imported, basePath, source, aiTextureType_CLEARCOAT, false, 0);
        material.clearcoatRoughnessTexture = addTexture(scene, imported, basePath, source,
                                                        aiTextureType_CLEARCOAT, false, 1);
        material.clearcoatNormalTexture = addTexture(scene, imported, basePath, source,
                                                     aiTextureType_CLEARCOAT, false, 2);
        material.anisotropyTexture = addTexture(scene, imported, basePath, source,
                                                aiTextureType_ANISOTROPY, false);
        material.sheenColorTexture = addTexture(scene, imported, basePath, source,
                                                aiTextureType_SHEEN, true, 0);
        material.sheenRoughnessTexture = addTexture(scene, imported, basePath, source,
                                                    aiTextureType_SHEEN, false, 1);
        normalizeMaterial(material);
        scene.materials.push_back(std::move(material));
    }
    ensureDefaultPlasticMaterial(scene);

    scene.meshes.reserve(imported->mNumMeshes);
    for (unsigned meshIndex = 0; meshIndex < imported->mNumMeshes; ++meshIndex)
    {
        const aiMesh* source = imported->mMeshes[meshIndex];
        Mesh mesh{};
        mesh.name = source->mName.C_Str();
        mesh.materialIndex = source->mMaterialIndex;
        mesh.vertices.resize(source->mNumVertices);
        for (unsigned vertexIndex = 0; vertexIndex < source->mNumVertices; ++vertexIndex)
        {
            Vertex& vertex = mesh.vertices[vertexIndex];
            const aiVector3D& p = source->mVertices[vertexIndex];
            vertex.position = {p.x, p.y, p.z};
            if (source->HasNormals())
            {
                const aiVector3D& n = source->mNormals[vertexIndex];
                vertex.normal = {n.x, n.y, n.z};
            }
            if (source->HasTangentsAndBitangents())
            {
                const aiVector3D& t = source->mTangents[vertexIndex];
                const aiVector3D& b = source->mBitangents[vertexIndex];
                const Vec3 tangent{t.x, t.y, t.z};
                const Vec3 bitangent{b.x, b.y, b.z};
                const float handedness = dot(cross(vertex.normal, tangent), bitangent) < 0.0f ? -1.0f : 1.0f;
                vertex.tangent = {t.x, t.y, t.z, handedness};
            }
            if (source->HasTextureCoords(0))
            {
                const aiVector3D& uv = source->mTextureCoords[0][vertexIndex];
                vertex.uv = {uv.x, uv.y};
            }
            if (source->HasTextureCoords(1))
            {
                const aiVector3D& uv = source->mTextureCoords[1][vertexIndex];
                vertex.uv1 = {uv.x, uv.y};
            }
            else
                vertex.uv1 = vertex.uv;
        }

        MeshLod baseLod{};
        baseLod.indices.reserve(source->mNumFaces * 3);
        for (unsigned faceIndex = 0; faceIndex < source->mNumFaces; ++faceIndex)
        {
            const aiFace& face = source->mFaces[faceIndex];
            if (face.mNumIndices == 3)
                baseLod.indices.insert(baseLod.indices.end(), face.mIndices, face.mIndices + 3);
        }
        mesh.lods.push_back(std::move(baseLod));
        processMesh(mesh, options.enableOptionalMeshoptimizerPasses);
        scene.meshes.push_back(std::move(mesh));
    }

    std::unordered_map<std::string, Mat4> nodeTransforms;
    std::function<void(const aiNode*, const Mat4&)> visitNode;
    visitNode = [&](const aiNode* node, const Mat4& parentTransform) {
        const Mat4 transform = parentTransform * toMat4(node->mTransformation);
        nodeTransforms.insert_or_assign(node->mName.C_Str(), transform);
        for (unsigned index = 0; index < node->mNumMeshes; ++index)
        {
            scene.instances.push_back({node->mName.C_Str(), node->mMeshes[index], transform, transform});
        }
        for (unsigned child = 0; child < node->mNumChildren; ++child)
            visitNode(node->mChildren[child], transform);
    };
    visitNode(imported->mRootNode, Mat4::identity());

    scene.lights.reserve(imported->mNumLights);
    for (unsigned lightIndex = 0; lightIndex < imported->mNumLights; ++lightIndex)
    {
        const aiLight& source = *imported->mLights[lightIndex];
        if (source.mType == aiLightSource_AMBIENT || source.mType == aiLightSource_UNDEFINED)
            continue;
        const auto transformIt = nodeTransforms.find(source.mName.C_Str());
        const Mat4 transform = transformIt != nodeTransforms.end() ? transformIt->second : Mat4::identity();
        Light light{};
        light.name = source.mName.C_Str();
        light.type = source.mType == aiLightSource_DIRECTIONAL ? LightType::Directional
                   : source.mType == aiLightSource_SPOT ? LightType::Spot
                   : source.mType == aiLightSource_AREA ? LightType::Area
                                                        : LightType::Point;
        light.position = transformPoint(transform, {source.mPosition.x, source.mPosition.y, source.mPosition.z});
        light.direction = transformVector(transform, {source.mDirection.x, source.mDirection.y,
                                                       source.mDirection.z});
        light.color = {std::max(source.mColorDiffuse.r, 0.0f), std::max(source.mColorDiffuse.g, 0.0f),
                       std::max(source.mColorDiffuse.b, 0.0f)};
        light.intensity = 1.0f;
        light.innerCone = std::max(source.mAngleInnerCone, 0.0f);
        light.outerCone = std::max(source.mAngleOuterCone, light.innerCone);
        light.areaSize = {std::max(source.mSize.x, 0.001f), std::max(source.mSize.y, 0.001f)};
        scene.lights.push_back(light);
    }

    if (scene.lights.empty())
        scene.lights.push_back(Light{.name = "Sun", .type = LightType::Directional});
    if (options.addGroundPlane)
        appendGroundPlane(scene, options.enableOptionalMeshoptimizerPasses);

    log(LogLevel::Info, "Loaded scene '" + scene.name + "' with " + std::to_string(scene.meshes.size()) + " meshes");
    return {.scene = std::move(scene)};
}

bool AssetLoader::loadHdrEnvironment(const std::filesystem::path& path,
                                     Environment& environment,
                                     std::string& error) const
{
    const std::string filename = path.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_is_hdr(filename.c_str()))
    {
        error = "The selected file is not a Radiance HDR image";
        return false;
    }
    if (!stbi_info(filename.c_str(), &width, &height, &channels))
    {
        error = stbi_failure_reason() ? stbi_failure_reason() : "Cannot read HDR metadata";
        return false;
    }
    constexpr std::uint64_t maxHdrPixels = 8192ull * 4096ull;
    if (width <= 0 || height <= 0 || width > 16384 || height > 8192 ||
        static_cast<std::uint64_t>(width) * height > maxHdrPixels)
    {
        error = "HDR dimensions are invalid or exceed the 8192x4096 pixel budget";
        return false;
    }
    float* pixels = stbi_loadf(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
    {
        error = stbi_failure_reason() ? stbi_failure_reason() : "Unknown HDR decode error";
        return false;
    }

    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    std::vector<Vec4> decoded;
    decoded.reserve(pixelCount + pixelCount / 3);
    decoded.resize(pixelCount);
    for (std::size_t index = 0; index < pixelCount; ++index)
    {
        const float* source = pixels + index * 4;
        decoded[index] = {std::isfinite(source[0]) ? std::max(source[0], 0.0f) : 0.0f,
                          std::isfinite(source[1]) ? std::max(source[1], 0.0f) : 0.0f,
                          std::isfinite(source[2]) ? std::max(source[2], 0.0f) : 0.0f,
                          1.0f};
    }
    stbi_image_free(pixels);

    std::vector<float> conditionalCdf(static_cast<std::size_t>(height) * (static_cast<std::size_t>(width) + 1));
    std::vector<float> marginalCdf(static_cast<std::size_t>(height) + 1, 0.0f);
    float importanceTotal = 0.0f;
    for (int y = 0; y < height; ++y)
    {
        float* rowCdf = conditionalCdf.data() + static_cast<std::size_t>(y) * (static_cast<std::size_t>(width) + 1);
        rowCdf[0] = 0.0f;
        float rowTotal = 0.0f;
        for (int x = 0; x < width; ++x)
        {
            const Vec4& pixel = decoded[static_cast<std::size_t>(y) * width + x];
            rowTotal += std::max(pixel.x * 0.2126f + pixel.y * 0.7152f + pixel.z * 0.0722f, 1.0e-8f);
            rowCdf[x + 1] = rowTotal;
        }
        for (int x = 1; x <= width; ++x)
            rowCdf[x] /= rowTotal;
        const float theta = (static_cast<float>(y) + 0.5f) * kPi / static_cast<float>(height);
        importanceTotal += rowTotal * std::sin(theta);
        marginalCdf[y + 1] = importanceTotal;
    }
    for (int y = 1; y <= height; ++y)
        marginalCdf[y] /= importanceTotal;

    std::uint32_t mipCount = 1;
    std::uint32_t previousWidth = static_cast<std::uint32_t>(width);
    std::uint32_t previousHeight = static_cast<std::uint32_t>(height);
    std::size_t previousOffset = 0;
    while (previousWidth > 1 || previousHeight > 1)
    {
        const std::uint32_t mipWidth = std::max(previousWidth / 2, 1u);
        const std::uint32_t mipHeight = std::max(previousHeight / 2, 1u);
        const std::size_t mipOffset = decoded.size();
        decoded.resize(mipOffset + static_cast<std::size_t>(mipWidth) * mipHeight);
        for (std::uint32_t y = 0; y < mipHeight; ++y)
        {
            const std::uint32_t y0 = std::min(y * 2, previousHeight - 1);
            const std::uint32_t y1 = std::min(y0 + 1, previousHeight - 1);
            for (std::uint32_t x = 0; x < mipWidth; ++x)
            {
                const std::uint32_t x0 = std::min(x * 2, previousWidth - 1);
                const std::uint32_t x1 = std::min(x0 + 1, previousWidth - 1);
                const Vec4& a = decoded[previousOffset + static_cast<std::size_t>(y0) * previousWidth + x0];
                const Vec4& b = decoded[previousOffset + static_cast<std::size_t>(y0) * previousWidth + x1];
                const Vec4& c = decoded[previousOffset + static_cast<std::size_t>(y1) * previousWidth + x0];
                const Vec4& d = decoded[previousOffset + static_cast<std::size_t>(y1) * previousWidth + x1];
                decoded[mipOffset + static_cast<std::size_t>(y) * mipWidth + x] =
                    {(a.x + b.x + c.x + d.x) * 0.25f,
                     (a.y + b.y + c.y + d.y) * 0.25f,
                     (a.z + b.z + c.z + d.z) * 0.25f, 1.0f};
            }
        }
        previousOffset = mipOffset;
        previousWidth = mipWidth;
        previousHeight = mipHeight;
        ++mipCount;
    }

    environment.hdrPath = path;
    environment.hdrWidth = static_cast<std::uint32_t>(width);
    environment.hdrHeight = static_cast<std::uint32_t>(height);
    environment.hdrMipCount = mipCount;
    environment.hdrPixels = std::move(decoded);
    environment.hdrConditionalCdf = std::move(conditionalCdf);
    environment.hdrMarginalCdf = std::move(marginalCdf);
    environment.hdrImportanceTotal = importanceTotal;
    generateIblAtlas(environment);
    environment.mode = GlobalLightMode::HdrEnvironment;
    error.clear();
    return true;
}

Scene AssetLoader::createProceduralCube(AssetLoadOptions options) const
{
    Scene scene = makeDefaultScene();
    ensureDefaultPlasticMaterial(scene);
    Mesh mesh{};
    mesh.name = "Cube";
    mesh.materialIndex = 0;

    const std::array<Vec3, 8> positions{{
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
    }};
    for (const Vec3 position : positions)
        mesh.vertices.push_back({position, normalize(position), {1, 0, 0, 1}, {0, 0}});

    MeshLod lod{};
    lod.indices = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    };
    mesh.lods.push_back(std::move(lod));
    processMesh(mesh, options.enableOptionalMeshoptimizerPasses);
    scene.meshes.push_back(std::move(mesh));
    scene.instances.push_back({"Cube", 0, Mat4::identity(), Mat4::identity()});
    if (options.addGroundPlane)
        appendGroundPlane(scene, options.enableOptionalMeshoptimizerPasses);
    return scene;
}

Scene AssetLoader::createPbrMaterialComparisonScene(AssetLoadOptions options) const
{
    Scene scene = createProceduralCube({.enableOptionalMeshoptimizerPasses = options.enableOptionalMeshoptimizerPasses,
                                        .addGroundPlane = false});
    scene.name = "PBR Material Comparison";
    Mesh referenceMesh{};
    referenceMesh.name = "Reference Sphere";
    constexpr std::uint32_t longitudeSegments = 32;
    constexpr std::uint32_t latitudeSegments = 16;
    for (std::uint32_t latitude = 0; latitude <= latitudeSegments; ++latitude)
    {
        const float v = static_cast<float>(latitude) / latitudeSegments;
        const float theta = v * kPi;
        const float sineTheta = std::sin(theta);
        const float cosineTheta = std::cos(theta);
        for (std::uint32_t longitude = 0; longitude <= longitudeSegments; ++longitude)
        {
            const float u = static_cast<float>(longitude) / longitudeSegments;
            const float phi = u * 2.0f * kPi;
            const Vec3 normal{sineTheta * std::cos(phi), cosineTheta, sineTheta * std::sin(phi)};
            const Vec3 position = normal * 1.08f;
            const Vec3 tangent{-std::sin(phi), 0.0f, std::cos(phi)};
            referenceMesh.vertices.push_back({position, normal, {tangent.x, tangent.y, tangent.z, 1.0f}, {u, v}});
        }
    }
    MeshLod sphereLod{};
    for (std::uint32_t latitude = 0; latitude < latitudeSegments; ++latitude)
    {
        for (std::uint32_t longitude = 0; longitude < longitudeSegments; ++longitude)
        {
            const std::uint32_t a = latitude * (longitudeSegments + 1) + longitude;
            const std::uint32_t b = a + longitudeSegments + 1;
            if (latitude != 0)
                sphereLod.indices.insert(sphereLod.indices.end(), {a, a + 1, b});
            if (latitude + 1 != latitudeSegments)
                sphereLod.indices.insert(sphereLod.indices.end(), {a + 1, b + 1, b});
        }
    }
    referenceMesh.lods.push_back(std::move(sphereLod));
    processMesh(referenceMesh, options.enableOptionalMeshoptimizerPasses);
    // The diagnostic spheres must remain a complete reference under every camera angle; their
    // two pole fans are intentionally excluded from cone culling while frustum culling stays active.
    for (Meshlet& meshlet : referenceMesh.lods.front().meshlets)
        meshlet.normalCone = {};
    scene.meshes.clear();
    scene.instances.clear();
    scene.materials.clear();

    scene.materials = {
        {.name = "Dielectric Smooth", .baseColor = {0.72f, 0.18f, 0.08f, 1.0f}, .roughness = 0.12f},
        {.name = "Dielectric Rough", .baseColor = {0.18f, 0.42f, 0.72f, 1.0f}, .roughness = 0.72f},
        {.name = "Brushed Copper", .baseColor = {0.95f, 0.43f, 0.18f, 1.0f}, .metallic = 1.0f,
         .roughness = 0.3f, .anisotropy = 0.82f, .anisotropyRotation = 0.2f},
        {.name = "Clearcoat Plastic", .baseColor = {0.08f, 0.32f, 0.08f, 1.0f}, .roughness = 0.42f,
         .clearcoat = 1.0f, .clearcoatRoughness = 0.08f},
        {.name = "Clearcoat Metal", .baseColor = {0.75f, 0.55f, 0.16f, 1.0f}, .metallic = 1.0f,
         .roughness = 0.3f, .clearcoat = 1.0f, .clearcoatRoughness = 0.16f},
        {.name = "Absorbing Glass", .baseColor = {0.96f, 0.99f, 1.0f, 1.0f}, .roughness = 0.06f,
         .transmission = 1.0f, .indexOfRefraction = 1.5f, .absorptionColor = {0.62f, 0.88f, 0.96f},
         .absorptionDistance = 2.0f},
        {.name = "Cloth Sheen", .baseColor = {0.22f, 0.025f, 0.04f, 1.0f}, .roughness = 0.72f,
         .sheenColor = {0.8f, 0.22f, 0.28f}, .sheenRoughness = 0.58f},
        {.name = "Emissive", .baseColor = {0.02f, 0.02f, 0.02f, 1.0f}, .emissive = {8.0f, 2.5f, 0.6f},
         .roughness = 0.45f},
        {.name = "Wax SSS", .baseColor = {0.72f, 0.12f, 0.06f, 1.0f}, .roughness = 0.5f,
         .subsurface = 0.85f, .subsurfaceColor = {1.0f, 0.22f, 0.12f}, .subsurfaceRadius = 0.55f},
        {.name = "Homogeneous Volume", .baseColor = {0.9f, 0.96f, 1.0f, 1.0f}, .roughness = 0.12f,
         .transmission = 1.0f, .indexOfRefraction = 1.05f, .volumeAbsorption = {0.08f, 0.03f, 0.01f},
         .volumeDensity = 0.8f, .volumeScattering = {0.32f, 0.38f, 0.45f}, .volumeAnisotropy = 0.35f},
        {.name = "Default Plastic", .baseColor = {0.75f, 0.75f, 0.75f, 1.0f}, .roughness = 0.5f},
    };

    for (std::uint32_t index = 0; index < scene.materials.size() - 1; ++index)
    {
        Mesh mesh = referenceMesh;
        mesh.name = scene.materials[index].name;
        mesh.materialIndex = index;
        const std::uint32_t meshIndex = static_cast<std::uint32_t>(scene.meshes.size());
        scene.meshes.push_back(std::move(mesh));
        Mat4 transform = Mat4::identity();
        transform.m[12] = (static_cast<float>(index % 5) - 2.0f) * 2.7f;
        transform.m[13] = static_cast<float>(index / 5) * 2.7f;
        scene.instances.push_back({scene.materials[index].name, meshIndex, transform, transform});
    }
    if (options.addGroundPlane)
        appendGroundPlane(scene, options.enableOptionalMeshoptimizerPasses);
    return scene;
}

void AssetLoader::appendGroundPlane(Scene& scene, bool enableOptionalPasses)
{
    Vec3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
    bool hasPoint = false;
    const auto includeMesh = [&](const Mesh& mesh, const Mat4& transform) {
        if (mesh.isGroundPlane)
            return;
        for (const Vertex& vertex : mesh.vertices)
        {
            const Vec3& p = vertex.position;
            const Vec3 world{
                transform.m[0] * p.x + transform.m[4] * p.y + transform.m[8] * p.z + transform.m[12],
                transform.m[1] * p.x + transform.m[5] * p.y + transform.m[9] * p.z + transform.m[13],
                transform.m[2] * p.x + transform.m[6] * p.y + transform.m[10] * p.z + transform.m[14],
            };
            minimum.x = std::min(minimum.x, world.x);
            minimum.y = std::min(minimum.y, world.y);
            minimum.z = std::min(minimum.z, world.z);
            maximum.x = std::max(maximum.x, world.x);
            maximum.y = std::max(maximum.y, world.y);
            maximum.z = std::max(maximum.z, world.z);
            hasPoint = true;
        }
    };

    if (!scene.instances.empty())
    {
        for (const Instance& instance : scene.instances)
        {
            if (instance.meshIndex < scene.meshes.size())
                includeMesh(scene.meshes[instance.meshIndex], instance.transform);
        }
    }
    else
    {
        for (const Mesh& mesh : scene.meshes)
            includeMesh(mesh, Mat4::identity());
    }
    if (!hasPoint)
        return;

    const Vec3 extent = maximum - minimum;
    const Vec3 center = (minimum + maximum) * 0.5f;
    const float halfExtent = std::max({extent.x, extent.z, extent.y * 0.75f, 1.0f}) * 0.75f;
    const float verticalOffset = std::max(extent.y * 0.00001f, 0.0001f);
    const float groundY = minimum.y - verticalOffset;

    const std::uint32_t materialIndex = ensureDefaultPlasticMaterial(scene);

    Mesh ground{};
    ground.name = "Ground Plane";
    ground.materialIndex = materialIndex;
    ground.isGroundPlane = true;
    ground.vertices = {
        {{center.x - halfExtent, groundY, center.z - halfExtent}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        {{center.x + halfExtent, groundY, center.z - halfExtent}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        {{center.x + halfExtent, groundY, center.z + halfExtent}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        {{center.x - halfExtent, groundY, center.z + halfExtent}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    };
    MeshLod lod{};
    lod.indices = {0, 2, 1, 0, 3, 2};
    ground.lods.push_back(std::move(lod));
    processMesh(ground, enableOptionalPasses);

    const std::uint32_t groundMeshIndex = static_cast<std::uint32_t>(scene.meshes.size());
    scene.meshes.push_back(std::move(ground));
    scene.instances.push_back({"Ground Plane", groundMeshIndex, Mat4::identity(), Mat4::identity()});
}

std::uint32_t AssetLoader::ensureDefaultPlasticMaterial(Scene& scene)
{
    const Material defaultPlastic = makeDefaultPlasticMaterial();
    const auto matchingMaterial = std::find_if(scene.materials.begin(), scene.materials.end(), [&](const Material& material) {
        return material.name == defaultPlastic.name && material.baseColor.x == defaultPlastic.baseColor.x &&
               material.baseColor.y == defaultPlastic.baseColor.y && material.baseColor.z == defaultPlastic.baseColor.z &&
               material.metallic == defaultPlastic.metallic && material.roughness == defaultPlastic.roughness;
    });
    if (matchingMaterial != scene.materials.end())
        return static_cast<std::uint32_t>(std::distance(scene.materials.begin(), matchingMaterial));

    const std::uint32_t materialId = static_cast<std::uint32_t>(scene.materials.size());
    scene.materials.push_back(defaultPlastic);
    return materialId;
}

void AssetLoader::setDefaultPlasticOverride(Scene& scene, bool enabled)
{
    scene.materialOverrideId = enabled ? ensureDefaultPlasticMaterial(scene) : kInvalidMaterialId;
}

void AssetLoader::processMesh(Mesh& mesh, bool enableOptionalPasses)
{
    if (mesh.vertices.empty() || mesh.lods.empty() || mesh.lods.front().indices.empty())
        return;

    MeshLod& base = mesh.lods.front();
    if (enableOptionalPasses)
    {
        std::vector<unsigned int> remap(mesh.vertices.size());
        const std::size_t uniqueVertices = meshopt_generateVertexRemap(
            remap.data(), base.indices.data(), base.indices.size(), mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex));

        std::vector<Vertex> remappedVertices(uniqueVertices);
        std::vector<std::uint32_t> remappedIndices(base.indices.size());
        meshopt_remapVertexBuffer(remappedVertices.data(), mesh.vertices.data(), mesh.vertices.size(), sizeof(Vertex), remap.data());
        meshopt_remapIndexBuffer(remappedIndices.data(), base.indices.data(), base.indices.size(), remap.data());
        mesh.vertices = std::move(remappedVertices);
        base.indices = std::move(remappedIndices);

        meshopt_optimizeVertexCache(base.indices.data(), base.indices.data(), base.indices.size(), mesh.vertices.size());
        meshopt_optimizeOverdraw(base.indices.data(), base.indices.data(), base.indices.size(), &mesh.vertices[0].position.x,
                                 mesh.vertices.size(), sizeof(Vertex), 1.05f);
        meshopt_optimizeVertexFetch(mesh.vertices.data(), base.indices.data(), base.indices.size(), mesh.vertices.data(),
                                    mesh.vertices.size(), sizeof(Vertex));
    }

    const auto buildMeshlets = [&](MeshLod& lod, bool buildBounds) {
        constexpr std::size_t kMaxVertices = 64;
        constexpr std::size_t kMaxTriangles = 124;
        const std::size_t maxMeshlets = meshopt_buildMeshletsBound(lod.indices.size(), kMaxVertices, kMaxTriangles);
        std::vector<meshopt_Meshlet> nativeMeshlets(maxMeshlets);
        lod.meshletVertices.resize(maxMeshlets * kMaxVertices);
        lod.meshletTriangles.resize(maxMeshlets * kMaxTriangles * 3);
        const std::size_t meshletCount = meshopt_buildMeshlets(
            nativeMeshlets.data(), lod.meshletVertices.data(), lod.meshletTriangles.data(), lod.indices.data(),
            lod.indices.size(), &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex),
            kMaxVertices, kMaxTriangles, 0.5f);
        nativeMeshlets.resize(meshletCount);
        lod.meshlets.clear();
        lod.meshlets.reserve(meshletCount);
        for (const meshopt_Meshlet& native : nativeMeshlets)
        {
            Meshlet meshlet{native.vertex_offset, native.triangle_offset, native.vertex_count, native.triangle_count};
            if (buildBounds)
            {
                const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                    lod.meshletVertices.data() + native.vertex_offset,
                    lod.meshletTriangles.data() + native.triangle_offset,
                    native.triangle_count, &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex));
                meshlet.boundingSphere = {bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius};
                meshlet.normalCone = {static_cast<float>(bounds.cone_axis_s8[0]) / 127.0f,
                                      static_cast<float>(bounds.cone_axis_s8[1]) / 127.0f,
                                      static_cast<float>(bounds.cone_axis_s8[2]) / 127.0f,
                                      static_cast<float>(bounds.cone_cutoff_s8) / 127.0f};
            }
            lod.meshlets.push_back(meshlet);
        }
        if (!nativeMeshlets.empty())
        {
            const meshopt_Meshlet& last = nativeMeshlets.back();
            lod.meshletVertices.resize(last.vertex_offset + last.vertex_count);
            lod.meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));
        }
    };
    buildMeshlets(base, enableOptionalPasses);

    if (!enableOptionalPasses)
        return;

    constexpr std::array<float, 2> lodRatios{0.5f, 0.25f};
    constexpr std::array<float, 11> attributeWeights{
        1.0f, 1.0f, 1.0f,       // normal
        0.25f, 0.25f, 0.25f, 0.1f, // tangent and handedness
        1.0f, 1.0f,              // UV0
        1.0f, 1.0f,              // UV1
    };
    const std::vector<std::uint32_t> sourceIndices = base.indices;
    for (const float ratio : lodRatios)
    {
        const std::size_t targetCount = std::max<std::size_t>(3, static_cast<std::size_t>(sourceIndices.size() * ratio) / 3 * 3);
        MeshLod lod{};
        lod.indices.resize(sourceIndices.size());
        lod.indices.resize(meshopt_simplifyWithAttributes(
            lod.indices.data(), sourceIndices.data(), sourceIndices.size(), &mesh.vertices[0].position.x,
            mesh.vertices.size(), sizeof(Vertex), &mesh.vertices[0].normal.x, sizeof(Vertex),
            attributeWeights.data(), attributeWeights.size(), nullptr, targetCount, 1e-2f,
            meshopt_SimplifyLockBorder, &lod.simplificationError));
        if (lod.indices.size() >= 3 && lod.indices.size() < base.indices.size())
        {
            meshopt_optimizeVertexCache(lod.indices.data(), lod.indices.data(), lod.indices.size(), mesh.vertices.size());
            buildMeshlets(lod, true);
            mesh.lods.push_back(std::move(lod));
        }
    }
}
} // namespace vor
