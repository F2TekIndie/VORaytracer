#include "Assets/AssetLoader.h"

#include "Core/Log.h"

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
    if (material->GetTextureCount(type) <= textureIndex ||
        material->GetTexture(type, textureIndex, &texturePath) != AI_SUCCESS)
        return -1;

    const std::string assimpPath = texturePath.C_Str();
    std::filesystem::path resolved = std::filesystem::path(assimpPath);
    if (!resolved.is_absolute() && !resolved.native().starts_with(L"*"))
        resolved = basePath / resolved;
    resolved = resolved.lexically_normal();

    const auto found = std::find_if(scene.textures.begin(), scene.textures.end(), [&](const TextureReference& value) {
        return value.path == resolved && value.srgb == srgb;
    });
    if (found != scene.textures.end())
        return static_cast<std::int32_t>(std::distance(scene.textures.begin(), found));

    TextureReference texture{};
    texture.path = resolved;
    texture.srgb = srgb;
    if (!decodeTexture(imported, resolved, assimpPath, texture))
    {
        log(LogLevel::Warning, "Could not decode material texture '" + resolved.string() + "'");
        return -1;
    }
    const auto duplicateContent = std::find_if(scene.textures.begin(), scene.textures.end(),
                                               [&](const TextureReference& value) {
        return value.srgb == texture.srgb && value.width == texture.width &&
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
                           aiProcess_SortByPType;
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
        if (source->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS)
        {
            if (std::strcmp(alphaMode.C_Str(), "MASK") == 0)
                material.alphaMode = AlphaMode::Mask;
            else if (std::strcmp(alphaMode.C_Str(), "BLEND") == 0)
                material.alphaMode = AlphaMode::Blend;
        }

        int doubleSided = 0;
        source->Get(AI_MATKEY_TWOSIDED, doubleSided);
        material.doubleSided = doubleSided != 0;
        material.baseColorTexture = addTexture(scene, imported, basePath, source, aiTextureType_BASE_COLOR, true);
        if (material.baseColorTexture < 0)
            material.baseColorTexture = addTexture(scene, imported, basePath, source, aiTextureType_DIFFUSE, true);
        material.normalTexture = addTexture(scene, imported, basePath, source, aiTextureType_NORMALS, false);
        if (material.normalTexture < 0)
            material.normalTexture = addTexture(scene, imported, basePath, source, aiTextureType_HEIGHT, false);
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

    std::function<void(const aiNode*, const Mat4&)> visitNode;
    visitNode = [&](const aiNode* node, const Mat4& parentTransform) {
        const Mat4 transform = parentTransform * toMat4(node->mTransformation);
        for (unsigned index = 0; index < node->mNumMeshes; ++index)
        {
            scene.instances.push_back({node->mName.C_Str(), node->mMeshes[index], transform, transform});
        }
        for (unsigned child = 0; child < node->mNumChildren; ++child)
            visitNode(node->mChildren[child], transform);
    };
    visitNode(imported->mRootNode, Mat4::identity());

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
                sphereLod.indices.insert(sphereLod.indices.end(), {a, b, a + 1});
            if (latitude + 1 != latitudeSegments)
                sphereLod.indices.insert(sphereLod.indices.end(), {a + 1, b, b + 1});
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

    constexpr std::size_t kMaxVertices = 64;
    constexpr std::size_t kMaxTriangles = 124;
    const std::size_t maxMeshlets = meshopt_buildMeshletsBound(base.indices.size(), kMaxVertices, kMaxTriangles);
    std::vector<meshopt_Meshlet> nativeMeshlets(maxMeshlets);
    base.meshletVertices.resize(maxMeshlets * kMaxVertices);
    base.meshletTriangles.resize(maxMeshlets * kMaxTriangles * 3);
    const std::size_t meshletCount = meshopt_buildMeshlets(
        nativeMeshlets.data(), base.meshletVertices.data(), base.meshletTriangles.data(), base.indices.data(),
        base.indices.size(), &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex), kMaxVertices, kMaxTriangles, 0.5f);
    nativeMeshlets.resize(meshletCount);
    base.meshlets.reserve(meshletCount);
    for (const meshopt_Meshlet& native : nativeMeshlets)
    {
        Meshlet meshlet{native.vertex_offset, native.triangle_offset, native.vertex_count, native.triangle_count};
        if (enableOptionalPasses)
        {
            const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                base.meshletVertices.data() + native.vertex_offset,
                base.meshletTriangles.data() + native.triangle_offset,
                native.triangle_count, &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex));
            meshlet.boundingSphere = {bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius};
            meshlet.normalCone = {static_cast<float>(bounds.cone_axis_s8[0]) / 127.0f,
                                  static_cast<float>(bounds.cone_axis_s8[1]) / 127.0f,
                                  static_cast<float>(bounds.cone_axis_s8[2]) / 127.0f,
                                  static_cast<float>(bounds.cone_cutoff_s8) / 127.0f};
        }
        base.meshlets.push_back(meshlet);
    }

    if (!nativeMeshlets.empty())
    {
        const meshopt_Meshlet& last = nativeMeshlets.back();
        base.meshletVertices.resize(last.vertex_offset + last.vertex_count);
        base.meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));
    }

    if (!enableOptionalPasses)
        return;

    constexpr std::array<float, 2> lodRatios{0.5f, 0.25f};
    const std::vector<std::uint32_t> sourceIndices = base.indices;
    for (const float ratio : lodRatios)
    {
        const std::size_t targetCount = std::max<std::size_t>(3, static_cast<std::size_t>(sourceIndices.size() * ratio) / 3 * 3);
        MeshLod lod{};
        lod.indices.resize(sourceIndices.size());
        lod.indices.resize(meshopt_simplify(lod.indices.data(), sourceIndices.data(), sourceIndices.size(),
                                            &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex), targetCount,
                                            1e-2f, meshopt_SimplifyLockBorder, &lod.simplificationError));
        if (lod.indices.size() >= 3 && lod.indices.size() < base.indices.size())
        {
            meshopt_optimizeVertexCache(lod.indices.data(), lod.indices.data(), lod.indices.size(), mesh.vertices.size());
            mesh.lods.push_back(std::move(lod));
        }
    }
}
} // namespace vor
