#include "Assets/AssetLoader.h"

#include "Core/Log.h"

#include <assimp/Importer.hpp>
#include <assimp/GltfMaterial.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <meshoptimizer.h>

#include <algorithm>
#include <array>
#include <cstring>
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

std::int32_t addTexture(Scene& scene, const std::filesystem::path& basePath, aiMaterial* material, aiTextureType type, bool srgb)
{
    aiString texturePath;
    if (material->GetTextureCount(type) == 0 || material->GetTexture(type, 0, &texturePath) != AI_SUCCESS)
        return -1;

    std::filesystem::path resolved = std::filesystem::path(texturePath.C_Str());
    if (!resolved.is_absolute() && !resolved.native().starts_with(L"*"))
        resolved = basePath / resolved;

    const auto found = std::find_if(scene.textures.begin(), scene.textures.end(), [&](const TextureReference& value) {
        return value.path == resolved && value.srgb == srgb;
    });
    if (found != scene.textures.end())
        return static_cast<std::int32_t>(std::distance(scene.textures.begin(), found));

    scene.textures.push_back({resolved.lexically_normal(), srgb});
    return static_cast<std::int32_t>(scene.textures.size() - 1);
}
} // namespace

AssetLoadResult AssetLoader::load(const std::filesystem::path& path) const
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
        material.baseColorTexture = addTexture(scene, basePath, source, aiTextureType_BASE_COLOR, true);
        if (material.baseColorTexture < 0)
            material.baseColorTexture = addTexture(scene, basePath, source, aiTextureType_DIFFUSE, true);
        material.normalTexture = addTexture(scene, basePath, source, aiTextureType_NORMALS, false);
        material.metallicRoughnessTexture = addTexture(scene, basePath, source, aiTextureType_METALNESS, false);
        material.occlusionTexture = addTexture(scene, basePath, source, aiTextureType_AMBIENT_OCCLUSION, false);
        material.emissiveTexture = addTexture(scene, basePath, source, aiTextureType_EMISSIVE, true);
        scene.materials.push_back(std::move(material));
    }

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
                vertex.tangent = {t.x, t.y, t.z, 1.0f};
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
        optimizeMesh(mesh);
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
        scene.lights.push_back(Light{.name = "Sun"});

    log(LogLevel::Info, "Loaded scene '" + scene.name + "' with " + std::to_string(scene.meshes.size()) + " meshes");
    return {.scene = std::move(scene)};
}

Scene AssetLoader::createProceduralCube() const
{
    Scene scene = makeDefaultScene();
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
    optimizeMesh(mesh);
    scene.meshes.push_back(std::move(mesh));
    scene.instances.push_back({"Cube", 0, Mat4::identity(), Mat4::identity()});
    return scene;
}

void AssetLoader::optimizeMesh(Mesh& mesh)
{
    if (mesh.vertices.empty() || mesh.lods.empty() || mesh.lods.front().indices.empty())
        return;

    MeshLod& base = mesh.lods.front();
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
        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            base.meshletVertices.data() + native.vertex_offset,
            base.meshletTriangles.data() + native.triangle_offset,
            native.triangle_count, &mesh.vertices[0].position.x, mesh.vertices.size(), sizeof(Vertex));
        base.meshlets.push_back({native.vertex_offset, native.triangle_offset, native.vertex_count, native.triangle_count,
                                 {bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius},
                                 {static_cast<float>(bounds.cone_axis_s8[0]) / 127.0f,
                                  static_cast<float>(bounds.cone_axis_s8[1]) / 127.0f,
                                  static_cast<float>(bounds.cone_axis_s8[2]) / 127.0f,
                                  static_cast<float>(bounds.cone_cutoff_s8) / 127.0f}});
    }

    if (!nativeMeshlets.empty())
    {
        const meshopt_Meshlet& last = nativeMeshlets.back();
        base.meshletVertices.resize(last.vertex_offset + last.vertex_count);
        base.meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));
    }

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
