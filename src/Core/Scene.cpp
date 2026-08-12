#include "Core/Scene.h"

namespace vor
{
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
