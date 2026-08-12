#pragma once

#include "Core/Scene.h"

#include <filesystem>
#include <string>

namespace vor
{
struct AssetLoadResult
{
    Scene scene;
    std::string error;

    [[nodiscard]] explicit operator bool() const { return error.empty(); }
};

struct AssetLoadOptions
{
    bool enableOptionalMeshoptimizerPasses{true};
    bool addGroundPlane{false};
};

class AssetLoader
{
public:
    AssetLoadResult load(const std::filesystem::path& path, AssetLoadOptions options = {}) const;
    bool loadHdrEnvironment(const std::filesystem::path& path, Environment& environment, std::string& error) const;
    Scene createProceduralCube(AssetLoadOptions options = {}) const;
    static std::uint32_t ensureDefaultPlasticMaterial(Scene& scene);
    static void setDefaultPlasticOverride(Scene& scene, bool enabled);

private:
    static void appendGroundPlane(Scene& scene, bool enableOptionalPasses);
    static void processMesh(Mesh& mesh, bool enableOptionalPasses);
};
} // namespace vor
