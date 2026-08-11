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
    bool overrideWithDefaultPlastic{false};
    bool addGroundPlane{false};
};

class AssetLoader
{
public:
    AssetLoadResult load(const std::filesystem::path& path, AssetLoadOptions options = {}) const;
    Scene createProceduralCube(AssetLoadOptions options = {}) const;

private:
    static void appendGroundPlane(Scene& scene, bool enableOptionalPasses);
    static void processMesh(Mesh& mesh, bool enableOptionalPasses);
};
} // namespace vor
