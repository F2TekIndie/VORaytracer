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

class AssetLoader
{
public:
    AssetLoadResult load(const std::filesystem::path& path) const;
    Scene createProceduralCube() const;

private:
    static void optimizeMesh(Mesh& mesh);
};
} // namespace vor

