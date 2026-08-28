#pragma once

#include "Core/Scene.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vor
{
struct GpuLightAlias
{
    float probability{1.0f};
    std::uint32_t alias{};
    float selectionPdf{};
    float padding{};
};
static_assert(sizeof(GpuLightAlias) == 16);

struct LightAliasTable
{
    std::vector<GpuLightAlias> entries;
    float totalPower{};
};

[[nodiscard]] inline float lightSamplingPower(const Light& light)
{
    const float luminance = std::max(light.color.x * 0.2126f + light.color.y * 0.7152f +
                                         light.color.z * 0.0722f,
                                     0.0f);
    float power = luminance * std::max(light.intensity, 0.0f);
    if (light.type == LightType::Area)
        power *= std::max(light.areaSize.x * light.areaSize.y, 1.0e-6f);
    return power;
}

[[nodiscard]] inline LightAliasTable buildLightAliasTable(const std::vector<Light>& lights)
{
    const std::size_t count = std::max<std::size_t>(lights.size(), 1);
    LightAliasTable result;
    result.entries.resize(count);
    std::vector<float> weights(count, 0.0f);
    for (std::size_t index = 0; index < lights.size(); ++index)
    {
        weights[index] = lightSamplingPower(lights[index]);
        result.totalPower += weights[index];
    }
    if (result.totalPower <= 1.0e-8f)
        return result;

    std::vector<float> scaled(count);
    std::vector<std::uint32_t> small;
    std::vector<std::uint32_t> large;
    small.reserve(count);
    large.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        result.entries[index].selectionPdf = weights[index] / result.totalPower;
        scaled[index] = result.entries[index].selectionPdf * static_cast<float>(count);
        (scaled[index] < 1.0f ? small : large).push_back(index);
    }
    while (!small.empty() && !large.empty())
    {
        const std::uint32_t low = small.back();
        small.pop_back();
        const std::uint32_t high = large.back();
        large.pop_back();
        result.entries[low].probability = scaled[low];
        result.entries[low].alias = high;
        scaled[high] = scaled[high] + scaled[low] - 1.0f;
        (scaled[high] < 1.0f ? small : large).push_back(high);
    }
    for (const std::uint32_t index : large)
    {
        result.entries[index].probability = 1.0f;
        result.entries[index].alias = index;
    }
    for (const std::uint32_t index : small)
    {
        result.entries[index].probability = 1.0f;
        result.entries[index].alias = index;
    }
    return result;
}
} // namespace vor
