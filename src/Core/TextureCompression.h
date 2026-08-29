#pragma once

#include "Core/Scene.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace vor
{
namespace detail
{
inline std::uint16_t packRgb565(const std::array<std::uint8_t, 3>& c)
{
    return static_cast<std::uint16_t>(((c[0] * 31u + 127u) / 255u) << 11u |
                                      ((c[1] * 63u + 127u) / 255u) << 5u |
                                      ((c[2] * 31u + 127u) / 255u));
}

inline std::array<std::uint8_t, 3> unpackRgb565(std::uint16_t c)
{
    return {static_cast<std::uint8_t>((((c >> 11u) & 31u) * 255u + 15u) / 31u),
            static_cast<std::uint8_t>((((c >> 5u) & 63u) * 255u + 31u) / 63u),
            static_cast<std::uint8_t>(((c & 31u) * 255u + 15u) / 31u)};
}

inline void encodeBc3Block(const std::array<std::array<std::uint8_t, 4>, 16>& pixels,
                           std::uint8_t* destination)
{
    std::uint8_t alphaMin = 255, alphaMax = 0;
    std::array<std::uint8_t, 3> colorMin{255, 255, 255}, colorMax{};
    for (const auto& pixel : pixels)
    {
        alphaMin = std::min(alphaMin, pixel[3]);
        alphaMax = std::max(alphaMax, pixel[3]);
        for (std::uint32_t channel = 0; channel < 3; ++channel)
        {
            colorMin[channel] = std::min(colorMin[channel], pixel[channel]);
            colorMax[channel] = std::max(colorMax[channel], pixel[channel]);
        }
    }

    destination[0] = alphaMax;
    destination[1] = alphaMin;
    std::array<std::uint8_t, 8> alphaPalette{alphaMax, alphaMin};
    if (alphaMax > alphaMin)
    {
        for (std::uint32_t i = 1; i <= 6; ++i)
            alphaPalette[i + 1] = static_cast<std::uint8_t>(((7u - i) * alphaMax + i * alphaMin + 3u) / 7u);
    }
    else
    {
        for (std::uint32_t i = 1; i <= 4; ++i)
            alphaPalette[i + 1] = static_cast<std::uint8_t>(((5u - i) * alphaMax + i * alphaMin + 2u) / 5u);
        alphaPalette[6] = 0;
        alphaPalette[7] = 255;
    }
    std::uint64_t alphaIndices = 0;
    for (std::uint32_t p = 0; p < 16; ++p)
    {
        std::uint32_t best = 0, bestError = UINT32_MAX;
        for (std::uint32_t i = 0; i < 8; ++i)
        {
            const int delta = static_cast<int>(pixels[p][3]) - alphaPalette[i];
            const auto error = static_cast<std::uint32_t>(delta * delta);
            if (error < bestError) { best = i; bestError = error; }
        }
        alphaIndices |= static_cast<std::uint64_t>(best) << (p * 3u);
    }
    for (std::uint32_t byte = 0; byte < 6; ++byte)
        destination[2 + byte] = static_cast<std::uint8_t>(alphaIndices >> (byte * 8u));

    std::uint16_t color0 = packRgb565(colorMax);
    std::uint16_t color1 = packRgb565(colorMin);
    if (color0 <= color1)
    {
        if (color1 != UINT16_MAX) color0 = static_cast<std::uint16_t>(color1 + 1u);
        else if (color1 > 0) color1 = static_cast<std::uint16_t>(color1 - 1u);
    }
    destination[8] = static_cast<std::uint8_t>(color0);
    destination[9] = static_cast<std::uint8_t>(color0 >> 8u);
    destination[10] = static_cast<std::uint8_t>(color1);
    destination[11] = static_cast<std::uint8_t>(color1 >> 8u);
    std::array<std::array<std::uint8_t, 3>, 4> palette{};
    palette[0] = unpackRgb565(color0);
    palette[1] = unpackRgb565(color1);
    for (std::uint32_t channel = 0; channel < 3; ++channel)
    {
        palette[2][channel] = static_cast<std::uint8_t>((2u * palette[0][channel] + palette[1][channel] + 1u) / 3u);
        palette[3][channel] = static_cast<std::uint8_t>((palette[0][channel] + 2u * palette[1][channel] + 1u) / 3u);
    }
    std::uint32_t colorIndices = 0;
    for (std::uint32_t p = 0; p < 16; ++p)
    {
        std::uint32_t best = 0, bestError = UINT32_MAX;
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            std::uint32_t error = 0;
            for (std::uint32_t channel = 0; channel < 3; ++channel)
            {
                const int delta = static_cast<int>(pixels[p][channel]) - palette[i][channel];
                error += static_cast<std::uint32_t>(delta * delta);
            }
            if (error < bestError) { best = i; bestError = error; }
        }
        colorIndices |= best << (p * 2u);
    }
    for (std::uint32_t byte = 0; byte < 4; ++byte)
        destination[12 + byte] = static_cast<std::uint8_t>(colorIndices >> (byte * 8u));
}
} // namespace detail

inline std::size_t bc3MipByteSize(std::uint32_t width, std::uint32_t height)
{
    return static_cast<std::size_t>(std::max((width + 3u) / 4u, 1u)) *
           std::max((height + 3u) / 4u, 1u) * 16u;
}

inline void compressTextureBc3(TextureReference& texture)
{
    texture.bc3MipOffsets.clear();
    texture.bc3Pixels.clear();
    if (!texture.valid()) return;
    std::uint32_t width = texture.width, height = texture.height;
    for (std::uint32_t mip = 0; mip < texture.mipCount; ++mip)
    {
        texture.bc3MipOffsets.push_back(static_cast<std::uint32_t>(texture.bc3Pixels.size()));
        const std::uint32_t blocksWide = std::max((width + 3u) / 4u, 1u);
        const std::uint32_t blocksHigh = std::max((height + 3u) / 4u, 1u);
        const std::size_t outputOffset = texture.bc3Pixels.size();
        texture.bc3Pixels.resize(outputOffset + static_cast<std::size_t>(blocksWide) * blocksHigh * 16u);
        const std::uint8_t* source = texture.rgba8Pixels.data() + texture.mipOffsets[mip];
        for (std::uint32_t by = 0; by < blocksHigh; ++by)
            for (std::uint32_t bx = 0; bx < blocksWide; ++bx)
            {
                std::array<std::array<std::uint8_t, 4>, 16> block{};
                for (std::uint32_t y = 0; y < 4; ++y)
                    for (std::uint32_t x = 0; x < 4; ++x)
                    {
                        const std::uint32_t sx = std::min(bx * 4u + x, width - 1u);
                        const std::uint32_t sy = std::min(by * 4u + y, height - 1u);
                        const std::uint8_t* pixel = source + (static_cast<std::size_t>(sy) * width + sx) * 4u;
                        std::copy_n(pixel, 4, block[y * 4u + x].begin());
                    }
                detail::encodeBc3Block(block, texture.bc3Pixels.data() + outputOffset +
                                        (static_cast<std::size_t>(by) * blocksWide + bx) * 16u);
            }
        width = std::max(width / 2u, 1u);
        height = std::max(height / 2u, 1u);
    }
}
} // namespace vor
