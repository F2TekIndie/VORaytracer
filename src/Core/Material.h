#pragma once

#include "Core/Math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vor
{
inline constexpr std::uint32_t kInvalidTextureId = std::numeric_limits<std::uint32_t>::max();

enum class MaterialFlags : std::uint32_t
{
    None = 0,
    DoubleSided = 1u << 0,
    AlphaMask = 1u << 1,
    Transmission = 1u << 2,
    Clearcoat = 1u << 3,
    Sheen = 1u << 4,
    Anisotropy = 1u << 5,
    Emissive = 1u << 6,
    Subsurface = 1u << 7,
    Volume = 1u << 8,
};

constexpr MaterialFlags operator|(MaterialFlags left, MaterialFlags right)
{
    return static_cast<MaterialFlags>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
}

constexpr MaterialFlags& operator|=(MaterialFlags& left, MaterialFlags right)
{
    left = left | right;
    return left;
}

constexpr bool hasFlag(MaterialFlags value, MaterialFlags flag)
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flag)) != 0;
}

// This layout is mirrored verbatim in shaders/Shared/MaterialTypes.slang. Keep every
// group 16-byte aligned so the same buffer can be consumed by Vulkan and CUDA/OptiX.
struct alignas(16) GpuMaterial
{
    Vec4 baseColorFactor{};
    Vec4 emissiveAndMetallic{};
    Vec4 surfaceParameters{};             // roughness, normal scale, AO strength, alpha cutoff
    Vec4 transmissionClearcoat{};         // transmission, IOR, clearcoat, clearcoat roughness
    Vec4 anisotropySheen{};               // anisotropy, rotation, sheen roughness, bump scale
    Vec4 sheenColorAbsorptionDistance{};
    Vec4 absorptionColorSubsurface{};     // absorption RGB, subsurface weight
    Vec4 subsurfaceColorRadius{};
    Vec4 volumeAbsorptionDensity{};
    Vec4 volumeScatteringAnisotropy{};
    std::array<std::uint32_t, 4> materialFlags{}; // flags, alpha mode, height texture, packed height dimensions
    std::array<std::uint32_t, 4> textureIndices0{}; // base color, metallic/roughness, normal, emissive
    std::array<std::uint32_t, 4> textureIndices1{}; // AO, transmission, clearcoat, clearcoat roughness
    std::array<std::uint32_t, 4> textureIndices2{}; // clearcoat normal, sheen color, sheen roughness, anisotropy
    std::array<std::uint32_t, 4> textureIndices3{}; // opacity, reserved, reserved, reserved
};

static_assert(std::is_standard_layout_v<GpuMaterial>);
static_assert(alignof(GpuMaterial) == 16);
static_assert(sizeof(GpuMaterial) == 240);
static_assert(offsetof(GpuMaterial, materialFlags) == 160);
static_assert(offsetof(GpuMaterial, textureIndices0) == 176);
static_assert(offsetof(GpuMaterial, textureIndices1) == 192);
static_assert(offsetof(GpuMaterial, textureIndices2) == 208);
static_assert(offsetof(GpuMaterial, textureIndices3) == 224);
} // namespace vor
