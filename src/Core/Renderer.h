#pragma once

#include "Core/Scene.h"

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace vor
{
enum class BackendKind : std::uint32_t
{
    VulkanHybrid,
    Optix,
};

struct RenderSettings
{
    BackendKind backend{BackendKind::VulkanHybrid};
    std::uint32_t samplesPerFrame{1};
    std::uint32_t maxBounces{4};
    float exposure{};
    float renderScale{1.0f};
    bool rayTracedShadows{true};
    bool rayTracedReflections{true};
    bool indirectLighting{true};
    bool denoiser{false};
    bool vsync{true};
    bool showMeshlets{false};
};

struct RendererStats
{
    std::string deviceName;
    float frameMilliseconds{};
    float gpuMilliseconds{};
    std::uint64_t frameIndex{};
    std::uint64_t accumulatedSamples{};
    std::uint32_t visibleMeshlets{};
    std::uint32_t totalMeshlets{};
    std::uint64_t tracedRays{};
};

class IRenderBackend
{
public:
    virtual ~IRenderBackend() = default;
    virtual bool initialize(GLFWwindow* window) = 0;
    virtual void shutdown() = 0;
    virtual void setScene(const Scene* scene) = 0;
    virtual void resize(std::uint32_t width, std::uint32_t height) = 0;
    virtual void resetAccumulation() = 0;
    virtual bool render(const Camera& camera, const RenderSettings& settings) = 0;
    [[nodiscard]] virtual bool available() const = 0;
    [[nodiscard]] virtual const char* unavailableReason() const = 0;
    [[nodiscard]] virtual const RendererStats& stats() const = 0;
};
} // namespace vor

