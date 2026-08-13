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

enum class DebugView : std::uint32_t
{
    Beauty,
    BaseColor,
    Metallic,
    Roughness,
    ShadingNormal,
    GeometricNormal,
    Tangent,
    Bitangent,
    Occlusion,
    Emissive,
    DiffuseLobe,
    SpecularLobe,
    ClearcoatLobe,
    SheenLobe,
    TransmissionLobe,
    BsdfPdf,
    MaterialId,
    TextureIds,
    Medium,
    PathDepth,
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
    DebugView debugView{DebugView::Beauty};
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
    std::uint64_t gpuSceneBytes{};
    std::uint64_t textureBytes{};
    std::uint64_t materialBytes{};
    std::uint32_t residentTextures{};
    std::uint32_t residentMaterials{};
    std::uint32_t descriptorCapacity{};
};

struct GpuInteropSurface
{
    void* memoryHandle{};
    void* cudaReadySemaphoreHandle{};
    void* vulkanCompleteSemaphoreHandle{};
    std::uint64_t allocationSize{};
    std::uint64_t pixelByteSize{};
    std::uint64_t bufferByteSize{};
    std::uint64_t inputOffset{};
    std::uint64_t denoisedOffset{};
    std::uint64_t displayOffset{};
    std::uint64_t generation{};
    std::uint32_t width{};
    std::uint32_t height{};
    bool bgra{};

    [[nodiscard]] explicit operator bool() const
    {
        return memoryHandle && cudaReadySemaphoreHandle && vulkanCompleteSemaphoreHandle && pixelByteSize > 0 &&
               bufferByteSize >= displayOffset + pixelByteSize;
    }
};

class IRenderBackend
{
public:
    virtual ~IRenderBackend() = default;
    virtual bool initialize(GLFWwindow* window) = 0;
    virtual void shutdown() = 0;
    virtual void setScene(const Scene* scene) = 0;
    virtual bool updateMaterial(std::uint32_t materialIndex) = 0;
    virtual void resize(std::uint32_t width, std::uint32_t height) = 0;
    virtual void resetAccumulation() = 0;
    virtual bool render(const Camera& camera, const RenderSettings& settings) = 0;
    [[nodiscard]] virtual bool available() const = 0;
    [[nodiscard]] virtual const char* unavailableReason() const = 0;
    [[nodiscard]] virtual const RendererStats& stats() const = 0;
};
} // namespace vor
