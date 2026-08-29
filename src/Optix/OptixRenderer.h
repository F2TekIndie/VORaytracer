#pragma once

#include "Core/Renderer.h"

#include <cuda.h>
#include <optix.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vor
{
class OptixRenderer final : public IRenderBackend
{
public:
    OptixRenderer() = default;
    ~OptixRenderer() override;

    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;

    bool initialize(GLFWwindow* window) override;
    void shutdown() override;
    void setScene(const Scene* scene) override;
    bool updateMaterial(std::uint32_t materialIndex) override;
    void resize(std::uint32_t width, std::uint32_t height) override;
    void resetAccumulation() override;
    bool render(const Camera& camera, const RenderSettings& settings) override;
    [[nodiscard]] bool available() const override { return available_; }
    [[nodiscard]] const char* unavailableReason() const override { return unavailableReason_.c_str(); }
    [[nodiscard]] const RendererStats& stats() const override { return stats_; }
    bool setGpuInteropSurface(const GpuInteropSurface& surface);
    bool denoiseVulkanFrame(float exposure, bool temporalRendering);
    void clearGpuInteropSurface();
    void invalidateTemporalHistory() { denoiserTemporalHistoryValid_ = false; }
    [[nodiscard]] bool hasGpuInteropSurface() const { return interopOutputBuffer_ != 0; }
    [[nodiscard]] std::uint32_t outputWidth() const { return width_; }
    [[nodiscard]] std::uint32_t outputHeight() const { return height_; }
    void setTextureStreamingOptions(bool enabled, std::uint32_t budgetMiB)
    {
        textureStreamingEnabled_ = enabled;
        textureBudgetMiB_ = budgetMiB > 0 ? budgetMiB : 1u;
    }

private:
    static constexpr std::uint32_t kLaunchSlotCount = 3;

    static void contextLog(unsigned int level, const char* tag, const char* message, void* data);
    bool resizeOutput();
    bool createDenoiser();
    bool ensureDenoiserResources();
    bool resizeDenoiser();
    bool invokeDenoiser(bool temporalRendering);
    void destroyDenoiserResources();
    void destroyDenoiser();
    void destroyOutputBuffers();
    bool createPipeline();
    void destroyPipeline();
    bool buildSceneAcceleration();
    void destroySceneAcceleration();
    bool uploadTextureResources();
    void destroyTextureResources();
    bool updateShaderBindingTable(const Camera& camera, const RenderSettings& settings,
                                  std::uint32_t launchSlot);
    bool waitForLaunchSlot(std::uint32_t launchSlot);
    void setError(std::string message);

    const Scene* scene_{};
    CUdevice cudaDevice_{};
    CUcontext cudaContext_{};
    CUstream cudaStream_{};
    std::array<CUevent, kLaunchSlotCount> launchSlotComplete_{};
    CUevent launchTimingStart_{};
    CUevent launchTimingEnd_{};
    OptixDeviceContext optixContext_{};
    OptixModule module_{};
    OptixProgramGroup raygenProgramGroup_{};
    OptixProgramGroup toneMapProgramGroup_{};
    OptixProgramGroup missProgramGroup_{};
    OptixProgramGroup hitProgramGroup_{};
    OptixPipeline pipeline_{};
    CUdeviceptr outputBuffer_{};
    CUdeviceptr sampleCountBuffer_{};
    CUdeviceptr luminanceMomentsBuffer_{};
    CUdeviceptr interopBaseBuffer_{};
    CUdeviceptr interopInputBuffer_{};
    CUdeviceptr interopDenoisedBuffer_{};
    CUdeviceptr interopOutputBuffer_{};
    CUdeviceptr interopFlowBuffer_{};
    CUdeviceptr interopAlbedoBuffer_{};
    CUdeviceptr interopNormalDepthBuffer_{};
    CUexternalMemory externalMemory_{};
    CUexternalSemaphore cudaReadySemaphore_{};
    CUexternalSemaphore vulkanCompleteSemaphore_{};
    std::array<CUdeviceptr, kLaunchSlotCount> raygenRecords_{};
    std::array<CUdeviceptr, kLaunchSlotCount> toneMapRecords_{};
    CUdeviceptr missRecord_{};
    CUdeviceptr hitRecord_{};
    CUdeviceptr vertexBuffer_{};
    CUdeviceptr normalBuffer_{};
    CUdeviceptr tangentBuffer_{};
    CUdeviceptr uvBuffer_{};
    CUdeviceptr indexBuffer_{};
    CUdeviceptr instanceBuffer_{};
    CUdeviceptr materialBuffer_{};
    CUdeviceptr textureMetadataBuffer_{};
    CUdeviceptr lightBuffer_{};
    CUdeviceptr lightAliasBuffer_{};
    CUdeviceptr emissiveTriangleBuffer_{};
    CUdeviceptr environmentBuffer_{};
    CUdeviceptr environmentConditionalCdfBuffer_{};
    CUdeviceptr environmentMarginalCdfBuffer_{};
    CUdeviceptr textureTableBuffer_{};
    std::vector<CUmipmappedArray> textureArrays_;
    std::vector<CUtexObject> textureObjects_;
    void* textureUploadHost_{};
    bool textureStreamingEnabled_{true};
    std::uint32_t textureBudgetMiB_{512};
    std::size_t materialTextureBytes_{};
    std::size_t vertexCount_{};
    std::size_t triangleCount_{};
    std::size_t materialCount_{};
    std::size_t textureMetadataCount_{};
    std::size_t lightCount_{};
    float analyticLightPower_{};
    std::size_t instanceCount_{};
    std::uint32_t motionTransformCount_{};
    std::size_t emissiveTriangleCount_{};
    float emissiveLightPower_{};
    std::size_t environmentPixelCount_{};
    std::size_t environmentConditionalCdfCount_{};
    std::size_t environmentMarginalCdfCount_{};
    CUdeviceptr gasBuffer_{};
    CUdeviceptr iasInstanceBuffer_{};
    CUdeviceptr motionTransformBuffer_{};
    std::vector<CUdeviceptr> meshGasBuffers_;
    OptixTraversableHandle gasHandle_{};
    OptixDenoiser denoiser_{};
    CUdeviceptr denoiserState_{};
    CUdeviceptr denoiserScratch_{};
    CUdeviceptr denoiserPreviousOutput_{};
    CUdeviceptr denoiserPreviousInternalGuide_{};
    CUdeviceptr denoiserOutputInternalGuide_{};
    std::size_t denoiserStateSize_{};
    std::size_t denoiserScratchSize_{};
    std::size_t denoiserInternalGuidePixelSize_{};
    bool denoiserTemporalHistoryValid_{};
    void* launchParametersHost_{};
    std::uint64_t interopGeneration_{};
    bool firstInteropLaunch_{true};
    std::array<bool, kLaunchSlotCount> launchSlotPending_{};
    bool launchTimingPending_{};
    bool resetSamplingBuffersPending_{true};
    bool interopBgra_{true};
    std::uint32_t width_{1};
    std::uint32_t height_{1};
    std::uint64_t launchSlotCursor_{};
    bool available_{};
    std::string unavailableReason_;
    RendererStats stats_{};
    Camera previousCamera_{};
    bool previousCameraValid_{};
};
} // namespace vor
