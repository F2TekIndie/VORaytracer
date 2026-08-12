#pragma once

#include "Core/Renderer.h"

#include <cuda.h>
#include <optix.h>

#include <cstdint>
#include <string>

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
    void resize(std::uint32_t width, std::uint32_t height) override;
    void resetAccumulation() override;
    bool render(const Camera& camera, const RenderSettings& settings) override;
    [[nodiscard]] bool available() const override { return available_; }
    [[nodiscard]] const char* unavailableReason() const override { return unavailableReason_.c_str(); }
    [[nodiscard]] const RendererStats& stats() const override { return stats_; }
    bool setGpuInteropSurface(const GpuInteropSurface& surface);
    void clearGpuInteropSurface();
    [[nodiscard]] bool hasGpuInteropSurface() const { return interopOutputBuffer_ != 0; }
    [[nodiscard]] std::uint32_t outputWidth() const { return width_; }
    [[nodiscard]] std::uint32_t outputHeight() const { return height_; }

private:
    static void contextLog(unsigned int level, const char* tag, const char* message, void* data);
    bool resizeOutput();
    bool createDenoiser();
    bool ensureDenoiserResources();
    bool resizeDenoiser();
    bool invokeDenoiser();
    void destroyDenoiserResources();
    void destroyDenoiser();
    void destroyOutputBuffers();
    bool createPipeline();
    void destroyPipeline();
    bool buildSceneAcceleration();
    void destroySceneAcceleration();
    bool updateShaderBindingTable(const Camera& camera, const RenderSettings& settings);
    void setError(std::string message);

    const Scene* scene_{};
    CUdevice cudaDevice_{};
    CUcontext cudaContext_{};
    CUstream cudaStream_{};
    CUevent launchParameterCopyComplete_{};
    OptixDeviceContext optixContext_{};
    OptixModule module_{};
    OptixProgramGroup raygenProgramGroup_{};
    OptixProgramGroup toneMapProgramGroup_{};
    OptixProgramGroup missProgramGroup_{};
    OptixProgramGroup hitProgramGroup_{};
    OptixPipeline pipeline_{};
    CUdeviceptr outputBuffer_{};
    CUdeviceptr denoisedOutputBuffer_{};
    CUdeviceptr albedoGuideBuffer_{};
    CUdeviceptr normalGuideBuffer_{};
    CUdeviceptr interopOutputBuffer_{};
    CUexternalMemory externalMemory_{};
    CUexternalSemaphore cudaReadySemaphore_{};
    CUexternalSemaphore vulkanCompleteSemaphore_{};
    CUdeviceptr raygenRecord_{};
    CUdeviceptr toneMapRecord_{};
    CUdeviceptr missRecord_{};
    CUdeviceptr hitRecord_{};
    CUdeviceptr vertexBuffer_{};
    CUdeviceptr normalBuffer_{};
    CUdeviceptr indexBuffer_{};
    CUdeviceptr triangleMaterialIndexBuffer_{};
    CUdeviceptr materialBuffer_{};
    CUdeviceptr environmentBuffer_{};
    std::size_t vertexCount_{};
    std::size_t triangleCount_{};
    std::size_t materialCount_{};
    std::size_t environmentPixelCount_{};
    CUdeviceptr gasBuffer_{};
    OptixTraversableHandle gasHandle_{};
    OptixDenoiser denoiser_{};
    CUdeviceptr denoiserState_{};
    CUdeviceptr denoiserScratch_{};
    std::size_t denoiserStateSize_{};
    std::size_t denoiserScratchSize_{};
    void* launchParametersHost_{};
    std::uint64_t interopGeneration_{};
    bool firstInteropLaunch_{true};
    bool launchParameterCopyPending_{};
    bool interopBgra_{true};
    std::uint32_t width_{1};
    std::uint32_t height_{1};
    bool available_{};
    std::string unavailableReason_;
    RendererStats stats_{};
};
} // namespace vor
