#pragma once

#include "Core/Renderer.h"

#include <cuda.h>
#include <optix.h>

#include <cstdint>
#include <span>
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
    void resize(std::uint32_t width, std::uint32_t height) override;
    void resetAccumulation() override;
    bool render(const Camera& camera, const RenderSettings& settings) override;
    [[nodiscard]] bool available() const override { return available_; }
    [[nodiscard]] const char* unavailableReason() const override { return unavailableReason_.c_str(); }
    [[nodiscard]] const RendererStats& stats() const override { return stats_; }
    [[nodiscard]] std::span<const std::uint32_t> displayPixels() const { return displayPixels_; }
    [[nodiscard]] std::uint32_t outputWidth() const { return width_; }
    [[nodiscard]] std::uint32_t outputHeight() const { return height_; }

private:
    static void contextLog(unsigned int level, const char* tag, const char* message, void* data);
    bool resizeOutput();
    bool createPipeline();
    void destroyPipeline();
    bool buildSceneAcceleration();
    void destroySceneAcceleration();
    bool updateShaderBindingTable(const Camera& camera, const RenderSettings& settings);
    void setError(std::string message);

    const Scene* scene_{};
    CUdevice cudaDevice_{};
    CUcontext cudaContext_{};
    OptixDeviceContext optixContext_{};
    OptixModule module_{};
    OptixProgramGroup raygenProgramGroup_{};
    OptixProgramGroup missProgramGroup_{};
    OptixProgramGroup hitProgramGroup_{};
    OptixPipeline pipeline_{};
    CUdeviceptr outputBuffer_{};
    CUdeviceptr raygenRecord_{};
    CUdeviceptr missRecord_{};
    CUdeviceptr hitRecord_{};
    CUdeviceptr vertexBuffer_{};
    CUdeviceptr indexBuffer_{};
    CUdeviceptr gasBuffer_{};
    OptixTraversableHandle gasHandle_{};
    std::vector<float> hostOutput_;
    std::vector<std::uint32_t> displayPixels_;
    std::uint32_t width_{1};
    std::uint32_t height_{1};
    bool available_{};
    std::string unavailableReason_;
    RendererStats stats_{};
};
} // namespace vor
