#pragma once

#include "Assets/AssetLoader.h"
#include "Core/Renderer.h"
#include "Optix/OptixRenderer.h"
#include "Vulkan/VulkanRenderer.h"

#include <array>
#include <chrono>

struct GLFWwindow;

namespace vor
{
class Application
{
public:
    int run();

private:
    bool initialize();
    void shutdown();
    void drawUi();
    void drawMainMenu();
    void drawScenePanel();
    void drawRenderPanel();
    void drawStatsPanel();
    void loadSceneFromUi();
    bool reloadCurrentScene();
    void frameCameraToScene();
    void handleCameraNavigation();
    void resetCameraAccumulation();

    GLFWwindow* window_{};
    AssetLoader assetLoader_;
    Scene scene_;
    VulkanRenderer vulkanRenderer_;
    OptixRenderer optixRenderer_;
    RenderSettings settings_{};
    BackendKind previousBackend_{BackendKind::VulkanHybrid};
    Vec3 lightTarget_{};
    std::array<char, 1024> assetPath_{};
    std::string statusMessage_{"Ready"};
    bool meshoptimizerEnabled_{true};
    bool defaultPlasticEnabled_{false};
    bool groundPlaneEnabled_{true};
    std::uint32_t interopWidth_{};
    std::uint32_t interopHeight_{};
};
} // namespace vor
