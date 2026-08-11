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

    GLFWwindow* window_{};
    AssetLoader assetLoader_;
    Scene scene_;
    VulkanRenderer vulkanRenderer_;
    OptixRenderer optixRenderer_;
    RenderSettings settings_{};
    BackendKind previousBackend_{BackendKind::VulkanHybrid};
    std::array<char, 1024> assetPath_{};
    std::string statusMessage_{"Ready"};
};
} // namespace vor

