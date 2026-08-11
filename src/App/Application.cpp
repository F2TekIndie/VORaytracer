#include "App/Application.h"

#include "Core/Log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>

namespace vor
{
int Application::run()
{
    if (!initialize())
    {
        shutdown();
        return 1;
    }

    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        if (width <= 0 || height <= 0)
        {
            glfwWaitEvents();
            continue;
        }

        vulkanRenderer_.resize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
        optixRenderer_.resize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
        vulkanRenderer_.beginUiFrame();
        drawUi();

        if (settings_.backend != previousBackend_)
        {
            vulkanRenderer_.resetAccumulation();
            optixRenderer_.resetAccumulation();
            previousBackend_ = settings_.backend;
        }

        if (settings_.backend == BackendKind::Optix && optixRenderer_.available())
        {
            if (optixRenderer_.render(scene_.camera, settings_))
                vulkanRenderer_.setExternalImage(optixRenderer_.displayPixels(), optixRenderer_.outputWidth(),
                                                 optixRenderer_.outputHeight());
        }
        else
            vulkanRenderer_.clearExternalImage();
        if (!vulkanRenderer_.render(scene_.camera, settings_))
            break;
    }

    shutdown();
    return 0;
}

bool Application::initialize()
{
    if (!glfwInit())
    {
        log(LogLevel::Error, "glfwInit failed");
        return false;
    }
    if (!glfwVulkanSupported())
    {
        log(LogLevel::Error, "GLFW reports no Vulkan loader");
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(1600, 900, "VORaytracer - Vulkan Mesh Shaders + OptiX", nullptr, nullptr);
    if (!window_)
    {
        log(LogLevel::Error, "glfwCreateWindow failed");
        return false;
    }

    if (!vulkanRenderer_.initialize(window_))
    {
        log(LogLevel::Error, vulkanRenderer_.unavailableReason());
        return false;
    }
    if (!optixRenderer_.initialize(window_))
        statusMessage_ = std::string("OptiX unavailable: ") + optixRenderer_.unavailableReason();
    else if (const char* requestedBackend = std::getenv("VOR_BACKEND");
             requestedBackend && std::strcmp(requestedBackend, "optix") == 0)
    {
        settings_.backend = BackendKind::Optix;
        previousBackend_ = BackendKind::Optix;
    }

    scene_ = assetLoader_.createProceduralCube();
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    return true;
}

void Application::shutdown()
{
    optixRenderer_.shutdown();
    vulkanRenderer_.shutdown();
    if (window_)
        glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
}

void Application::drawUi()
{
    drawMainMenu();
    drawScenePanel();
    drawRenderPanel();
    drawStatsPanel();

    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("Status", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings))
        ImGui::TextUnformatted(statusMessage_.c_str());
    ImGui::End();
}

void Application::drawMainMenu()
{
    if (!ImGui::BeginMainMenuBar())
        return;
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Load path"))
            loadSceneFromUi();
        if (ImGui::MenuItem("Procedural cube"))
        {
            scene_ = assetLoader_.createProceduralCube();
            vulkanRenderer_.setScene(&scene_);
            optixRenderer_.setScene(&scene_);
            statusMessage_ = "Loaded procedural cube";
        }
        if (ImGui::MenuItem("Exit"))
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        ImGui::EndMenu();
    }
    ImGui::TextDisabled("  %s", scene_.name.c_str());
    ImGui::EndMainMenuBar();
}

void Application::drawScenePanel()
{
    ImGui::Begin("Scene");
    ImGui::InputText("Asset path", assetPath_.data(), assetPath_.size());
    ImGui::SameLine();
    if (ImGui::Button("Load"))
        loadSceneFromUi();
    ImGui::Separator();
    ImGui::Text("Meshes: %zu", scene_.meshes.size());
    ImGui::Text("Instances: %zu", scene_.instances.size());
    ImGui::Text("Materials: %zu", scene_.materials.size());
    ImGui::Text("Triangles: %zu", scene_.triangleCount());
    ImGui::Text("Meshlets: %zu", scene_.meshletCount());
    if (ImGui::TreeNode("Instances"))
    {
        for (const Instance& instance : scene_.instances)
            ImGui::BulletText("%s (mesh %u)", instance.name.c_str(), instance.meshIndex);
        ImGui::TreePop();
    }
    ImGui::End();
}

void Application::drawRenderPanel()
{
    ImGui::Begin("Renderer");
    int backend = static_cast<int>(settings_.backend);
    if (ImGui::RadioButton("Vulkan Mesh + Ray Query", backend == 0))
        settings_.backend = BackendKind::VulkanHybrid;
    if (!optixRenderer_.available())
        ImGui::BeginDisabled();
    if (ImGui::RadioButton("NVIDIA OptiX", backend == 1))
        settings_.backend = BackendKind::Optix;
    if (!optixRenderer_.available())
    {
        ImGui::EndDisabled();
        ImGui::TextWrapped("OptiX: %s", optixRenderer_.unavailableReason());
    }

    ImGui::SeparatorText("PBR / Ray tracing");
    ImGui::SliderInt("Samples/frame", reinterpret_cast<int*>(&settings_.samplesPerFrame), 1, 16);
    ImGui::SliderInt("Max bounces", reinterpret_cast<int*>(&settings_.maxBounces), 1, 16);
    ImGui::SliderFloat("Exposure", &settings_.exposure, -8.0f, 8.0f);
    ImGui::Checkbox("Ray-traced shadows", &settings_.rayTracedShadows);
    ImGui::Checkbox("Ray-traced reflections", &settings_.rayTracedReflections);
    ImGui::Checkbox("Indirect lighting", &settings_.indirectLighting);
    ImGui::Checkbox("Denoiser", &settings_.denoiser);
    ImGui::Checkbox("Meshlet debug colors", &settings_.showMeshlets);

    ImGui::SeparatorText("Camera");
    ImGui::DragFloat3("Position", &scene_.camera.position.x, 0.02f);
    ImGui::DragFloat3("Target", &scene_.camera.target.x, 0.02f);
    ImGui::SliderFloat("Vertical FOV", &scene_.camera.verticalFovDegrees, 20.0f, 100.0f);
    ImGui::End();
}

void Application::drawStatsPanel()
{
    const RendererStats& stats = settings_.backend == BackendKind::Optix && optixRenderer_.available()
                                     ? optixRenderer_.stats()
                                     : vulkanRenderer_.stats();
    ImGui::Begin("Performance");
    ImGui::Text("Device: %s", stats.deviceName.c_str());
    ImGui::Text("CPU frame: %.3f ms", stats.frameMilliseconds);
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(stats.frameIndex));
    ImGui::Text("Accumulated samples: %llu", static_cast<unsigned long long>(stats.accumulatedSamples));
    ImGui::Text("Meshlets: %u / %u", stats.visibleMeshlets, stats.totalMeshlets);
    ImGui::Text("Rays: %llu", static_cast<unsigned long long>(stats.tracedRays));
    ImGui::Text("Ray Query feature: %s", vulkanRenderer_.rayQueryAvailable() ? "yes" : "no");
    ImGui::Text("Task Shader feature: %s", vulkanRenderer_.taskShaderAvailable() ? "yes" : "no");
    ImGui::End();
}

void Application::loadSceneFromUi()
{
    const std::filesystem::path path(assetPath_.data());
    if (path.empty())
    {
        statusMessage_ = "Enter an asset path first";
        return;
    }
    AssetLoadResult result = assetLoader_.load(path);
    if (!result)
    {
        statusMessage_ = "Import failed: " + result.error;
        return;
    }
    scene_ = std::move(result.scene);
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    statusMessage_ = "Loaded " + path.string();
}
} // namespace vor
