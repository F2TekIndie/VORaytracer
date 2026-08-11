#include "App/Application.h"

#include "Core/Log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>

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
        handleCameraNavigation();
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

    const AssetLoadOptions loadOptions{
        .enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
        .overrideWithDefaultPlastic = defaultPlasticEnabled_,
        .addGroundPlane = groundPlaneEnabled_,
    };
    scene_ = assetLoader_.createProceduralCube(loadOptions);
    frameCameraToScene();
    if (const char* startupScene = std::getenv("VOR_SCENE"); startupScene && *startupScene)
    {
        AssetLoadResult result = assetLoader_.load(std::filesystem::path(startupScene), loadOptions);
        if (result)
        {
            scene_ = std::move(result.scene);
            frameCameraToScene();
            statusMessage_ = std::string("Loaded and framed ") + startupScene;
        }
        else
            statusMessage_ = "Startup import failed: " + result.error;
    }
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
            scene_ = assetLoader_.createProceduralCube(
                {.enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
                 .overrideWithDefaultPlastic = defaultPlasticEnabled_,
                 .addGroundPlane = groundPlaneEnabled_});
            frameCameraToScene();
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
    const bool previousMeshoptimizerState = meshoptimizerEnabled_;
    const ImVec4 toggleColor = meshoptimizerEnabled_ ? ImVec4(0.16f, 0.52f, 0.24f, 1.0f)
                                                      : ImVec4(0.42f, 0.18f, 0.18f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, toggleColor);
    if (ImGui::Button(meshoptimizerEnabled_ ? "meshoptimizer: ON" : "meshoptimizer: OFF", ImVec2(210.0f, 0.0f)))
    {
        meshoptimizerEnabled_ = !meshoptimizerEnabled_;
        if (!reloadCurrentScene())
            meshoptimizerEnabled_ = previousMeshoptimizerState;
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Always active: meshlet partitioning required by the mesh shader.\n"
                          "Toggle controls remap, cache, overdraw, vertex fetch, bounds and LOD simplification.\n"
                          "Changing it reloads the current model.");
    }
    const bool previousDefaultPlasticState = defaultPlasticEnabled_;
    const ImVec4 plasticToggleColor = defaultPlasticEnabled_ ? ImVec4(0.16f, 0.52f, 0.24f, 1.0f)
                                                              : ImVec4(0.42f, 0.18f, 0.18f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, plasticToggleColor);
    if (ImGui::Button(defaultPlasticEnabled_ ? "Default plastic: ON" : "Default plastic: OFF",
                      ImVec2(210.0f, 0.0f)))
    {
        defaultPlasticEnabled_ = !defaultPlasticEnabled_;
        if (!reloadCurrentScene())
            defaultPlasticEnabled_ = previousDefaultPlasticState;
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Overrides every loaded mesh with one light-gray plastic material.\n"
                          "Albedo: 0.75 | Metallic: 0.0 | Roughness: 0.5\n"
                          "Changing it reloads the current model.");
    }
    const bool previousGroundPlaneState = groundPlaneEnabled_;
    const ImVec4 groundToggleColor = groundPlaneEnabled_ ? ImVec4(0.16f, 0.52f, 0.24f, 1.0f)
                                                          : ImVec4(0.42f, 0.18f, 0.18f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, groundToggleColor);
    if (ImGui::Button(groundPlaneEnabled_ ? "Ground plane: ON" : "Ground plane: OFF", ImVec2(210.0f, 0.0f)))
    {
        groundPlaneEnabled_ = !groundPlaneEnabled_;
        if (!reloadCurrentScene())
            groundPlaneEnabled_ = previousGroundPlaneState;
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Adds a model-sized ground plane below the scene.\n"
                          "Material: Default Plastic (Albedo 0.75, Metallic 0.0, Roughness 0.5).\n"
                          "Changing it reloads the current model.");
    }
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

    ImGui::SeparatorText("Key light");
    if (!scene_.lights.empty())
    {
        Light& light = scene_.lights.front();
        bool lightChanged = ImGui::DragFloat3("Light position", &light.position.x, 0.02f);
        lightChanged |= ImGui::DragFloat3("Light target", &lightTarget_.x, 0.02f);
        lightChanged |= ImGui::ColorEdit3("Light color", &light.color.x);
        lightChanged |= ImGui::DragFloat("Light intensity", &light.intensity, 0.05f, 0.0f, 100.0f, "%.2f");
        if (lightChanged)
        {
            light.intensity = std::max(light.intensity, 0.0f);
            const Vec3 direction = lightTarget_ - light.position;
            if (length(direction) > 1e-6f)
                light.direction = normalize(direction);
            resetCameraAccumulation();
        }
    }
    ImGui::TextDisabled("Shift+LMB: orbit light | Shift+MMB: pan | Shift+wheel: zoom");

    ImGui::SeparatorText("Camera");
    bool cameraChanged = ImGui::DragFloat3("Position", &scene_.camera.position.x, 0.02f);
    cameraChanged |= ImGui::DragFloat3("Target", &scene_.camera.target.x, 0.02f);
    cameraChanged |= ImGui::SliderFloat("Vertical FOV", &scene_.camera.verticalFovDegrees, 20.0f, 100.0f);
    if (ImGui::Button("Frame model"))
    {
        frameCameraToScene();
        cameraChanged = true;
    }
    ImGui::TextDisabled("LMB drag: orbit camera | MMB drag: pan | wheel: zoom");
    if (cameraChanged)
        resetCameraAccumulation();
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
    AssetLoadResult result = assetLoader_.load(
        path, {.enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
               .overrideWithDefaultPlastic = defaultPlasticEnabled_,
               .addGroundPlane = groundPlaneEnabled_});
    if (!result)
    {
        statusMessage_ = "Import failed: " + result.error;
        return;
    }
    scene_ = std::move(result.scene);
    frameCameraToScene();
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    statusMessage_ = "Loaded and framed " + path.string();
}

bool Application::reloadCurrentScene()
{
    Scene reloadedScene;
    const AssetLoadOptions options{
        .enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
        .overrideWithDefaultPlastic = defaultPlasticEnabled_,
        .addGroundPlane = groundPlaneEnabled_,
    };
    if (scene_.sourcePath.empty())
        reloadedScene = assetLoader_.createProceduralCube(options);
    else
    {
        AssetLoadResult result = assetLoader_.load(scene_.sourcePath, options);
        if (!result)
        {
            statusMessage_ = "Reload failed: " + result.error;
            return false;
        }
        reloadedScene = std::move(result.scene);
    }

    scene_ = std::move(reloadedScene);
    frameCameraToScene();
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    statusMessage_ = std::string("Reloaded: meshoptimizer ") + (meshoptimizerEnabled_ ? "ON" : "OFF") +
                     ", default plastic " + (defaultPlasticEnabled_ ? "ON" : "OFF") +
                     ", ground plane " + (groundPlaneEnabled_ ? "ON" : "OFF");
    return true;
}

void Application::frameCameraToScene()
{
    Vec3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                 std::numeric_limits<float>::lowest()};
    bool hasPoint = false;

    const auto includeMesh = [&](const Mesh& mesh, const Mat4& transform) {
        if (mesh.isGroundPlane)
            return;
        for (const Vertex& vertex : mesh.vertices)
        {
            const Vec3& p = vertex.position;
            const Vec3 world{
                transform.m[0] * p.x + transform.m[4] * p.y + transform.m[8] * p.z + transform.m[12],
                transform.m[1] * p.x + transform.m[5] * p.y + transform.m[9] * p.z + transform.m[13],
                transform.m[2] * p.x + transform.m[6] * p.y + transform.m[10] * p.z + transform.m[14],
            };
            minimum.x = std::min(minimum.x, world.x);
            minimum.y = std::min(minimum.y, world.y);
            minimum.z = std::min(minimum.z, world.z);
            maximum.x = std::max(maximum.x, world.x);
            maximum.y = std::max(maximum.y, world.y);
            maximum.z = std::max(maximum.z, world.z);
            hasPoint = true;
        }
    };

    if (!scene_.instances.empty())
    {
        for (const Instance& instance : scene_.instances)
        {
            if (instance.meshIndex < scene_.meshes.size())
                includeMesh(scene_.meshes[instance.meshIndex], instance.transform);
        }
    }
    else
    {
        const Mat4 identity = Mat4::identity();
        for (const Mesh& mesh : scene_.meshes)
            includeMesh(mesh, identity);
    }
    if (!hasPoint)
        return;

    const Vec3 center = (minimum + maximum) * 0.5f;
    float radius = length(maximum - minimum) * 0.5f;
    radius = std::max(radius, 0.001f);

    int framebufferWidth = 1;
    int framebufferHeight = 1;
    glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    const float aspect = static_cast<float>(std::max(framebufferWidth, 1)) /
                         static_cast<float>(std::max(framebufferHeight, 1));
    const float verticalFov = scene_.camera.verticalFovDegrees * kPi / 180.0f;
    const float horizontalFov = 2.0f * std::atan(std::tan(verticalFov * 0.5f) * aspect);
    const float fittingFov = std::min(verticalFov, horizontalFov);
    const float distance = radius / std::max(std::sin(fittingFov * 0.5f), 0.01f) * 1.15f;

    Vec3 viewDirection = normalize(scene_.camera.position - scene_.camera.target);
    if (length(viewDirection) < 0.5f)
        viewDirection = normalize(Vec3{0.65f, 0.35f, 1.0f});
    scene_.camera.target = center;
    scene_.camera.position = center + viewDirection * distance;
    scene_.camera.nearPlane = std::max(radius * 0.001f, 0.0001f);
    scene_.camera.farPlane = std::max(distance + radius * 4.0f, radius * 20.0f);

    if (scene_.lights.empty())
        scene_.lights.push_back(Light{.name = "Key Light"});
    Light& light = scene_.lights.front();
    lightTarget_ = center;
    Vec3 lightDirection = normalize(light.direction);
    if (length(lightDirection) < 0.5f)
        lightDirection = normalize(Vec3{0.45f, -0.85f, -0.3f});
    const float lightDistance = radius * 2.5f;
    light.position = lightTarget_ - lightDirection * lightDistance;
    light.direction = normalize(lightTarget_ - light.position);
    light.range = radius * 10.0f;
    resetCameraAccumulation();
}

void Application::handleCameraNavigation()
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    Camera& camera = scene_.camera;
    const bool moveLight = io.KeyShift && !scene_.lights.empty();
    Light* light = moveLight ? &scene_.lights.front() : nullptr;
    Vec3* position = moveLight ? &light->position : &camera.position;
    Vec3* target = moveLight ? &lightTarget_ : &camera.target;
    Vec3 offset = *position - *target;
    float distance = length(offset);
    if (distance <= 1e-6f)
    {
        offset = {0.0f, 0.0f, 1.0f};
        distance = 1.0f;
    }
    bool changed = false;

    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
        (std::abs(io.MouseDelta.x) > 0.0f || std::abs(io.MouseDelta.y) > 0.0f))
    {
        float yaw = std::atan2(offset.x, offset.z);
        float pitch = std::asin(std::clamp(offset.y / distance, -1.0f, 1.0f));
        yaw -= io.MouseDelta.x * 0.005f;
        pitch = std::clamp(pitch + io.MouseDelta.y * 0.005f, -1.5533f, 1.5533f);
        const float horizontal = std::cos(pitch) * distance;
        offset = {std::sin(yaw) * horizontal, std::sin(pitch) * distance, std::cos(yaw) * horizontal};
        *position = *target + offset;
        changed = true;
    }

    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) &&
        (std::abs(io.MouseDelta.x) > 0.0f || std::abs(io.MouseDelta.y) > 0.0f))
    {
        const Vec3 forward = normalize(*target - *position);
        const Vec3 right = normalize(cross(forward, camera.up));
        const Vec3 correctedUp = normalize(cross(right, forward));
        int width = 1;
        int height = 1;
        glfwGetFramebufferSize(window_, &width, &height);
        const float worldPerPixel = 2.0f * distance *
                                    std::tan(camera.verticalFovDegrees * kPi / 360.0f) /
                                    static_cast<float>(std::max(height, 1));
        const Vec3 translation = right * (-io.MouseDelta.x * worldPerPixel) +
                                 correctedUp * (io.MouseDelta.y * worldPerPixel);
        *position = *position + translation;
        *target = *target + translation;
        changed = true;
    }

    if (std::abs(io.MouseWheel) > 0.0f)
    {
        const float minimumDistance = std::max(camera.nearPlane * 2.0f, 0.0002f);
        const float maximumDistance = std::max(camera.farPlane * 0.9f, minimumDistance * 2.0f);
        distance = std::clamp(distance * std::exp(-io.MouseWheel * 0.16f), minimumDistance, maximumDistance);
        Vec3 zoomDirection = normalize(*position - *target);
        if (length(zoomDirection) < 0.5f)
            zoomDirection = normalize(offset);
        *position = *target + zoomDirection * distance;
        changed = true;
    }

    if (changed)
    {
        if (light)
        {
            light->direction = normalize(lightTarget_ - light->position);
            light->range = std::max(length(lightTarget_ - light->position) * 4.0f, 1.0f);
        }
        resetCameraAccumulation();
    }
}

void Application::resetCameraAccumulation()
{
    vulkanRenderer_.resetAccumulation();
    optixRenderer_.resetAccumulation();
}
} // namespace vor
