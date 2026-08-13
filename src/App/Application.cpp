#include "App/Application.h"

#include "Core/Log.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <shobjidl.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace vor
{
namespace
{
using Microsoft::WRL::ComPtr;

struct FileDialogResult
{
    std::optional<std::filesystem::path> path;
    std::string error;
};

std::string hresultMessage(const char* operation, HRESULT result)
{
    std::ostringstream message;
    message << operation << " failed (HRESULT 0x" << std::hex << std::uppercase
            << static_cast<unsigned long>(result) << ')';
    return message.str();
}

std::string pathToUtf8(const std::filesystem::path& path)
{
    const std::wstring& wide = path.native();
    if (wide.empty())
        return {};
    const int byteCount = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                                               nullptr, 0, nullptr, nullptr);
    if (byteCount <= 0)
        return {};
    std::string result(static_cast<std::size_t>(byteCount), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                        result.data(), byteCount, nullptr, nullptr);
    return result;
}

std::filesystem::path pathFromUtf8(const char* path)
{
    if (!path || !*path)
        return {};
    const int characterCount = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
    if (characterCount <= 1)
        return {};
    std::wstring wide(static_cast<std::size_t>(characterCount), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide.data(), characterCount);
    wide.pop_back();
    return std::filesystem::path(wide);
}

std::filesystem::path dialogInitialFolder(const std::filesystem::path& currentPath)
{
    std::error_code error;
    if (!currentPath.empty())
    {
        std::filesystem::path candidate = std::filesystem::is_directory(currentPath, error)
                                              ? currentPath : currentPath.parent_path();
        error.clear();
        if (!candidate.empty() && std::filesystem::is_directory(candidate, error))
            return candidate;
    }

    std::filesystem::path search = std::filesystem::current_path(error);
    for (int level = 0; !search.empty() && level < 6; ++level)
    {
        const std::filesystem::path assets = search / L"assets";
        error.clear();
        if (std::filesystem::is_directory(assets, error))
            return assets;
        const std::filesystem::path parent = search.parent_path();
        if (parent == search)
            break;
        search = parent;
    }
    return {};
}

FileDialogResult showOpenFileDialog(GLFWwindow* window, const wchar_t* title,
                                    const COMDLG_FILTERSPEC* filters, UINT filterCount,
                                    const wchar_t* defaultExtension,
                                    const std::filesystem::path& currentPath)
{
    const HRESULT initializeResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    const bool uninitialize = SUCCEEDED(initializeResult);
    if (FAILED(initializeResult) && initializeResult != RPC_E_CHANGED_MODE)
        return {.error = hresultMessage("CoInitializeEx", initializeResult)};

    FileDialogResult result;
    ComPtr<IFileOpenDialog> dialog;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(dialog.GetAddressOf()));
    if (FAILED(hr))
        result.error = hresultMessage("CoCreateInstance(IFileOpenDialog)", hr);
    else
    {
        FILEOPENDIALOGOPTIONS options{};
        if (SUCCEEDED(dialog->GetOptions(&options)))
            dialog->SetOptions(options | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST |
                               FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR);
        dialog->SetTitle(title);
        dialog->SetFileTypes(filterCount, filters);
        dialog->SetFileTypeIndex(1);
        dialog->SetDefaultExtension(defaultExtension);

        const std::filesystem::path initialFolder = dialogInitialFolder(currentPath);
        if (!initialFolder.empty())
        {
            ComPtr<IShellItem> folder;
            if (SUCCEEDED(SHCreateItemFromParsingName(initialFolder.c_str(), nullptr,
                                                      IID_PPV_ARGS(folder.GetAddressOf()))))
                dialog->SetFolder(folder.Get());
        }

        hr = dialog->Show(window ? glfwGetWin32Window(window) : nullptr);
        if (hr != HRESULT_FROM_WIN32(ERROR_CANCELLED))
        {
            if (FAILED(hr))
                result.error = hresultMessage("IFileOpenDialog::Show", hr);
            else
            {
                ComPtr<IShellItem> item;
                hr = dialog->GetResult(item.GetAddressOf());
                if (FAILED(hr))
                    result.error = hresultMessage("IFileOpenDialog::GetResult", hr);
                else
                {
                    PWSTR selectedPath = nullptr;
                    hr = item->GetDisplayName(SIGDN_FILESYSPATH, &selectedPath);
                    if (FAILED(hr))
                        result.error = hresultMessage("IShellItem::GetDisplayName", hr);
                    else
                    {
                        result.path = std::filesystem::path(selectedPath);
                        CoTaskMemFree(selectedPath);
                    }
                }
            }
        }
    }

    dialog.Reset();
    if (uninitialize)
        CoUninitialize();
    return result;
}

template <std::size_t Size>
bool writePathField(std::array<char, Size>& field, const std::filesystem::path& path, std::string& error)
{
    const std::string utf8 = pathToUtf8(path);
    if (utf8.empty() || utf8.size() >= field.size())
    {
        error = "The selected path is too long for the path field";
        return false;
    }
    std::fill(field.begin(), field.end(), '\0');
    std::memcpy(field.data(), utf8.data(), utf8.size());
    return true;
}
} // namespace

int Application::run()
{
    if (!initialize())
    {
        shutdown();
        return 1;
    }

    std::uint64_t testFrameLimit = 0;
    if (const char* testFrames = std::getenv("VOR_TEST_FRAMES"); testFrames && *testFrames)
        testFrameLimit = std::strtoull(testFrames, nullptr, 10);
    std::uint64_t renderedFrames = 0;
    bool renderLoopSucceeded = true;
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

        const auto framebufferWidth = static_cast<std::uint32_t>(width);
        const auto framebufferHeight = static_cast<std::uint32_t>(height);
        vulkanRenderer_.resize(framebufferWidth, framebufferHeight);
        optixRenderer_.resize(framebufferWidth, framebufferHeight);
        if (interopWidth_ != framebufferWidth || interopHeight_ != framebufferHeight)
        {
            optixRenderer_.clearGpuInteropSurface();
            if (vulkanRenderer_.createGpuInteropSurface(framebufferWidth, framebufferHeight) &&
                optixRenderer_.setGpuInteropSurface(vulkanRenderer_.gpuInteropSurface()))
            {
                interopWidth_ = framebufferWidth;
                interopHeight_ = framebufferHeight;
            }
        }
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
            if (!optixRenderer_.render(scene_.camera, settings_))
            {
                renderLoopSucceeded = false;
                break;
            }
            vulkanRenderer_.setGpuInteropFrameReady(true);
        }
        else if (settings_.backend == BackendKind::VulkanHybrid && settings_.denoiser &&
                 optixRenderer_.available())
        {
            if (vulkanRenderer_.renderDenoiserInput(scene_.camera, settings_) &&
                optixRenderer_.denoiseVulkanFrame(settings_.exposure))
                vulkanRenderer_.setGpuInteropFrameReady(true);
            else
                vulkanRenderer_.clearExternalImage();
        }
        else
            vulkanRenderer_.clearExternalImage();
        if (!vulkanRenderer_.render(scene_.camera, settings_))
        {
            renderLoopSucceeded = false;
            break;
        }
        ++renderedFrames;
        if (testFrameLimit > 0 && renderedFrames >= testFrameLimit)
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }

    const RendererStats& finalStats = settings_.backend == BackendKind::Optix && optixRenderer_.available()
                                          ? optixRenderer_.stats() : vulkanRenderer_.stats();
    log(LogLevel::Info, "Render summary: " + std::to_string(renderedFrames) + " frames, CPU " +
                            std::to_string(finalStats.frameMilliseconds) + " ms, GPU " +
                            std::to_string(finalStats.gpuMilliseconds) + " ms, scene " +
                            std::to_string(finalStats.gpuSceneBytes) + " bytes");
    shutdown();
    return renderLoopSucceeded ? 0 : 1;
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
    if (const char* requestedDenoiser = std::getenv("VOR_DENOISER");
        requestedDenoiser && std::strcmp(requestedDenoiser, "1") == 0)
        settings_.denoiser = true;
    if (const char* requestedMeshletDebug = std::getenv("VOR_MESHLET_DEBUG");
        requestedMeshletDebug && std::strcmp(requestedMeshletDebug, "1") == 0)
        settings_.showMeshlets = true;
    if (const char* requestedReflections = std::getenv("VOR_REFLECTIONS"); requestedReflections)
        settings_.rayTracedReflections = std::strcmp(requestedReflections, "0") != 0;
    if (const char* requestedIndirect = std::getenv("VOR_INDIRECT_LIGHTING"); requestedIndirect)
        settings_.indirectLighting = std::strcmp(requestedIndirect, "0") != 0;
    if (const char* requestedDebugView = std::getenv("VOR_DEBUG_VIEW"); requestedDebugView && *requestedDebugView)
    {
        const unsigned long value = std::strtoul(requestedDebugView, nullptr, 10);
        if (value <= static_cast<unsigned long>(DebugView::PathDepth))
            settings_.debugView = static_cast<DebugView>(value);
    }
    if (const char* requestedDefaultPlastic = std::getenv("VOR_DEFAULT_PLASTIC");
        requestedDefaultPlastic && std::strcmp(requestedDefaultPlastic, "1") == 0)
        defaultPlasticEnabled_ = true;

    const AssetLoadOptions loadOptions{
        .enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
        .addGroundPlane = groundPlaneEnabled_,
    };
    const char* comparisonScene = std::getenv("VOR_PBR_COMPARISON");
    scene_ = comparisonScene && std::strcmp(comparisonScene, "1") == 0
                 ? assetLoader_.createPbrMaterialComparisonScene(loadOptions)
                 : assetLoader_.createProceduralCube(loadOptions);
    frameCameraToScene();
    if (const char* requestedGlobalLight = std::getenv("VOR_GLOBAL_LIGHT"); requestedGlobalLight)
    {
        if (std::strcmp(requestedGlobalLight, "sky") == 0)
            scene_.environment.mode = GlobalLightMode::ProceduralSky;
        else if (std::strcmp(requestedGlobalLight, "hdr") == 0)
            scene_.environment.mode = GlobalLightMode::HdrEnvironment;
    }
    if (const char* startupHdr = std::getenv("VOR_HDR"); startupHdr && *startupHdr)
    {
        std::string error;
        if (!assetLoader_.loadHdrEnvironment(startupHdr, scene_.environment, error))
            statusMessage_ = "Startup HDR failed: " + error;
        else
            std::strncpy(environmentPath_.data(), startupHdr, environmentPath_.size() - 1);
    }
    if (const char* startupScene = std::getenv("VOR_SCENE"); startupScene && *startupScene)
    {
        AssetLoadResult result = assetLoader_.load(std::filesystem::path(startupScene), loadOptions);
        if (result)
        {
            Environment environment = std::move(scene_.environment);
            scene_ = std::move(result.scene);
            scene_.environment = std::move(environment);
            frameCameraToScene();
            statusMessage_ = std::string("Loaded and framed ") + startupScene;
        }
        else
            statusMessage_ = "Startup import failed: " + result.error;
    }
    if (const char* materialPreset = std::getenv("VOR_MATERIAL_PRESET"); materialPreset)
    {
        if (std::strcmp(materialPreset, "clearcoat") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.clearcoat = 1.0f;
                material.clearcoatRoughness = 0.12f;
            }
        }
        else if (std::strcmp(materialPreset, "glass") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.baseColor = {0.96f, 0.99f, 1.0f, 1.0f};
                material.metallic = 0.0f;
                material.roughness = 0.08f;
                material.transmission = 1.0f;
                material.indexOfRefraction = 1.5f;
                material.absorptionColor = {0.72f, 0.90f, 0.98f};
                material.absorptionDistance = 2.0f;
            }
        }
        else if (std::strcmp(materialPreset, "anisotropic") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.baseColor = {0.55f, 0.18f, 0.06f, 1.0f};
                material.metallic = 1.0f;
                material.roughness = 0.32f;
                material.anisotropy = 0.85f;
                material.anisotropyRotation = 0.35f;
            }
        }
        else if (std::strcmp(materialPreset, "cloth") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.baseColor = {0.18f, 0.025f, 0.035f, 1.0f};
                material.metallic = 0.0f;
                material.roughness = 0.72f;
                material.sheenColor = {0.75f, 0.18f, 0.22f};
                material.sheenRoughness = 0.58f;
            }
        }
        else if (std::strcmp(materialPreset, "emissive") == 0 && !scene_.materials.empty())
        {
            Material& material = scene_.materials.front();
            material.baseColor = {0.06f, 0.06f, 0.06f, 1.0f};
            material.emissive = {12.0f, 4.0f, 1.0f};
            material.metallic = 0.0f;
            material.roughness = 0.45f;
        }
        else if (std::strcmp(materialPreset, "wax") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.baseColor = {0.72f, 0.12f, 0.06f, 1.0f};
                material.metallic = 0.0f;
                material.roughness = 0.5f;
                material.subsurface = 0.85f;
                material.subsurfaceColor = {1.0f, 0.22f, 0.12f};
                material.subsurfaceRadius = 0.55f;
            }
        }
        else if (std::strcmp(materialPreset, "volume") == 0)
        {
            for (Material& material : scene_.materials)
            {
                material.baseColor = {0.9f, 0.96f, 1.0f, 1.0f};
                material.metallic = 0.0f;
                material.roughness = 0.12f;
                material.transmission = 1.0f;
                material.indexOfRefraction = 1.05f;
                material.absorptionColor = {1.0f, 1.0f, 1.0f};
                material.volumeAbsorption = {0.08f, 0.03f, 0.01f};
                material.volumeScattering = {0.32f, 0.38f, 0.45f};
                material.volumeDensity = 0.8f;
                material.volumeAnisotropy = 0.35f;
            }
        }
    }
    AssetLoader::setDefaultPlasticOverride(scene_, defaultPlasticEnabled_);
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
        if (ImGui::MenuItem("Open model..."))
            openModelFileDialog();
        if (ImGui::MenuItem("Open HDR environment..."))
            openHdrFileDialog();
        if (ImGui::MenuItem("Procedural cube"))
        {
            Environment environment = std::move(scene_.environment);
            scene_ = assetLoader_.createProceduralCube(
                {.enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
                 .addGroundPlane = groundPlaneEnabled_});
            scene_.environment = std::move(environment);
            AssetLoader::setDefaultPlasticOverride(scene_, defaultPlasticEnabled_);
            frameCameraToScene();
            vulkanRenderer_.setScene(&scene_);
            optixRenderer_.setScene(&scene_);
            statusMessage_ = "Loaded procedural cube";
        }
        if (ImGui::MenuItem("PBR material comparison"))
        {
            Environment environment = std::move(scene_.environment);
            scene_ = assetLoader_.createPbrMaterialComparisonScene(
                {.enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
                 .addGroundPlane = groundPlaneEnabled_});
            scene_.environment = std::move(environment);
            defaultPlasticEnabled_ = false;
            frameCameraToScene();
            vulkanRenderer_.setScene(&scene_);
            optixRenderer_.setScene(&scene_);
            statusMessage_ = "Loaded PBR material comparison scene";
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
    ImGui::SameLine();
    if (ImGui::Button("Browse...##Model"))
        openModelFileDialog();
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
    const ImVec4 plasticToggleColor = defaultPlasticEnabled_ ? ImVec4(0.16f, 0.52f, 0.24f, 1.0f)
                                                              : ImVec4(0.42f, 0.18f, 0.18f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, plasticToggleColor);
    if (ImGui::Button(defaultPlasticEnabled_ ? "Default plastic: ON" : "Default plastic: OFF",
                      ImVec2(210.0f, 0.0f)))
    {
        defaultPlasticEnabled_ = !defaultPlasticEnabled_;
        AssetLoader::setDefaultPlasticOverride(scene_, defaultPlasticEnabled_);
        vulkanRenderer_.resetAccumulation();
        optixRenderer_.resetAccumulation();
        statusMessage_ = std::string("Material override: Default Plastic ") +
                         (defaultPlasticEnabled_ ? "ON" : "OFF");
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Overrides every loaded mesh with one light-gray plastic material.\n"
                          "Albedo: 0.75 | Metallic: 0.0 | Roughness: 0.5\n"
                          "Original materials stay resident; changing it does not reload geometry.");
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
    drawMaterialEditor();
    ImGui::End();
}

void Application::drawMaterialEditor()
{
    if (!ImGui::TreeNode("Material editor"))
        return;
    if (scene_.meshes.empty() || scene_.materials.empty())
    {
        ImGui::TextDisabled("No editable material");
        ImGui::TreePop();
        return;
    }

    selectedMeshIndex_ = std::min(selectedMeshIndex_, static_cast<std::uint32_t>(scene_.meshes.size() - 1));
    selectedMaterialIndex_ = std::min(selectedMaterialIndex_, static_cast<std::uint32_t>(scene_.materials.size() - 1));
    if (ImGui::BeginCombo("Mesh", scene_.meshes[selectedMeshIndex_].name.c_str()))
    {
        for (std::uint32_t index = 0; index < scene_.meshes.size(); ++index)
        {
            if (ImGui::Selectable(scene_.meshes[index].name.c_str(), selectedMeshIndex_ == index))
            {
                selectedMeshIndex_ = index;
                selectedMaterialIndex_ = std::min(scene_.meshes[index].materialIndex,
                                                   static_cast<std::uint32_t>(scene_.materials.size() - 1));
            }
        }
        ImGui::EndCombo();
    }
    Mesh& mesh = scene_.meshes[selectedMeshIndex_];
    const std::uint32_t assignedMaterial = std::min(mesh.materialIndex,
                                                     static_cast<std::uint32_t>(scene_.materials.size() - 1));
    if (ImGui::BeginCombo("Assigned material", scene_.materials[assignedMaterial].name.c_str()))
    {
        for (std::uint32_t index = 0; index < scene_.materials.size(); ++index)
        {
            if (ImGui::Selectable(scene_.materials[index].name.c_str(), assignedMaterial == index))
            {
                mesh.materialIndex = index;
                selectedMaterialIndex_ = index;
                vulkanRenderer_.setScene(&scene_);
                optixRenderer_.setScene(&scene_);
                statusMessage_ = "Updated mesh material assignment";
            }
        }
        ImGui::EndCombo();
    }

    if (ImGui::BeginCombo("Edit material", scene_.materials[selectedMaterialIndex_].name.c_str()))
    {
        for (std::uint32_t index = 0; index < scene_.materials.size(); ++index)
        {
            if (ImGui::Selectable(scene_.materials[index].name.c_str(), selectedMaterialIndex_ == index))
                selectedMaterialIndex_ = index;
        }
        ImGui::EndCombo();
    }

    Material& material = scene_.materials[selectedMaterialIndex_];
    ImGui::TextDisabled("Vulkan: bounded transmission, thickness SSS and homogeneous fog approximations.");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("OptiX evaluates the full stochastic BSDF/medium transport.\n"
                          "Vulkan remains non-progressive and uses a fixed real-time Ray Query budget.");
    const Vec3 previousEmissive = material.emissive;
    bool changed = false;
    changed |= ImGui::ColorEdit4("Base color", &material.baseColor.x);
    changed |= ImGui::ColorEdit3("Emissive", &material.emissive.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
    changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.02f, 1.0f);
    changed |= ImGui::DragFloat("Normal scale", &material.normalScale, 0.01f, 0.0f, 4.0f);
    changed |= ImGui::SliderFloat("AO strength", &material.occlusionStrength, 0.0f, 1.0f);
    int alphaMode = static_cast<int>(material.alphaMode);
    static constexpr const char* alphaModeNames[] = {"Opaque", "Mask", "Blend"};
    if (ImGui::Combo("Alpha mode", &alphaMode, alphaModeNames, static_cast<int>(std::size(alphaModeNames))))
    {
        material.alphaMode = static_cast<AlphaMode>(alphaMode);
        changed = true;
    }
    changed |= ImGui::SliderFloat("Alpha cutoff", &material.alphaCutoff, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Transmission", &material.transmission, 0.0f, 1.0f);
    changed |= ImGui::DragFloat("IOR", &material.indexOfRefraction, 0.01f, 1.0f, 3.0f);
    changed |= ImGui::SliderFloat("Clearcoat", &material.clearcoat, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Clearcoat roughness", &material.clearcoatRoughness, 0.02f, 1.0f);
    changed |= ImGui::SliderFloat("Anisotropy", &material.anisotropy, -0.99f, 0.99f);
    changed |= ImGui::SliderFloat("Anisotropy rotation", &material.anisotropyRotation, 0.0f, 1.0f);
    changed |= ImGui::ColorEdit3("Sheen color", &material.sheenColor.x);
    changed |= ImGui::SliderFloat("Sheen roughness", &material.sheenRoughness, 0.0f, 1.0f);
    changed |= ImGui::ColorEdit3("Absorption color", &material.absorptionColor.x);
    changed |= ImGui::DragFloat("Absorption distance", &material.absorptionDistance, 0.05f, 0.0001f, 10000.0f);
    changed |= ImGui::SliderFloat("Subsurface", &material.subsurface, 0.0f, 1.0f);
    changed |= ImGui::ColorEdit3("Subsurface color", &material.subsurfaceColor.x);
    changed |= ImGui::DragFloat("Subsurface radius", &material.subsurfaceRadius, 0.01f, 0.001f, 100.0f);
    changed |= ImGui::ColorEdit3("Volume absorption", &material.volumeAbsorption.x,
                                 ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Volume scattering", &material.volumeScattering.x,
                                 ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
    changed |= ImGui::DragFloat("Volume density", &material.volumeDensity, 0.01f, 0.0f, 100.0f);
    changed |= ImGui::SliderFloat("Volume anisotropy", &material.volumeAnisotropy, -0.99f, 0.99f);
    changed |= ImGui::Checkbox("Double sided", &material.doubleSided);

    if (ImGui::TreeNode("Texture assignments"))
    {
        ImGui::Text("Base color: %d | Normal: %d | Metallic-Roughness: %d", material.baseColorTexture,
                    material.normalTexture, material.metallicRoughnessTexture);
        ImGui::Text("AO: %d | Emissive: %d | Transmission: %d", material.occlusionTexture,
                    material.emissiveTexture, material.transmissionTexture);
        ImGui::Text("Clearcoat: %d | Roughness: %d | Normal: %d", material.clearcoatTexture,
                    material.clearcoatRoughnessTexture, material.clearcoatNormalTexture);
        ImGui::Text("Sheen color: %d | Roughness: %d | Anisotropy: %d", material.sheenColorTexture,
                    material.sheenRoughnessTexture, material.anisotropyTexture);
        ImGui::TreePop();
    }
    ImGui::TextDisabled("Factor edits update only the resident GPU material buffers.");

    if (changed)
    {
        material.baseColor.x = std::max(material.baseColor.x, 0.0f);
        material.baseColor.y = std::max(material.baseColor.y, 0.0f);
        material.baseColor.z = std::max(material.baseColor.z, 0.0f);
        material.emissive.x = std::max(material.emissive.x, 0.0f);
        material.emissive.y = std::max(material.emissive.y, 0.0f);
        material.emissive.z = std::max(material.emissive.z, 0.0f);
        const bool emissionChanged = previousEmissive.x != material.emissive.x ||
                                     previousEmissive.y != material.emissive.y ||
                                     previousEmissive.z != material.emissive.z;
        if (emissionChanged || !vulkanRenderer_.updateMaterial(selectedMaterialIndex_) ||
            !optixRenderer_.updateMaterial(selectedMaterialIndex_))
        {
            // Emission changes also rebuild the OptiX mesh-light CDF. The CPU scene is not re-imported.
            vulkanRenderer_.setScene(&scene_);
            optixRenderer_.setScene(&scene_);
        }
        else
        {
            vulkanRenderer_.resetAccumulation();
            optixRenderer_.resetAccumulation();
        }
        statusMessage_ = "Updated material GPU data: " + material.name;
    }
    ImGui::TreePop();
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
    if (settings_.backend != BackendKind::Optix)
        ImGui::BeginDisabled();
    ImGui::SliderInt("Samples/frame", reinterpret_cast<int*>(&settings_.samplesPerFrame), 1, 16);
    if (ImGui::SliderInt("Max bounces", reinterpret_cast<int*>(&settings_.maxBounces), 1, 16))
        optixRenderer_.resetAccumulation();
    if (settings_.backend != BackendKind::Optix)
        ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("OptiX path-tracing setting. Vulkan uses a fixed real-time Ray Query budget.");
    ImGui::SliderFloat("Exposure", &settings_.exposure, -8.0f, 8.0f);
    ImGui::Checkbox("Ray-traced shadows", &settings_.rayTracedShadows);
    if (ImGui::Checkbox("Ray-traced reflections", &settings_.rayTracedReflections))
    {
        vulkanRenderer_.resetAccumulation();
        optixRenderer_.resetAccumulation();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Vulkan: one inline Ray Query reflection.\nOptiX: rough PBR reflection paths up to Max bounces.");
    if (ImGui::Checkbox("Indirect lighting", &settings_.indirectLighting))
        resetCameraAccumulation();
    if (settings_.backend != BackendKind::VulkanHybrid || !optixRenderer_.available())
        ImGui::BeginDisabled();
    if (ImGui::Checkbox("Vulkan post-render denoiser", &settings_.denoiser))
        resetCameraAccumulation();
    if (settings_.backend != BackendKind::VulkanHybrid || !optixRenderer_.available())
        ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Optional NVIDIA OptiX AI denoiser applied after Vulkan rendering.\n"
                          "The OptiX path tracer itself is always shown without denoising.");
    if (settings_.backend != BackendKind::VulkanHybrid)
        ImGui::BeginDisabled();
    ImGui::Checkbox("Meshlet debug colors", &settings_.showMeshlets);
    if (settings_.backend != BackendKind::VulkanHybrid)
        ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Shows every Vulkan meshlet with a stable pseudo-random color.");
    static constexpr const char* debugViewNames[] = {
        "Beauty", "Base color", "Metallic", "Roughness", "Shading normal", "Geometric normal",
        "Tangent", "Bitangent", "AO", "Emissive", "Diffuse lobe", "Specular lobe", "Clearcoat lobe",
        "Sheen lobe", "Transmission lobe", "BSDF PDF", "Material ID", "Texture IDs", "Medium", "Path depth"};
    int debugView = static_cast<int>(settings_.debugView);
    if (ImGui::Combo("Debug view", &debugView, debugViewNames,
                     static_cast<int>(std::size(debugViewNames))))
    {
        settings_.debugView = static_cast<DebugView>(debugView);
        resetCameraAccumulation();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Shared material diagnostics. Vulkan transmission, subsurface and volume views show the documented real-time approximations.");

    ImGui::SeparatorText("Global light");
    int globalLightMode = static_cast<int>(scene_.environment.mode);
    bool environmentChanged = false;
    environmentChanged |= ImGui::RadioButton("Directional", &globalLightMode,
                                              static_cast<int>(GlobalLightMode::Directional));
    ImGui::SameLine();
    environmentChanged |= ImGui::RadioButton("HDR", &globalLightMode,
                                              static_cast<int>(GlobalLightMode::HdrEnvironment));
    ImGui::SameLine();
    environmentChanged |= ImGui::RadioButton("Procedural sky", &globalLightMode,
                                              static_cast<int>(GlobalLightMode::ProceduralSky));
    if (environmentChanged)
    {
        scene_.environment.mode = static_cast<GlobalLightMode>(globalLightMode);
        resetCameraAccumulation();
    }
    if (scene_.environment.mode == GlobalLightMode::HdrEnvironment)
    {
        ImGui::InputText("HDR path", environmentPath_.data(), environmentPath_.size());
        ImGui::SameLine();
        if (ImGui::Button("Load HDR"))
            loadEnvironmentFromUi();
        ImGui::SameLine();
        if (ImGui::Button("Browse...##HDR"))
            openHdrFileDialog();
        if (scene_.environment.hasHdr())
            ImGui::TextDisabled("%ux%u - %s", scene_.environment.hdrWidth, scene_.environment.hdrHeight,
                                scene_.environment.hdrPath.filename().string().c_str());
        else
            ImGui::TextDisabled("No HDR loaded");
    }
    bool lightingChanged = ImGui::DragFloat("Environment intensity", &scene_.environment.intensity,
                                             0.02f, 0.0f, 100.0f, "%.2f");
    lightingChanged |= ImGui::Checkbox("Visible background", &scene_.environment.visibleBackground);
    if (scene_.environment.mode == GlobalLightMode::ProceduralSky)
    {
        lightingChanged |= ImGui::ColorEdit3("Sky zenith", &scene_.environment.zenithColor.x);
        lightingChanged |= ImGui::ColorEdit3("Sky horizon", &scene_.environment.horizonColor.x);
        lightingChanged |= ImGui::ColorEdit3("Sky ground", &scene_.environment.groundColor.x);
    }
    if (lightingChanged)
    {
        scene_.environment.intensity = std::max(scene_.environment.intensity, 0.0f);
        resetCameraAccumulation();
    }

    ImGui::SeparatorText("Shared light transform");
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
            syncGlobalLightTransform();
            resetCameraAccumulation();
        }
    }
    ImGui::TextDisabled("Shift+LMB: rotate all light modes | Shift+MMB: pan | Shift+wheel: zoom");

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
    ImGui::Text("GPU render: %.3f ms", stats.gpuMilliseconds);
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(stats.frameIndex));
    if (settings_.backend == BackendKind::Optix)
        ImGui::Text("Accumulated samples: %llu", static_cast<unsigned long long>(stats.accumulatedSamples));
    else
        ImGui::TextDisabled("Accumulated samples: n/a (real-time Vulkan)");
    ImGui::Text("Meshlets: %u / %u", stats.visibleMeshlets, stats.totalMeshlets);
    ImGui::Text("Rays: %llu", static_cast<unsigned long long>(stats.tracedRays));
    ImGui::Text("GPU scene: %.2f MiB", static_cast<double>(stats.gpuSceneBytes) / (1024.0 * 1024.0));
    ImGui::Text("Materials: %u (%.2f KiB)", stats.residentMaterials,
                static_cast<double>(stats.materialBytes) / 1024.0);
    ImGui::Text("Textures: %u / %u (%.2f MiB)", stats.residentTextures, stats.descriptorCapacity,
                static_cast<double>(stats.textureBytes) / (1024.0 * 1024.0));
    ImGui::Text("Ray Query feature: %s", vulkanRenderer_.rayQueryAvailable() ? "yes" : "no");
    ImGui::Text("Task Shader feature: %s", vulkanRenderer_.taskShaderAvailable() ? "yes" : "no");
    ImGui::End();
}

void Application::loadSceneFromUi()
{
    const std::filesystem::path path = pathFromUtf8(assetPath_.data());
    if (path.empty())
    {
        statusMessage_ = "Enter an asset path first";
        return;
    }
    AssetLoadResult result = assetLoader_.load(
        path, {.enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
               .addGroundPlane = groundPlaneEnabled_});
    if (!result)
    {
        statusMessage_ = "Import failed: " + result.error;
        return;
    }
    Environment environment = std::move(scene_.environment);
    scene_ = std::move(result.scene);
    scene_.environment = std::move(environment);
    AssetLoader::setDefaultPlasticOverride(scene_, defaultPlasticEnabled_);
    frameCameraToScene();
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    statusMessage_ = "Loaded and framed " + pathToUtf8(path);
}

void Application::loadEnvironmentFromUi()
{
    const std::filesystem::path path = pathFromUtf8(environmentPath_.data());
    if (path.empty())
    {
        statusMessage_ = "Enter an HDR path first";
        return;
    }
    std::string error;
    if (!assetLoader_.loadHdrEnvironment(path, scene_.environment, error))
    {
        statusMessage_ = "HDR import failed: " + error;
        return;
    }
    vulkanRenderer_.setScene(&scene_);
    optixRenderer_.setScene(&scene_);
    resetCameraAccumulation();
    statusMessage_ = "Loaded HDR environment " + pathToUtf8(path);
}

void Application::openModelFileDialog()
{
    static constexpr COMDLG_FILTERSPEC filters[]{
        {L"Supported 3D models", L"*.fbx;*.obj;*.gltf;*.glb;*.dae;*.3ds;*.ply;*.stl;*.blend"},
        {L"FBX files", L"*.fbx"},
        {L"Wavefront OBJ files", L"*.obj"},
        {L"glTF files", L"*.gltf;*.glb"},
        {L"COLLADA files", L"*.dae"},
        {L"All files", L"*.*"},
    };
    const std::filesystem::path current = assetPath_[0] ? pathFromUtf8(assetPath_.data()) : scene_.sourcePath;
    FileDialogResult result = showOpenFileDialog(window_, L"Open 3D model", filters,
                                                  static_cast<UINT>(std::size(filters)), L"fbx", current);
    if (!result.error.empty())
    {
        statusMessage_ = "File dialog failed: " + result.error;
        return;
    }
    if (!result.path)
        return;
    if (!writePathField(assetPath_, *result.path, statusMessage_))
        return;
    loadSceneFromUi();
}

void Application::openHdrFileDialog()
{
    static constexpr COMDLG_FILTERSPEC filters[]{
        {L"Radiance HDR images", L"*.hdr"},
        {L"All files", L"*.*"},
    };
    const std::filesystem::path current = environmentPath_[0]
                                              ? pathFromUtf8(environmentPath_.data())
                                              : scene_.environment.hdrPath;
    FileDialogResult result = showOpenFileDialog(window_, L"Open HDR environment", filters,
                                                  static_cast<UINT>(std::size(filters)), L"hdr", current);
    if (!result.error.empty())
    {
        statusMessage_ = "File dialog failed: " + result.error;
        return;
    }
    if (!result.path)
        return;
    if (!writePathField(environmentPath_, *result.path, statusMessage_))
        return;
    scene_.environment.mode = GlobalLightMode::HdrEnvironment;
    loadEnvironmentFromUi();
}

bool Application::reloadCurrentScene()
{
    Scene reloadedScene;
    const AssetLoadOptions options{
        .enableOptionalMeshoptimizerPasses = meshoptimizerEnabled_,
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

    Environment environment = std::move(scene_.environment);
    scene_ = std::move(reloadedScene);
    scene_.environment = std::move(environment);
    AssetLoader::setDefaultPlasticOverride(scene_, defaultPlasticEnabled_);
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
        scene_.lights.push_back(Light{.name = "Sun", .type = LightType::Directional});
    Light& light = scene_.lights.front();
    lightTarget_ = center;
    Vec3 lightDirection = normalize(light.direction);
    if (length(lightDirection) < 0.5f)
        lightDirection = normalize(Vec3{0.45f, -0.85f, -0.3f});
    const float lightDistance = radius * 2.5f;
    light.position = lightTarget_ - lightDirection * lightDistance;
    light.direction = normalize(lightTarget_ - light.position);
    light.type = LightType::Directional;
    syncGlobalLightTransform();
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
            syncGlobalLightTransform();
            light->range = std::max(length(lightTarget_ - light->position) * 4.0f, 1.0f);
        }
        resetCameraAccumulation();
    }
}

void Application::syncGlobalLightTransform()
{
    if (scene_.lights.empty())
        return;
    Light& light = scene_.lights.front();
    const Vec3 direction = lightTarget_ - light.position;
    if (length(direction) > 1e-6f)
        light.direction = normalize(direction);
    light.type = LightType::Directional;
    // Equirectangular HDR maps only have a meaningful rotational component. The same
    // shared transform drives the sun direction of the procedural sky.
    scene_.environment.rotationRadians = std::atan2(light.direction.x, light.direction.z);
}

void Application::resetCameraAccumulation()
{
    vulkanRenderer_.resetAccumulation();
    optixRenderer_.resetAccumulation();
}
} // namespace vor
