#include "Vulkan/VulkanRenderer.h"

#include "Core/Log.h"

#include <Windows.h>

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace vor
{
namespace
{
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*)
{
    const LogLevel level = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                               ? LogLevel::Error
                               : severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? LogLevel::Warning : LogLevel::Info;
    log(level, callbackData && callbackData->pMessage ? callbackData->pMessage : "Vulkan validation message");
    return VK_FALSE;
}

bool hasExtension(std::span<const VkExtensionProperties> extensions, std::string_view name)
{
    return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) { return extension.extensionName == name; });
}

std::filesystem::path executableDirectory()
{
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
}

void check(VkResult result, std::string_view operation)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
}

std::vector<std::byte> readBinary(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("Cannot open shader: " + path.string());
    const std::streamsize size = stream.tellg();
    if (size <= 0 || size % 4 != 0)
        throw std::runtime_error("Invalid SPIR-V file: " + path.string());
    stream.seekg(0);
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("Cannot read shader: " + path.string());
    return data;
}
} // namespace

VulkanRenderer::~VulkanRenderer()
{
    shutdown();
}

bool VulkanRenderer::initialize(GLFWwindow* window)
{
    window_ = window;
    try
    {
        if (!createInstance() || !createSurface() || !selectPhysicalDevice() || !createDevice() || !createSwapchain() ||
            !createCommands() || !createSceneDescriptors() || !createMeshPipeline() || !initializeImGui())
            return false;
        initialized_ = true;
        log(LogLevel::Info, "Vulkan mesh renderer initialized on " + stats_.deviceName);
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        shutdown();
        return false;
    }
}

void VulkanRenderer::shutdown()
{
    if (device_ != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device_);

    if (imguiInitialized_)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized_ = false;
    }

    destroyMeshPipeline();
    if (device_ != VK_NULL_HANDLE)
    {
        for (GpuBuffer& buffer : externalImageBuffers_)
        {
            if (buffer.buffer)
                vkDestroyBuffer(device_, buffer.buffer, nullptr);
            if (buffer.memory)
                vkFreeMemory(device_, buffer.memory, nullptr);
            buffer = {};
        }
        destroySceneResources();
        if (sceneDescriptorPool_)
            vkDestroyDescriptorPool(device_, sceneDescriptorPool_, nullptr);
        if (sceneDescriptorSetLayout_)
            vkDestroyDescriptorSetLayout(device_, sceneDescriptorSetLayout_, nullptr);
        sceneDescriptorPool_ = VK_NULL_HANDLE;
        sceneDescriptorSetLayout_ = VK_NULL_HANDLE;
        sceneDescriptorSet_ = VK_NULL_HANDLE;
        for (FrameResources& frame : frames_)
        {
            if (frame.inFlight)
                vkDestroyFence(device_, frame.inFlight, nullptr);
            if (frame.imageAvailable)
                vkDestroySemaphore(device_, frame.imageAvailable, nullptr);
            if (frame.renderFinished)
                vkDestroySemaphore(device_, frame.renderFinished, nullptr);
            frame = {};
        }
        if (commandPool_)
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        destroySwapchain();
        vkDestroyDevice(device_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    commandPool_ = VK_NULL_HANDLE;

    if (surface_ && instance_)
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;

#if VOR_DEBUG
    if (debugMessenger_ && instance_)
    {
        const auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyMessenger)
            destroyMessenger(instance_, debugMessenger_, nullptr);
    }
#endif
    debugMessenger_ = VK_NULL_HANDLE;
    if (instance_)
        vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
    initialized_ = false;
}

void VulkanRenderer::setScene(const Scene* scene)
{
    scene_ = scene;
    stats_.totalMeshlets = scene ? static_cast<std::uint32_t>(scene->meshletCount()) : 0;
    if (device_)
        uploadSceneResources();
    resetAccumulation();
}

void VulkanRenderer::resize(std::uint32_t width, std::uint32_t height)
{
    requestedWidth_ = width;
    requestedHeight_ = height;
    resizePending_ = width > 0 && height > 0;
}

void VulkanRenderer::resetAccumulation()
{
    stats_.accumulatedSamples = 0;
}

void VulkanRenderer::beginUiFrame()
{
    if (!imguiInitialized_)
        return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VulkanRenderer::setExternalImage(std::span<const std::uint32_t> rgbaPixels,
                                      std::uint32_t width,
                                      std::uint32_t height)
{
    if (rgbaPixels.size() != static_cast<std::size_t>(width) * height || rgbaPixels.empty())
    {
        clearExternalImage();
        return;
    }
    externalImagePixels_.assign(rgbaPixels.begin(), rgbaPixels.end());
    if (swapchainFormat_ == VK_FORMAT_B8G8R8A8_SRGB || swapchainFormat_ == VK_FORMAT_B8G8R8A8_UNORM)
    {
        for (std::uint32_t& pixel : externalImagePixels_)
        {
            const std::uint32_t red = pixel & 0xffu;
            const std::uint32_t blue = (pixel >> 16u) & 0xffu;
            pixel = (pixel & 0xff00ff00u) | (red << 16u) | blue;
        }
    }
    externalImageWidth_ = width;
    externalImageHeight_ = height;
    useExternalImage_ = true;
}

void VulkanRenderer::clearExternalImage()
{
    useExternalImage_ = false;
}

bool VulkanRenderer::render(const Camera& camera, const RenderSettings& settings)
{
    if (!initialized_)
        return false;

    const auto begin = std::chrono::steady_clock::now();
    try
    {
        if (resizePending_ && !recreateSwapchain())
            return false;

        const float aspect = static_cast<float>(swapchainExtent_.width) / static_cast<float>(std::max(swapchainExtent_.height, 1u));
        const Mat4 projection = perspective(camera.verticalFovDegrees * kPi / 180.0f, aspect, camera.nearPlane, camera.farPlane);
        const Mat4 view = lookAt(camera.position, camera.target, camera.up);
        framePushConstants_.viewProjection = projection * view;
        framePushConstants_.model = scene_ && !scene_->instances.empty() ? scene_->instances.front().transform : Mat4::identity();
        framePushConstants_.cameraPosition = {camera.position.x, camera.position.y, camera.position.z, 1.0f};
        framePushConstants_.rayTracedShadows = settings.rayTracedShadows && tlas_ ? 1u : 0u;
        Material material{};
        if (scene_ && !scene_->meshes.empty() && scene_->meshes.front().materialIndex < scene_->materials.size())
            material = scene_->materials[scene_->meshes.front().materialIndex];
        framePushConstants_.baseColorAndMetallic = {material.baseColor.x, material.baseColor.y, material.baseColor.z,
                                                    material.metallic};
        framePushConstants_.materialParameters = {material.roughness, 0.0f, 0.0f, 0.0f};

        ImGui::Render();
        FrameResources& frame = frames_[frameSlot_];
        check(vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX), "vkWaitForFences");

        std::uint32_t imageIndex = 0;
        VkResult acquireResult = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            resizePending_ = true;
            return true;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
            check(acquireResult, "vkAcquireNextImageKHR");

        const bool externalImageMatches = useExternalImage_ && externalImageWidth_ == swapchainExtent_.width &&
                                          externalImageHeight_ == swapchainExtent_.height;
        const GpuBuffer* externalBuffer = nullptr;
        if (externalImageMatches && updateExternalBuffer(frameSlot_))
            externalBuffer = &externalImageBuffers_[frameSlot_];

        check(vkResetFences(device_, 1, &frame.inFlight), "vkResetFences");
        check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");
        recordCommands(frame.commandBuffer, imageIndex, externalBuffer);

        const VkPipelineStageFlags waitStage = externalBuffer
                                                   ? VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                                   : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &frame.imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &frame.commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &frame.renderFinished;
        check(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, frame.inFlight), "vkQueueSubmit");

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &frame.renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult presentResult = vkQueuePresentKHR(graphicsQueue_, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
            resizePending_ = true;
        else
            check(presentResult, "vkQueuePresentKHR");

        frameSlot_ = (frameSlot_ + 1) % kFramesInFlight;
        ++stats_.frameIndex;
        stats_.accumulatedSamples += settings.samplesPerFrame;
        stats_.visibleMeshlets = stats_.totalMeshlets;
        stats_.tracedRays = rayQueryAvailable_ && settings.rayTracedShadows
                                ? static_cast<std::uint64_t>(swapchainExtent_.width) * swapchainExtent_.height
                                : 0;
        stats_.frameMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

bool VulkanRenderer::updateExternalBuffer(std::uint32_t frameIndex)
{
    if (frameIndex >= externalImageBuffers_.size() || externalImagePixels_.empty())
        return false;
    const VkDeviceSize byteCount = externalImagePixels_.size() * sizeof(std::uint32_t);
    GpuBuffer& buffer = externalImageBuffers_[frameIndex];
    if (buffer.size != byteCount)
    {
        if (buffer.buffer)
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory)
            vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = createUploadBuffer(externalImagePixels_.data(), byteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        return true;
    }
    void* mapped = nullptr;
    check(vkMapMemory(device_, buffer.memory, 0, byteCount, 0, &mapped), "vkMapMemory(OptiX display)");
    std::memcpy(mapped, externalImagePixels_.data(), static_cast<std::size_t>(byteCount));
    vkUnmapMemory(device_, buffer.memory);
    return true;
}

bool VulkanRenderer::createInstance()
{
    VkApplicationInfo applicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    applicationInfo.pApplicationName = "VORaytracer";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.pEngineName = "VORaytracer";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    std::uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (!glfwExtensions)
        throw std::runtime_error("GLFW returned no Vulkan instance extensions");
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
#if VOR_DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    std::vector<const char*> layers;
#if VOR_DEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback = debugCallback;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();
#if VOR_DEBUG
    createInfo.pNext = &debugInfo;
#endif
    check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");

#if VOR_DEBUG
    const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (createMessenger)
        check(createMessenger(instance_, &debugInfo, nullptr, &debugMessenger_), "vkCreateDebugUtilsMessengerEXT");
#endif
    return true;
}

bool VulkanRenderer::createSurface()
{
    check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_), "glfwCreateWindowSurface");
    return true;
}

bool VulkanRenderer::selectPhysicalDevice()
{
    std::uint32_t deviceCount = 0;
    check(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
    if (deviceCount == 0)
        throw std::runtime_error("No Vulkan physical device found");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    check(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

    int bestScore = -1;
    for (VkPhysicalDevice candidate : devices)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        std::uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, extensions.data());
        if (!hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME) || !hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME))
            continue;

        std::uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queues.data());
        std::optional<std::uint32_t> queueIndex;
        for (std::uint32_t index = 0; index < queueCount; ++index)
        {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, index, surface_, &present);
            if ((queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present)
            {
                queueIndex = index;
                break;
            }
        }
        if (!queueIndex)
            continue;

        VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &meshFeatures;
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        if (!meshFeatures.meshShader)
            continue;

        const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 100;
        if (score > bestScore)
        {
            bestScore = score;
            physicalDevice_ = candidate;
            graphicsQueueFamily_ = *queueIndex;
        }
    }

    if (!physicalDevice_)
        throw std::runtime_error("No Vulkan device with VK_EXT_mesh_shader and presentation support found");

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    stats_.deviceName = properties.deviceName;
    return true;
}

bool VulkanRenderer::createDevice()
{
    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extensionCount, supportedExtensions.data());

    const bool accelerationStructure = hasExtension(supportedExtensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    const bool deferredHost = hasExtension(supportedExtensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool rayQuery = hasExtension(supportedExtensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    const bool rayQueryExtensions = accelerationStructure && deferredHost && rayQuery;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = graphicsQueueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceRayQueryFeaturesKHR availableRayQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR availableAcceleration{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceMeshShaderFeaturesEXT availableMesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features available13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures availableBufferAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

    VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    available.pNext = &available13;
    available13.pNext = &availableBufferAddress;
    availableBufferAddress.pNext = &availableMesh;
    if (rayQueryExtensions)
    {
        availableMesh.pNext = &availableAcceleration;
        availableAcceleration.pNext = &availableRayQuery;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &available);

    if (!available13.dynamicRendering || !available13.synchronization2 || !availableMesh.meshShader)
        throw std::runtime_error("Required Vulkan 1.3 dynamic rendering/synchronization or mesh shader features are missing");
    taskShaderAvailable_ = availableMesh.taskShader == VK_TRUE;
    rayQueryAvailable_ = rayQueryExtensions && availableBufferAddress.bufferDeviceAddress &&
                         availableAcceleration.accelerationStructure && availableRayQuery.rayQuery;
    if (!rayQueryAvailable_)
        throw std::runtime_error("The Vulkan backend requires acceleration structures and VK_KHR_ray_query");

    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};
    if (rayQueryAvailable_)
    {
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.pNext = &bufferDeviceAddress;
    bufferDeviceAddress.bufferDeviceAddress = rayQueryAvailable_ ? VK_TRUE : VK_FALSE;
    bufferDeviceAddress.pNext = &meshFeatures;
    meshFeatures.meshShader = VK_TRUE;
    meshFeatures.taskShader = taskShaderAvailable_ ? VK_TRUE : VK_FALSE;
    if (rayQueryAvailable_)
    {
        meshFeatures.pNext = &accelerationFeatures;
        accelerationFeatures.accelerationStructure = VK_TRUE;
        accelerationFeatures.pNext = &rayQueryFeatures;
        rayQueryFeatures.rayQuery = VK_TRUE;
    }

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.pNext = &features13;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, graphicsQueueFamily_, 0, &graphicsQueue_);
    cmdDrawMeshTasks_ = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(device_, "vkCmdDrawMeshTasksEXT"));
    if (!cmdDrawMeshTasks_)
        throw std::runtime_error("vkCmdDrawMeshTasksEXT is unavailable");
    createAccelerationStructure_ = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device_, "vkCreateAccelerationStructureKHR"));
    destroyAccelerationStructure_ = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device_, "vkDestroyAccelerationStructureKHR"));
    getAccelerationStructureBuildSizes_ = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(device_, "vkGetAccelerationStructureBuildSizesKHR"));
    cmdBuildAccelerationStructures_ = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(device_, "vkCmdBuildAccelerationStructuresKHR"));
    getAccelerationStructureDeviceAddress_ = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(device_, "vkGetAccelerationStructureDeviceAddressKHR"));
    if (!createAccelerationStructure_ || !destroyAccelerationStructure_ || !getAccelerationStructureBuildSizes_ ||
        !cmdBuildAccelerationStructures_ || !getAccelerationStructureDeviceAddress_)
        throw std::runtime_error("Required Vulkan acceleration-structure entry points are unavailable");
    return true;
}

bool VulkanRenderer::createSwapchain()
{
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, formats.data());
    std::uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &modeCount, modes.data());

    VkSurfaceFormatKHR selectedFormat = formats.front();
    for (const VkSurfaceFormatKHR format : formats)
    {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            selectedFormat = format;
            break;
        }
    }
    swapchainFormat_ = selectedFormat.format;

    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        swapchainExtent_ = capabilities.currentExtent;
    else
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        swapchainExtent_.width = std::clamp<std::uint32_t>(static_cast<std::uint32_t>(std::max(width, 1)), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        swapchainExtent_.height = std::clamp<std::uint32_t>(static_cast<std::uint32_t>(std::max(height, 1)), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    const VkPresentModeKHR presentMode = std::ranges::find(modes, VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()
                                                 ? VK_PRESENT_MODE_MAILBOX_KHR
                                                 : VK_PRESENT_MODE_FIFO_KHR;
    std::uint32_t imageCount = std::max(capabilities.minImageCount + 1, 2u);
    if (capabilities.maxImageCount > 0)
        imageCount = std::min(imageCount, capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = selectedFormat.format;
    createInfo.imageColorSpace = selectedFormat.colorSpace;
    createInfo.imageExtent = swapchainExtent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_), "vkCreateSwapchainKHR");

    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());
    swapchainImageViews_.resize(imageCount);
    swapchainImageInitialized_.assign(imageCount, false);
    for (std::size_t index = 0; index < swapchainImages_.size(); ++index)
    {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = swapchainImages_[index];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat_;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        check(vkCreateImageView(device_, &viewInfo, nullptr, &swapchainImageViews_[index]), "vkCreateImageView");
    }
    return true;
}

bool VulkanRenderer::createCommands()
{
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamily_;
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");

    std::array<VkCommandBuffer, kFramesInFlight> buffers{};
    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = kFramesInFlight;
    check(vkAllocateCommandBuffers(device_, &allocateInfo, buffers.data()), "vkAllocateCommandBuffers");
    for (std::uint32_t index = 0; index < kFramesInFlight; ++index)
    {
        frames_[index].commandBuffer = buffers[index];
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frames_[index].imageAvailable), "vkCreateSemaphore");
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frames_[index].renderFinished), "vkCreateSemaphore");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(device_, &fenceInfo, nullptr, &frames_[index].inFlight), "vkCreateFence");
    }
    return true;
}

bool VulkanRenderer::createSceneDescriptors()
{
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    for (std::uint32_t index = 0; index < 4; ++index)
    {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &sceneDescriptorSetLayout_), "vkCreateDescriptorSetLayout");

    const std::array poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &sceneDescriptorPool_), "vkCreateDescriptorPool");
    return true;
}

bool VulkanRenderer::createMeshPipeline()
{
    const VkShaderModule meshModule = loadShaderModule(L"Shaders\\Mesh.spv");
    const VkShaderModule fragmentModule = loadShaderModule(L"Shaders\\PbrFragment.spv");
    const std::array stages{
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "MeshMain", nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "FragmentMain", nullptr},
    };

    try
    {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(FramePushConstants);
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &sceneDescriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &meshPipelineLayout_), "vkCreatePipelineLayout");

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &attachment;
        const std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &swapchainFormat_;
        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = meshPipelineLayout_;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshPipeline_), "vkCreateGraphicsPipelines");
    }
    catch (...)
    {
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, meshModule, nullptr);
        throw;
    }
    vkDestroyShaderModule(device_, fragmentModule, nullptr);
    vkDestroyShaderModule(device_, meshModule, nullptr);
    return true;
}

bool VulkanRenderer::initializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    if (!ImGui_ImplGlfw_InitForVulkan(window_, true))
        throw std::runtime_error("ImGui GLFW initialization failed");

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance_;
    initInfo.PhysicalDevice = physicalDevice_;
    initInfo.Device = device_;
    initInfo.QueueFamily = graphicsQueueFamily_;
    initInfo.Queue = graphicsQueue_;
    initInfo.DescriptorPoolSize = 128;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = static_cast<std::uint32_t>(swapchainImages_.size());
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat_;
    if (!ImGui_ImplVulkan_Init(&initInfo))
        throw std::runtime_error("ImGui Vulkan initialization failed");
    imguiInitialized_ = true;
    return true;
}

bool VulkanRenderer::recreateSwapchain()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width <= 0 || height <= 0)
        return true;
    vkDeviceWaitIdle(device_);
    const VkFormat oldFormat = swapchainFormat_;
    destroySwapchain();
    if (!createSwapchain())
        return false;
    if (swapchainFormat_ != oldFormat)
    {
        destroyMeshPipeline();
        createMeshPipeline();
    }
    if (imguiInitialized_)
        ImGui_ImplVulkan_SetMinImageCount(2);
    resizePending_ = false;
    resetAccumulation();
    return true;
}

void VulkanRenderer::destroySwapchain()
{
    if (!device_)
        return;
    for (VkImageView view : swapchainImageViews_)
        vkDestroyImageView(device_, view, nullptr);
    swapchainImageViews_.clear();
    swapchainImages_.clear();
    swapchainImageInitialized_.clear();
    if (swapchain_)
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void VulkanRenderer::destroyMeshPipeline()
{
    if (!device_)
        return;
    if (meshPipeline_)
        vkDestroyPipeline(device_, meshPipeline_, nullptr);
    if (meshPipelineLayout_)
        vkDestroyPipelineLayout(device_, meshPipelineLayout_, nullptr);
    meshPipeline_ = VK_NULL_HANDLE;
    meshPipelineLayout_ = VK_NULL_HANDLE;
}

void VulkanRenderer::destroySceneResources()
{
    if (!device_)
        return;
    destroyAccelerationStructures();
    const auto destroyBuffer = [&](GpuBuffer& buffer) {
        if (buffer.buffer)
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory)
            vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    };
    destroyBuffer(vertexBuffer_);
    destroyBuffer(meshletBuffer_);
    destroyBuffer(meshletVertexBuffer_);
    destroyBuffer(meshletTriangleBuffer_);
    destroyBuffer(geometryIndexBuffer_);
    uploadedMeshletCount_ = 0;
    sceneDescriptorSet_ = VK_NULL_HANDLE;
    if (sceneDescriptorPool_)
        vkResetDescriptorPool(device_, sceneDescriptorPool_, 0);
}

bool VulkanRenderer::uploadSceneResources()
{
    if (!device_ || !sceneDescriptorPool_)
        return false;
    vkDeviceWaitIdle(device_);
    destroySceneResources();
    if (!scene_ || scene_->meshes.empty() || scene_->meshes.front().lods.empty())
        return true;

    struct alignas(16) GpuVertex
    {
        Vec4 position;
        Vec4 normal;
        Vec4 tangent;
        Vec4 uv;
    };
    struct alignas(16) GpuMeshlet
    {
        std::uint32_t vertexOffset;
        std::uint32_t triangleOffset;
        std::uint32_t vertexCount;
        std::uint32_t triangleCount;
        Vec4 boundingSphere;
        Vec4 normalCone;
    };

    const Mesh& mesh = scene_->meshes.front();
    const MeshLod& lod = mesh.lods.front();
    if (mesh.vertices.empty() || lod.meshlets.empty())
        return true;

    std::vector<GpuVertex> vertices;
    vertices.reserve(mesh.vertices.size());
    for (const Vertex& vertex : mesh.vertices)
    {
        vertices.push_back({{vertex.position.x, vertex.position.y, vertex.position.z, 1.0f},
                            {vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0f},
                            vertex.tangent,
                            {vertex.uv.x, vertex.uv.y, 0.0f, 0.0f}});
    }

    std::vector<GpuMeshlet> meshlets;
    std::vector<std::uint32_t> packedTriangles;
    meshlets.reserve(lod.meshlets.size());
    packedTriangles.reserve(lod.indices.size() / 3);
    for (const Meshlet& meshlet : lod.meshlets)
    {
        const std::uint32_t triangleOffset = static_cast<std::uint32_t>(packedTriangles.size());
        for (std::uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle)
        {
            const std::size_t byteOffset = static_cast<std::size_t>(meshlet.triangleOffset) + triangle * 3;
            const std::uint32_t packed = static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset]) |
                                         (static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset + 1]) << 8) |
                                         (static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset + 2]) << 16);
            packedTriangles.push_back(packed);
        }
        meshlets.push_back({meshlet.vertexOffset, triangleOffset, meshlet.vertexCount, meshlet.triangleCount,
                            meshlet.boundingSphere, meshlet.normalCone});
    }

    vertexBuffer_ = createUploadBuffer(vertices.data(), vertices.size() * sizeof(GpuVertex),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    meshletBuffer_ = createUploadBuffer(meshlets.data(), meshlets.size() * sizeof(GpuMeshlet), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshletVertexBuffer_ = createUploadBuffer(lod.meshletVertices.data(), lod.meshletVertices.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshletTriangleBuffer_ = createUploadBuffer(packedTriangles.data(), packedTriangles.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    geometryIndexBuffer_ = createUploadBuffer(lod.indices.data(), lod.indices.size() * sizeof(std::uint32_t),
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    if (!buildAccelerationStructures())
        throw std::runtime_error("Failed to build Vulkan BLAS/TLAS");

    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = sceneDescriptorPool_;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &sceneDescriptorSetLayout_;
    check(vkAllocateDescriptorSets(device_, &allocateInfo, &sceneDescriptorSet_), "vkAllocateDescriptorSets");
    const std::array<VkDescriptorBufferInfo, 4> bufferInfos{{
        {vertexBuffer_.buffer, 0, vertexBuffer_.size},
        {meshletBuffer_.buffer, 0, meshletBuffer_.size},
        {meshletVertexBuffer_.buffer, 0, meshletVertexBuffer_.size},
        {meshletTriangleBuffer_.buffer, 0, meshletTriangleBuffer_.size},
    }};
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (std::uint32_t index = 0; index < 4; ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = sceneDescriptorSet_;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &bufferInfos[index];
    }
    VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    accelerationInfo.accelerationStructureCount = 1;
    accelerationInfo.pAccelerationStructures = &tlas_;
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].pNext = &accelerationInfo;
    writes[4].dstSet = sceneDescriptorSet_;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    uploadedMeshletCount_ = static_cast<std::uint32_t>(meshlets.size());
    stats_.totalMeshlets = uploadedMeshletCount_;
    return true;
}

bool VulkanRenderer::buildAccelerationStructures()
{
    try
    {
        destroyAccelerationStructures();
        if (!scene_ || scene_->meshes.empty() || scene_->meshes.front().lods.empty())
            return false;
        const Mesh& mesh = scene_->meshes.front();
        const MeshLod& lod = mesh.lods.front();
        const std::uint32_t primitiveCount = static_cast<std::uint32_t>(lod.indices.size() / 3);
        if (primitiveCount == 0)
            return false;

        const auto deviceAddress = [&](VkBuffer buffer) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = buffer;
            return vkGetBufferDeviceAddress(device_, &info);
        };
        const auto createDeviceBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) {
            GpuBuffer result{};
            result.size = size;
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateBuffer(device_, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer(acceleration structure)");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.pNext = &flags;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device_, &allocation, nullptr, &result.memory), "vkAllocateMemory(acceleration structure)");
            check(vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory(acceleration structure)");
            return result;
        };
        const auto destroyBuffer = [&](GpuBuffer& buffer) {
            if (buffer.buffer)
                vkDestroyBuffer(device_, buffer.buffer, nullptr);
            if (buffer.memory)
                vkFreeMemory(device_, buffer.memory, nullptr);
            buffer = {};
        };
        const auto submitBuild = [&](const VkAccelerationStructureBuildGeometryInfoKHR& build,
                                     const VkAccelerationStructureBuildRangeInfoKHR& range) {
            VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocation.commandPool = commandPool_;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1;
            VkCommandBuffer command = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(device_, &allocation, &command), "vkAllocateCommandBuffers(AS build)");
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(AS build)");
            const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};
            cmdBuildAccelerationStructures_(command, 1, &build, ranges);
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer(AS build)");
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            check(vkQueueSubmit(graphicsQueue_, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit(AS build)");
            check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(AS build)");
            vkFreeCommandBuffers(device_, commandPool_, 1, &command);
        };

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = deviceAddress(vertexBuffer_.buffer);
        triangles.vertexStride = sizeof(Vec4) * 4;
        triangles.maxVertex = static_cast<std::uint32_t>(mesh.vertices.size() - 1);
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = deviceAddress(geometryIndexBuffer_.buffer);
        VkAccelerationStructureGeometryKHR blasGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        blasGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        blasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        blasGeometry.geometry.triangles = triangles;
        VkAccelerationStructureBuildGeometryInfoKHR blasBuild{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasBuild.geometryCount = 1;
        blasBuild.pGeometries = &blasGeometry;
        VkAccelerationStructureBuildSizesInfoKHR blasSizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getAccelerationStructureBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuild,
                                            &primitiveCount, &blasSizes);
        blasStorage_ = createDeviceBuffer(blasSizes.accelerationStructureSize,
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR blasCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        blasCreate.buffer = blasStorage_.buffer;
        blasCreate.size = blasSizes.accelerationStructureSize;
        blasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        check(createAccelerationStructure_(device_, &blasCreate, nullptr, &blas_), "vkCreateAccelerationStructureKHR(BLAS)");
        GpuBuffer scratch = createDeviceBuffer(blasSizes.buildScratchSize,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        blasBuild.dstAccelerationStructure = blas_;
        blasBuild.scratchData.deviceAddress = deviceAddress(scratch.buffer);
        VkAccelerationStructureBuildRangeInfoKHR blasRange{};
        blasRange.primitiveCount = primitiveCount;
        submitBuild(blasBuild, blasRange);
        destroyBuffer(scratch);

        VkAccelerationStructureDeviceAddressInfoKHR blasAddressInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        blasAddressInfo.accelerationStructure = blas_;
        VkAccelerationStructureInstanceKHR instance{};
        Mat4 transform = scene_->instances.empty() ? Mat4::identity() : scene_->instances.front().transform;
        instance.transform.matrix[0][0] = transform.m[0];
        instance.transform.matrix[0][1] = transform.m[4];
        instance.transform.matrix[0][2] = transform.m[8];
        instance.transform.matrix[0][3] = transform.m[12];
        instance.transform.matrix[1][0] = transform.m[1];
        instance.transform.matrix[1][1] = transform.m[5];
        instance.transform.matrix[1][2] = transform.m[9];
        instance.transform.matrix[1][3] = transform.m[13];
        instance.transform.matrix[2][0] = transform.m[2];
        instance.transform.matrix[2][1] = transform.m[6];
        instance.transform.matrix[2][2] = transform.m[10];
        instance.transform.matrix[2][3] = transform.m[14];
        instance.instanceCustomIndex = 0;
        instance.mask = 0xff;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = getAccelerationStructureDeviceAddress_(device_, &blasAddressInfo);
        instanceBuffer_ = createUploadBuffer(&instance, sizeof(instance),
                                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

        VkAccelerationStructureGeometryInstancesDataKHR instances{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        instances.data.deviceAddress = deviceAddress(instanceBuffer_.buffer);
        VkAccelerationStructureGeometryKHR tlasGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeometry.geometry.instances = instances;
        VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuild.geometryCount = 1;
        tlasBuild.pGeometries = &tlasGeometry;
        const std::uint32_t instanceCount = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getAccelerationStructureBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild,
                                            &instanceCount, &tlasSizes);
        tlasStorage_ = createDeviceBuffer(tlasSizes.accelerationStructureSize,
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR tlasCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        tlasCreate.buffer = tlasStorage_.buffer;
        tlasCreate.size = tlasSizes.accelerationStructureSize;
        tlasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        check(createAccelerationStructure_(device_, &tlasCreate, nullptr, &tlas_), "vkCreateAccelerationStructureKHR(TLAS)");
        scratch = createDeviceBuffer(tlasSizes.buildScratchSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        tlasBuild.dstAccelerationStructure = tlas_;
        tlasBuild.scratchData.deviceAddress = deviceAddress(scratch.buffer);
        VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
        tlasRange.primitiveCount = 1;
        submitBuild(tlasBuild, tlasRange);
        destroyBuffer(scratch);
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyAccelerationStructures();
        return false;
    }
}

void VulkanRenderer::destroyAccelerationStructures()
{
    if (!device_)
        return;
    if (tlas_ && destroyAccelerationStructure_)
        destroyAccelerationStructure_(device_, tlas_, nullptr);
    if (blas_ && destroyAccelerationStructure_)
        destroyAccelerationStructure_(device_, blas_, nullptr);
    tlas_ = VK_NULL_HANDLE;
    blas_ = VK_NULL_HANDLE;
    const auto destroyBuffer = [&](GpuBuffer& buffer) {
        if (buffer.buffer)
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory)
            vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    };
    destroyBuffer(instanceBuffer_);
    destroyBuffer(tlasStorage_);
    destroyBuffer(blasStorage_);
}

VulkanRenderer::GpuBuffer VulkanRenderer::createUploadBuffer(
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage) const
{
    if (!data || size == 0)
        throw std::runtime_error("Cannot create an empty GPU buffer");
    GpuBuffer result{};
    result.size = size;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
    VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateFlagsInfo allocationFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0)
    {
        allocationFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocationInfo.pNext = &allocationFlags;
    }
    check(vkAllocateMemory(device_, &allocationInfo, nullptr, &result.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory");
    void* mapped = nullptr;
    check(vkMapMemory(device_, result.memory, 0, size, 0, &mapped), "vkMapMemory");
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, result.memory);
    return result;
}

std::uint32_t VulkanRenderer::findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags properties) const
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);
    for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
    {
        if ((typeBits & (1u << index)) != 0 &&
            (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties)
            return index;
    }
    throw std::runtime_error("No compatible Vulkan memory type found");
}

void VulkanRenderer::recordCommands(VkCommandBuffer commandBuffer,
                                    std::uint32_t imageIndex,
                                    const GpuBuffer* externalBuffer)
{
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VkImageMemoryBarrier2 toTarget{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toTarget.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    toTarget.srcAccessMask = 0;
    toTarget.dstStageMask = externalBuffer ? VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                           : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toTarget.dstAccessMask = externalBuffer ? VK_ACCESS_2_TRANSFER_WRITE_BIT
                                            : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toTarget.oldLayout = swapchainImageInitialized_[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
    toTarget.newLayout = externalBuffer ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                        : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toTarget.image = swapchainImages_[imageIndex];
    toTarget.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTarget.subresourceRange.levelCount = 1;
    toTarget.subresourceRange.layerCount = 1;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &toTarget;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    if (externalBuffer)
    {
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {swapchainExtent_.width, swapchainExtent_.height, 1};
        vkCmdCopyBufferToImage(commandBuffer, externalBuffer->buffer, swapchainImages_[imageIndex],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        VkImageMemoryBarrier2 toAttachment{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment.image = swapchainImages_[imageIndex];
        toAttachment.subresourceRange = toTarget.subresourceRange;
        dependency.pImageMemoryBarriers = &toAttachment;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = swapchainImageViews_[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = externalBuffer ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.008f, 0.012f, 0.022f, 1.0f}};
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea.extent = swapchainExtent_;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchainExtent_.width);
    viewport.height = static_cast<float>(swapchainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    if (!externalBuffer && sceneDescriptorSet_ && uploadedMeshletCount_ > 0)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout_, 0, 1,
                                &sceneDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, meshPipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(FramePushConstants), &framePushConstants_);
        cmdDrawMeshTasks_(commandBuffer, uploadedMeshletCount_, 1, 1);
    }
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRendering(commandBuffer);

    VkImageMemoryBarrier2 toPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toPresent.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.image = swapchainImages_[imageIndex];
    toPresent.subresourceRange = toTarget.subresourceRange;
    dependency.pImageMemoryBarriers = &toPresent;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    swapchainImageInitialized_[imageIndex] = true;
}

VkShaderModule VulkanRenderer::loadShaderModule(const wchar_t* relativePath) const
{
    const std::filesystem::path path = executableDirectory() / relativePath;
    const std::vector<std::byte> code = readBinary(path);
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device_, &createInfo, nullptr, &module), "vkCreateShaderModule");
    return module;
}

void VulkanRenderer::setError(std::string message)
{
    unavailableReason_ = std::move(message);
    log(LogLevel::Error, unavailableReason_);
}
} // namespace vor
