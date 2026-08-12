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
#include <cmath>
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

Vec3 transformNormal(const Mat4& transform, Vec3 normal)
{
    const float a00 = transform.m[0], a01 = transform.m[4], a02 = transform.m[8];
    const float a10 = transform.m[1], a11 = transform.m[5], a12 = transform.m[9];
    const float a20 = transform.m[2], a21 = transform.m[6], a22 = transform.m[10];
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a12 * a20 - a10 * a22;
    const float c02 = a10 * a21 - a11 * a20;
    const float c10 = a02 * a21 - a01 * a22;
    const float c11 = a00 * a22 - a02 * a20;
    const float c12 = a01 * a20 - a00 * a21;
    const float c20 = a01 * a12 - a02 * a11;
    const float c21 = a02 * a10 - a00 * a12;
    const float c22 = a00 * a11 - a01 * a10;
    const float determinant = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::abs(determinant) < 1e-8f)
        return normalize(normal);
    return normalize({(c00 * normal.x + c01 * normal.y + c02 * normal.z) / determinant,
                      (c10 * normal.x + c11 * normal.y + c12 * normal.z) / determinant,
                      (c20 * normal.x + c21 * normal.y + c22 * normal.z) / determinant});
}

Mat4 normalTransformMatrix(const Mat4& transform)
{
    const float a00 = transform.m[0], a01 = transform.m[4], a02 = transform.m[8];
    const float a10 = transform.m[1], a11 = transform.m[5], a12 = transform.m[9];
    const float a20 = transform.m[2], a21 = transform.m[6], a22 = transform.m[10];
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a12 * a20 - a10 * a22;
    const float c02 = a10 * a21 - a11 * a20;
    const float c10 = a02 * a21 - a01 * a22;
    const float c11 = a00 * a22 - a02 * a20;
    const float c12 = a01 * a20 - a00 * a21;
    const float c20 = a01 * a12 - a02 * a11;
    const float c21 = a02 * a10 - a00 * a12;
    const float c22 = a00 * a11 - a01 * a10;
    const float determinant = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::abs(determinant) < 1e-8f)
        return Mat4::identity();
    const float inverseDeterminant = 1.0f / determinant;
    Mat4 result = Mat4::identity();
    result.m[0] = c00 * inverseDeterminant;
    result.m[4] = c01 * inverseDeterminant;
    result.m[8] = c02 * inverseDeterminant;
    result.m[1] = c10 * inverseDeterminant;
    result.m[5] = c11 * inverseDeterminant;
    result.m[9] = c12 * inverseDeterminant;
    result.m[2] = c20 * inverseDeterminant;
    result.m[6] = c21 * inverseDeterminant;
    result.m[10] = c22 * inverseDeterminant;
    return result;
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
        destroyGpuInteropSurface();
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
            if (frame.postInFlight)
                vkDestroyFence(device_, frame.postInFlight, nullptr);
            if (frame.inFlight)
                vkDestroyFence(device_, frame.inFlight, nullptr);
            if (frame.imageAvailable)
                vkDestroySemaphore(device_, frame.imageAvailable, nullptr);
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
    if (width == 0 || height == 0)
        return;
    if ((!resizePending_ && swapchainExtent_.width == width && swapchainExtent_.height == height) ||
        (resizePending_ && requestedWidth_ == width && requestedHeight_ == height))
        return;
    requestedWidth_ = width;
    requestedHeight_ = height;
    resizePending_ = true;
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

void VulkanRenderer::clearExternalImage()
{
    gpuInteropFrameReady_ = false;
}

bool VulkanRenderer::createGpuInteropSurface(std::uint32_t width, std::uint32_t height)
{
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    if (gpuInteropSurfaceInfo_ && gpuInteropSurfaceInfo_.width == width && gpuInteropSurfaceInfo_.height == height)
        return true;
    try
    {
        check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(interoperability resize)");
        destroyGpuInteropSurface();
        const VkDeviceSize pixelByteSize = static_cast<VkDeviceSize>(width) * height * sizeof(std::uint32_t);
        const VkDeviceSize floatImageByteSize = static_cast<VkDeviceSize>(width) * height * sizeof(Vec4);
        const VkDeviceSize bufferByteSize = floatImageByteSize * 2 + pixelByteSize;
        VkDeviceSize allocationSize = 0;
        HANDLE memoryHandle = nullptr;
        gpuInteropBuffer_ = createExternalBuffer(bufferByteSize, allocationSize, memoryHandle);
        if (!createDenoiserInputImage(width, height))
            throw std::runtime_error("Failed to create Vulkan denoiser input image");
        gpuInteropSurfaceInfo_.memoryHandle = memoryHandle;
        HANDLE cudaReadyHandle = nullptr;
        HANDLE vulkanCompleteHandle = nullptr;
        cudaReadySemaphore_ = createExternalSemaphore(cudaReadyHandle);
        gpuInteropSurfaceInfo_.cudaReadySemaphoreHandle = cudaReadyHandle;
        vulkanCompleteSemaphore_ = createExternalSemaphore(vulkanCompleteHandle);
        gpuInteropSurfaceInfo_.vulkanCompleteSemaphoreHandle = vulkanCompleteHandle;

        VkCommandBuffer ownershipCommand = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandInfo.commandPool = commandPool_;
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &commandInfo, &ownershipCommand),
              "vkAllocateCommandBuffers(CUDA ownership)");
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(ownershipCommand, &beginInfo), "vkBeginCommandBuffer(CUDA ownership)");
        VkBufferMemoryBarrier2 releaseBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        releaseBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        releaseBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        releaseBarrier.srcQueueFamilyIndex = graphicsQueueFamily_;
        releaseBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        releaseBarrier.buffer = gpuInteropBuffer_.buffer;
        releaseBarrier.size = gpuInteropBuffer_.size;
        VkDependencyInfo releaseDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        releaseDependency.bufferMemoryBarrierCount = 1;
        releaseDependency.pBufferMemoryBarriers = &releaseBarrier;
        vkCmdPipelineBarrier2(ownershipCommand, &releaseDependency);
        check(vkEndCommandBuffer(ownershipCommand), "vkEndCommandBuffer(CUDA ownership)");
        VkSubmitInfo ownershipSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        ownershipSubmit.commandBufferCount = 1;
        ownershipSubmit.pCommandBuffers = &ownershipCommand;
        check(vkQueueSubmit(graphicsQueue_, 1, &ownershipSubmit, VK_NULL_HANDLE), "vkQueueSubmit(CUDA ownership)");
        check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(CUDA ownership)");
        vkFreeCommandBuffers(device_, commandPool_, 1, &ownershipCommand);

        gpuInteropSurfaceInfo_.allocationSize = allocationSize;
        gpuInteropSurfaceInfo_.pixelByteSize = pixelByteSize;
        gpuInteropSurfaceInfo_.bufferByteSize = bufferByteSize;
        gpuInteropSurfaceInfo_.inputOffset = 0;
        gpuInteropSurfaceInfo_.denoisedOffset = floatImageByteSize;
        gpuInteropSurfaceInfo_.displayOffset = floatImageByteSize * 2;
        gpuInteropSurfaceInfo_.generation = ++gpuInteropGeneration_;
        gpuInteropSurfaceInfo_.width = width;
        gpuInteropSurfaceInfo_.height = height;
        gpuInteropSurfaceInfo_.bgra = swapchainFormat_ == VK_FORMAT_B8G8R8A8_SRGB ||
                                      swapchainFormat_ == VK_FORMAT_B8G8R8A8_UNORM;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyGpuInteropSurface();
        return false;
    }
}

void VulkanRenderer::destroyGpuInteropSurface()
{
    gpuInteropFrameReady_ = false;
    firstDenoiserInput_ = true;
    destroyDenoiserInputImage();
    if (device_)
    {
        if (cudaReadySemaphore_)
            vkDestroySemaphore(device_, cudaReadySemaphore_, nullptr);
        if (vulkanCompleteSemaphore_)
            vkDestroySemaphore(device_, vulkanCompleteSemaphore_, nullptr);
        if (gpuInteropBuffer_.buffer)
            vkDestroyBuffer(device_, gpuInteropBuffer_.buffer, nullptr);
        if (gpuInteropBuffer_.memory)
            vkFreeMemory(device_, gpuInteropBuffer_.memory, nullptr);
    }
    if (gpuInteropSurfaceInfo_.memoryHandle)
        CloseHandle(static_cast<HANDLE>(gpuInteropSurfaceInfo_.memoryHandle));
    if (gpuInteropSurfaceInfo_.cudaReadySemaphoreHandle)
        CloseHandle(static_cast<HANDLE>(gpuInteropSurfaceInfo_.cudaReadySemaphoreHandle));
    if (gpuInteropSurfaceInfo_.vulkanCompleteSemaphoreHandle)
        CloseHandle(static_cast<HANDLE>(gpuInteropSurfaceInfo_.vulkanCompleteSemaphoreHandle));
    cudaReadySemaphore_ = VK_NULL_HANDLE;
    vulkanCompleteSemaphore_ = VK_NULL_HANDLE;
    gpuInteropBuffer_ = {};
    gpuInteropSurfaceInfo_ = {};
}

bool VulkanRenderer::createDenoiserInputImage(std::uint32_t width, std::uint32_t height)
{
    try
    {
        destroyDenoiserInputImage();
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(device_, &imageInfo, nullptr, &denoiserInputImage_),
              "vkCreateImage(Vulkan denoiser input)");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, denoiserInputImage_, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &allocation, nullptr, &denoiserInputImageMemory_),
              "vkAllocateMemory(Vulkan denoiser input)");
        check(vkBindImageMemory(device_, denoiserInputImage_, denoiserInputImageMemory_, 0),
              "vkBindImageMemory(Vulkan denoiser input)");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = denoiserInputImage_;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        check(vkCreateImageView(device_, &viewInfo, nullptr, &denoiserInputImageView_),
              "vkCreateImageView(Vulkan denoiser input)");
        denoiserInputImageInitialized_ = false;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyDenoiserInputImage();
        return false;
    }
}

void VulkanRenderer::destroyDenoiserInputImage()
{
    if (!device_)
        return;
    if (denoiserInputImageView_)
        vkDestroyImageView(device_, denoiserInputImageView_, nullptr);
    if (denoiserInputImage_)
        vkDestroyImage(device_, denoiserInputImage_, nullptr);
    if (denoiserInputImageMemory_)
        vkFreeMemory(device_, denoiserInputImageMemory_, nullptr);
    denoiserInputImageView_ = VK_NULL_HANDLE;
    denoiserInputImage_ = VK_NULL_HANDLE;
    denoiserInputImageMemory_ = VK_NULL_HANDLE;
    denoiserInputImageInitialized_ = false;
}

void VulkanRenderer::updateFrameConstants(const Camera& camera, const RenderSettings& settings)
{
    const float aspect = static_cast<float>(swapchainExtent_.width) /
                         static_cast<float>(std::max(swapchainExtent_.height, 1u));
    framePushConstants_.viewProjection =
        perspective(camera.verticalFovDegrees * kPi / 180.0f, aspect, camera.nearPlane, camera.farPlane) *
        lookAt(camera.position, camera.target, camera.up);
    framePushConstants_.cameraPosition = {camera.position.x, camera.position.y, camera.position.z, 1.0f};
    const Vec3 cameraForward = normalize(camera.target - camera.position);
    const Vec3 cameraRight = normalize(cross(cameraForward, camera.up));
    const Vec3 cameraUp = normalize(cross(cameraRight, cameraForward));
    framePushConstants_.cameraForwardAndFov = {cameraForward.x, cameraForward.y, cameraForward.z,
                                               std::tan(camera.verticalFovDegrees * kPi / 360.0f)};
    framePushConstants_.cameraRightAndAspect = {cameraRight.x, cameraRight.y, cameraRight.z, aspect};
    framePushConstants_.cameraUp = {cameraUp.x, cameraUp.y, cameraUp.z, 0.0f};
    const Light light = scene_ && !scene_->lights.empty() ? scene_->lights.front() : Light{};
    const Environment fallbackEnvironment{};
    const Environment& environment = scene_ ? scene_->environment : fallbackEnvironment;
    framePushConstants_.lightColorAndIntensity = {light.color.x, light.color.y, light.color.z, light.intensity};
    framePushConstants_.lightDirectionAndMode = {light.direction.x, light.direction.y, light.direction.z,
                                                 static_cast<float>(environment.mode)};
    framePushConstants_.environmentParameters = {static_cast<float>(environment.hdrWidth),
                                                  static_cast<float>(environment.hdrHeight),
                                                  environment.intensity, environment.rotationRadians};
    framePushConstants_.skyZenithAndVisibility = {environment.zenithColor.x, environment.zenithColor.y,
                                                  environment.zenithColor.z,
                                                  environment.visibleBackground ? 1.0f : 0.0f};
    framePushConstants_.skyHorizonAndIndirect = {environment.horizonColor.x, environment.horizonColor.y,
                                                 environment.horizonColor.z,
                                                 settings.indirectLighting ? 1.0f : 0.0f};
    framePushConstants_.skyGround = {environment.groundColor.x, environment.groundColor.y,
                                     environment.groundColor.z, 0.0f};
    framePushConstants_.rayTracedShadows = settings.rayTracedShadows && tlas_ ? 1u : 0u;
    framePushConstants_.rayTracedReflections = settings.rayTracedReflections && tlas_ ? 1u : 0u;
    framePushConstants_.exposure = settings.exposure;
    framePushConstants_.meshletCount = uploadedMeshletCount_;
    framePushConstants_.instanceCount = static_cast<std::uint32_t>(uploadedInstances_.size());
    framePushConstants_.showMeshlets = settings.showMeshlets ? 1u : 0u;
    framePushConstants_.padding[0] = 0u;
}

bool VulkanRenderer::renderDenoiserInput(const Camera& camera, const RenderSettings& settings)
{
    if (!initialized_ || !gpuInteropSurfaceInfo_ || !denoiserInputImage_)
        return false;
    try
    {
        if (resizePending_ && !recreateSwapchain())
            return false;
        if (gpuInteropSurfaceInfo_.width != swapchainExtent_.width ||
            gpuInteropSurfaceInfo_.height != swapchainExtent_.height)
            return false;
        updateFrameConstants(camera, settings);
        framePushConstants_.padding[0] = 1u;
        FrameResources& frame = frames_[frameSlot_];
        check(vkWaitForFences(device_, 1, &frame.postInFlight, VK_TRUE, UINT64_MAX),
              "vkWaitForFences(Vulkan denoiser input)");
        check(vkResetFences(device_, 1, &frame.postInFlight), "vkResetFences(Vulkan denoiser input)");
        check(vkResetCommandBuffer(frame.postCommandBuffer, 0), "vkResetCommandBuffer(Vulkan denoiser input)");
        recordDenoiserInputCommands(frame.postCommandBuffer);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        if (!firstDenoiserInput_)
        {
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &vulkanCompleteSemaphore_;
            submit.pWaitDstStageMask = &waitStage;
        }
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.postCommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &vulkanCompleteSemaphore_;
        check(vkQueueSubmit(graphicsQueue_, 1, &submit, frame.postInFlight),
              "vkQueueSubmit(Vulkan denoiser input)");
        firstDenoiserInput_ = false;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

void VulkanRenderer::recordDenoiserInputCommands(VkCommandBuffer commandBuffer)
{
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer(Vulkan denoiser input)");

    VkBufferMemoryBarrier2 acquireBuffer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    acquireBuffer.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    acquireBuffer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    acquireBuffer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    acquireBuffer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    acquireBuffer.dstQueueFamilyIndex = graphicsQueueFamily_;
    acquireBuffer.buffer = gpuInteropBuffer_.buffer;
    acquireBuffer.size = gpuInteropBuffer_.size;
    VkImageMemoryBarrier2 colorBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    colorBarrier.srcStageMask = denoiserInputImageInitialized_ ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_NONE;
    colorBarrier.srcAccessMask = denoiserInputImageInitialized_ ? VK_ACCESS_2_TRANSFER_READ_BIT : 0;
    colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    colorBarrier.oldLayout = denoiserInputImageInitialized_ ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                            : VK_IMAGE_LAYOUT_UNDEFINED;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.image = denoiserInputImage_;
    colorBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBarrier.subresourceRange.levelCount = 1;
    colorBarrier.subresourceRange.layerCount = 1;
    VkImageMemoryBarrier2 depthBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    depthBarrier.srcStageMask = depthImageInitialized_ ? VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                                       : VK_PIPELINE_STAGE_2_NONE;
    depthBarrier.srcAccessMask = depthImageInitialized_ ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0;
    depthBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.oldLayout = depthImageInitialized_ ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                                    : VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthBarrier.image = depthImage_;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.layerCount = 1;
    const std::array imageBarriers{colorBarrier, depthBarrier};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &acquireBuffer;
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size());
    dependency.pImageMemoryBarriers = imageBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    depthImageInitialized_ = true;

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = denoiserInputImageView_;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.008f, 0.012f, 0.022f, 1.0f}};
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView_;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = swapchainExtent_;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &colorAttachment;
    rendering.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(commandBuffer, &rendering);
    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchainExtent_.width);
    viewport.height = static_cast<float>(swapchainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    if (sceneDescriptorSet_ && uploadedMeshletCount_ > 0)
    {
        constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT |
                                               VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, denoiserBackgroundPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout_, 0, 1,
                                &sceneDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, meshPipelineLayout_, stages, 0, sizeof(FramePushConstants),
                           &framePushConstants_);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, denoiserMeshPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout_, 0, 1,
                                &sceneDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, meshPipelineLayout_, stages, 0, sizeof(FramePushConstants),
                           &framePushConstants_);
        cmdDrawMeshTasks_(commandBuffer, uploadedMeshletCount_, 1, 1);
    }
    vkCmdEndRendering(commandBuffer);

    VkImageMemoryBarrier2 toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toTransfer.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.image = denoiserInputImage_;
    toTransfer.subresourceRange = colorBarrier.subresourceRange;
    dependency.bufferMemoryBarrierCount = 0;
    dependency.pBufferMemoryBarriers = nullptr;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &toTransfer;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    VkBufferImageCopy copy{};
    copy.bufferOffset = gpuInteropSurfaceInfo_.inputOffset;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {swapchainExtent_.width, swapchainExtent_.height, 1};
    vkCmdCopyImageToBuffer(commandBuffer, denoiserInputImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           gpuInteropBuffer_.buffer, 1, &copy);

    VkBufferMemoryBarrier2 releaseBuffer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    releaseBuffer.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    releaseBuffer.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    releaseBuffer.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    releaseBuffer.srcQueueFamilyIndex = graphicsQueueFamily_;
    releaseBuffer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    releaseBuffer.buffer = gpuInteropBuffer_.buffer;
    releaseBuffer.size = gpuInteropBuffer_.size;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &releaseBuffer;
    dependency.imageMemoryBarrierCount = 0;
    dependency.pImageMemoryBarriers = nullptr;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(Vulkan denoiser input)");
    denoiserInputImageInitialized_ = true;
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

        updateFrameConstants(camera, settings);

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

        const bool gpuInteropMatches = gpuInteropFrameReady_ && gpuInteropSurfaceInfo_ &&
                                       gpuInteropSurfaceInfo_.width == swapchainExtent_.width &&
                                       gpuInteropSurfaceInfo_.height == swapchainExtent_.height;
        const GpuBuffer* externalBuffer = nullptr;
        if (gpuInteropMatches)
            externalBuffer = &gpuInteropBuffer_;

        check(vkResetFences(device_, 1, &frame.inFlight), "vkResetFences");
        check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");
        recordCommands(frame.commandBuffer, imageIndex, externalBuffer);

        std::array<VkSemaphore, 2> waitSemaphores{frame.imageAvailable, cudaReadySemaphore_};
        std::array<VkPipelineStageFlags, 2> waitStages{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                       VK_PIPELINE_STAGE_TRANSFER_BIT};
        const VkSemaphore renderFinished = renderFinishedSemaphores_[imageIndex];
        std::array<VkSemaphore, 2> signalSemaphores{renderFinished, vulkanCompleteSemaphore_};
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = gpuInteropMatches ? 2u : 1u;
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &frame.commandBuffer;
        submitInfo.signalSemaphoreCount = gpuInteropMatches ? 2u : 1u;
        submitInfo.pSignalSemaphores = signalSemaphores.data();
        check(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, frame.inFlight), "vkQueueSubmit");
        if (gpuInteropMatches)
            firstDenoiserInput_ = false;
        gpuInteropFrameReady_ = false;

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished;
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
        stats_.accumulatedSamples = 0;
        stats_.visibleMeshlets = stats_.totalMeshlets;
        const std::uint64_t raysPerPixel = (settings.rayTracedShadows ? 1u : 0u) +
                                           (settings.rayTracedReflections ? 1u : 0u) +
                                           (settings.rayTracedShadows && settings.rayTracedReflections ? 1u : 0u) +
                                           (settings.indirectLighting ? 1u : 0u) +
                                           (settings.indirectLighting && settings.rayTracedShadows ? 1u : 0u);
        stats_.tracedRays = rayQueryAvailable_
                                ? static_cast<std::uint64_t>(swapchainExtent_.width) * swapchainExtent_.height *
                                      raysPerPixel
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
        if (!hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME) ||
            !hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME) ||
            !hasExtension(extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) ||
            !hasExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME))
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
        if (!meshFeatures.meshShader || !meshFeatures.taskShader)
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
        throw std::runtime_error("No Vulkan device with task/mesh shader and presentation support found");

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
    VkPhysicalDeviceVulkan11Features available11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures availableBufferAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

    VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    available.pNext = &available13;
    available13.pNext = &available11;
    available11.pNext = &availableBufferAddress;
    availableBufferAddress.pNext = &availableMesh;
    if (rayQueryExtensions)
    {
        availableMesh.pNext = &availableAcceleration;
        availableAcceleration.pNext = &availableRayQuery;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &available);

    if (!available13.dynamicRendering || !available13.synchronization2 || !availableMesh.meshShader ||
        !availableMesh.taskShader || !available11.shaderDrawParameters)
        throw std::runtime_error("Required Vulkan 1.3 dynamic rendering/synchronization or task/mesh shader features are missing");
    taskShaderAvailable_ = true;
    rayQueryAvailable_ = rayQueryExtensions && availableBufferAddress.bufferDeviceAddress &&
                         availableAcceleration.accelerationStructure && availableRayQuery.rayQuery;
    if (!rayQueryAvailable_)
        throw std::runtime_error("The Vulkan backend requires acceleration structures and VK_KHR_ray_query");

    std::vector<const char*> extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
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
    VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.pNext = &features11;
    features11.shaderDrawParameters = VK_TRUE;
    features11.pNext = &bufferDeviceAddress;
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
    renderFinishedSemaphores_.resize(imageCount, VK_NULL_HANDLE);
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
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[index]),
              "vkCreateSemaphore(render finished)");
    }

    constexpr std::array depthFormats{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    depthFormat_ = VK_FORMAT_UNDEFINED;
    for (const VkFormat format : depthFormats)
    {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
        {
            depthFormat_ = format;
            break;
        }
    }
    if (depthFormat_ == VK_FORMAT_UNDEFINED)
        throw std::runtime_error("No supported Vulkan depth format found");

    VkImageCreateInfo depthImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageInfo.format = depthFormat_;
    depthImageInfo.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
    depthImageInfo.mipLevels = 1;
    depthImageInfo.arrayLayers = 1;
    depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(vkCreateImage(device_, &depthImageInfo, nullptr, &depthImage_), "vkCreateImage(depth)");

    VkMemoryRequirements depthMemoryRequirements{};
    vkGetImageMemoryRequirements(device_, depthImage_, &depthMemoryRequirements);
    VkMemoryAllocateInfo depthAllocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    depthAllocationInfo.allocationSize = depthMemoryRequirements.size;
    depthAllocationInfo.memoryTypeIndex = findMemoryType(depthMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(device_, &depthAllocationInfo, nullptr, &depthImageMemory_), "vkAllocateMemory(depth)");
    check(vkBindImageMemory(device_, depthImage_, depthImageMemory_, 0), "vkBindImageMemory(depth)");

    VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthViewInfo.image = depthImage_;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format = depthFormat_;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.subresourceRange.levelCount = 1;
    depthViewInfo.subresourceRange.layerCount = 1;
    check(vkCreateImageView(device_, &depthViewInfo, nullptr, &depthImageView_), "vkCreateImageView(depth)");
    depthImageInitialized_ = false;
    return true;
}

bool VulkanRenderer::createCommands()
{
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamily_;
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");

    std::array<VkCommandBuffer, kFramesInFlight * 2> buffers{};
    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = static_cast<std::uint32_t>(buffers.size());
    check(vkAllocateCommandBuffers(device_, &allocateInfo, buffers.data()), "vkAllocateCommandBuffers");
    for (std::uint32_t index = 0; index < kFramesInFlight; ++index)
    {
        frames_[index].commandBuffer = buffers[index];
        frames_[index].postCommandBuffer = buffers[kFramesInFlight + index];
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frames_[index].imageAvailable), "vkCreateSemaphore");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(device_, &fenceInfo, nullptr, &frames_[index].inFlight), "vkCreateFence");
        check(vkCreateFence(device_, &fenceInfo, nullptr, &frames_[index].postInFlight), "vkCreateFence(post process)");
    }
    return true;
}

bool VulkanRenderer::createSceneDescriptors()
{
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (std::uint32_t index = 0; index < 4; ++index)
    {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    bindings[0].stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT |
                             VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &sceneDescriptorSetLayout_), "vkCreateDescriptorSetLayout");

    const std::array poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
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
    const VkShaderModule taskModule = loadShaderModule(L"Shaders\\Task.spv");
    const VkShaderModule meshModule = loadShaderModule(L"Shaders\\Mesh.spv");
    const VkShaderModule fragmentModule = loadShaderModule(L"Shaders\\PbrFragment.spv");
    const VkShaderModule backgroundVertexModule = loadShaderModule(L"Shaders\\BackgroundVertex.spv");
    const VkShaderModule backgroundFragmentModule = loadShaderModule(L"Shaders\\BackgroundFragment.spv");
    const std::array stages{
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, taskModule, "TaskMain", nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "MeshMain", nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "FragmentMain", nullptr},
    };

    try
    {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT |
                                       VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
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
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        const std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &swapchainFormat_;
        renderingInfo.depthAttachmentFormat = depthFormat_;
        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = meshPipelineLayout_;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshPipeline_), "vkCreateGraphicsPipelines");
        const VkFormat denoiserFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        renderingInfo.pColorAttachmentFormats = &denoiserFormat;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &denoiserMeshPipeline_),
              "vkCreateGraphicsPipelines(Vulkan denoiser mesh input)");

        const std::array backgroundStages{
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                             VK_SHADER_STAGE_VERTEX_BIT, backgroundVertexModule,
                                             "BackgroundVertex", nullptr},
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                             VK_SHADER_STAGE_FRAGMENT_BIT, backgroundFragmentModule,
                                             "BackgroundFragment", nullptr},
        };
        VkPipelineDepthStencilStateCreateInfo backgroundDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        backgroundDepth.depthTestEnable = VK_FALSE;
        backgroundDepth.depthWriteEnable = VK_FALSE;
        VkPipelineVertexInputStateCreateInfo backgroundVertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo backgroundInputAssembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        backgroundInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(backgroundStages.size());
        pipelineInfo.pStages = backgroundStages.data();
        pipelineInfo.pVertexInputState = &backgroundVertexInput;
        pipelineInfo.pInputAssemblyState = &backgroundInputAssembly;
        pipelineInfo.pDepthStencilState = &backgroundDepth;
        renderingInfo.pColorAttachmentFormats = &swapchainFormat_;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &backgroundPipeline_),
              "vkCreateGraphicsPipelines(background)");
        renderingInfo.pColorAttachmentFormats = &denoiserFormat;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                        &denoiserBackgroundPipeline_),
              "vkCreateGraphicsPipelines(Vulkan denoiser background input)");
    }
    catch (...)
    {
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, meshModule, nullptr);
        vkDestroyShaderModule(device_, taskModule, nullptr);
        vkDestroyShaderModule(device_, backgroundFragmentModule, nullptr);
        vkDestroyShaderModule(device_, backgroundVertexModule, nullptr);
        throw;
    }
    vkDestroyShaderModule(device_, fragmentModule, nullptr);
    vkDestroyShaderModule(device_, meshModule, nullptr);
    vkDestroyShaderModule(device_, taskModule, nullptr);
    vkDestroyShaderModule(device_, backgroundFragmentModule, nullptr);
    vkDestroyShaderModule(device_, backgroundVertexModule, nullptr);
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
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
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
    if (depthImageView_)
        vkDestroyImageView(device_, depthImageView_, nullptr);
    if (depthImage_)
        vkDestroyImage(device_, depthImage_, nullptr);
    if (depthImageMemory_)
        vkFreeMemory(device_, depthImageMemory_, nullptr);
    depthImageView_ = VK_NULL_HANDLE;
    depthImage_ = VK_NULL_HANDLE;
    depthImageMemory_ = VK_NULL_HANDLE;
    depthImageInitialized_ = false;
    for (VkImageView view : swapchainImageViews_)
        vkDestroyImageView(device_, view, nullptr);
    for (VkSemaphore semaphore : renderFinishedSemaphores_)
    {
        if (semaphore)
            vkDestroySemaphore(device_, semaphore, nullptr);
    }
    swapchainImageViews_.clear();
    renderFinishedSemaphores_.clear();
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
    if (backgroundPipeline_)
        vkDestroyPipeline(device_, backgroundPipeline_, nullptr);
    if (denoiserMeshPipeline_)
        vkDestroyPipeline(device_, denoiserMeshPipeline_, nullptr);
    if (denoiserBackgroundPipeline_)
        vkDestroyPipeline(device_, denoiserBackgroundPipeline_, nullptr);
    if (meshPipelineLayout_)
        vkDestroyPipelineLayout(device_, meshPipelineLayout_, nullptr);
    meshPipeline_ = VK_NULL_HANDLE;
    backgroundPipeline_ = VK_NULL_HANDLE;
    denoiserMeshPipeline_ = VK_NULL_HANDLE;
    denoiserBackgroundPipeline_ = VK_NULL_HANDLE;
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
    destroyBuffer(sceneInstanceBuffer_);
    destroyBuffer(environmentBuffer_);
    uploadedGeometries_.clear();
    uploadedInstances_.clear();
    uploadedMeshletCount_ = 0;
    uploadedVertexCount_ = 0;
    uploadedTriangleCount_ = 0;
    sceneDescriptorSet_ = VK_NULL_HANDLE;
    if (sceneDescriptorPool_)
        vkResetDescriptorPool(device_, sceneDescriptorPool_, 0);
}

bool VulkanRenderer::uploadSceneResources()
{
    if (!device_ || !sceneDescriptorPool_)
        return false;
    check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(scene replacement)");
    destroySceneResources();
    if (!scene_ || scene_->meshes.empty())
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
        std::uint32_t instanceIndex;
        std::uint32_t padding[3];
        Vec4 boundingSphere;
        Vec4 normalCone;
        Vec4 baseColorAndMetallic;
        Vec4 materialParameters;
    };
    struct alignas(16) GpuInstance
    {
        Mat4 transform;
        Mat4 normalTransform;
        Vec4 scaleAndFlags;
        std::uint32_t geometry[4];
        Vec4 baseColorAndMetallic;
        Vec4 emissiveAndRoughness;
    };
    static_assert(sizeof(GpuMeshlet) == 96);
    static_assert(sizeof(GpuInstance) == 192);
    struct RasterMeshRange
    {
        std::uint32_t firstMeshlet{};
        std::uint32_t meshletCount{};
    };

    std::vector<GpuVertex> vertices;
    std::vector<GpuMeshlet> meshletTemplates;
    std::vector<GpuMeshlet> meshlets;
    std::vector<GpuInstance> gpuInstances;
    std::vector<std::uint32_t> meshletVertices;
    std::vector<std::uint32_t> packedTriangles;
    std::vector<std::uint32_t> geometryIndices;
    const Vec4 fallbackEnvironment{0.0f, 0.0f, 0.0f, 1.0f};
    const std::span<const Vec4> environmentPixels = scene_->environment.hasHdr()
                                                        ? std::span<const Vec4>(scene_->environment.hdrPixels)
                                                        : std::span<const Vec4>(&fallbackEnvironment, 1);
    std::vector<std::uint32_t> meshToGeometry(scene_->meshes.size(), UINT32_MAX);
    std::vector<RasterMeshRange> rasterRanges(scene_->meshes.size());

    for (std::uint32_t meshIndex = 0; meshIndex < scene_->meshes.size(); ++meshIndex)
    {
        const Mesh& mesh = scene_->meshes[meshIndex];
        if (mesh.vertices.empty() || mesh.lods.empty() || mesh.lods.front().meshlets.empty())
            continue;
        const MeshLod& lod = mesh.lods.front();
        const std::uint32_t vertexBase = static_cast<std::uint32_t>(vertices.size());
        const std::uint32_t indexBase = static_cast<std::uint32_t>(geometryIndices.size());
        meshToGeometry[meshIndex] = static_cast<std::uint32_t>(uploadedGeometries_.size());
        uploadedGeometries_.push_back({vertexBase, static_cast<std::uint32_t>(mesh.vertices.size()), indexBase,
                                       static_cast<std::uint32_t>(lod.indices.size())});
        for (const Vertex& vertex : mesh.vertices)
        {
            vertices.push_back({{vertex.position.x, vertex.position.y, vertex.position.z, 1.0f},
                                {vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0f},
                                vertex.tangent,
                                {vertex.uv.x, vertex.uv.y, 0.0f, 0.0f}});
        }

        Material material{};
        if (mesh.materialIndex < scene_->materials.size())
            material = scene_->materials[mesh.materialIndex];
        RasterMeshRange& rasterRange = rasterRanges[meshIndex];
        rasterRange.firstMeshlet = static_cast<std::uint32_t>(meshletTemplates.size());
        for (const Meshlet& meshlet : lod.meshlets)
        {
            const std::uint32_t meshletVertexOffset = static_cast<std::uint32_t>(meshletVertices.size());
            for (std::uint32_t index = 0; index < meshlet.vertexCount; ++index)
                meshletVertices.push_back(vertexBase + lod.meshletVertices[meshlet.vertexOffset + index]);
            const std::uint32_t triangleOffset = static_cast<std::uint32_t>(packedTriangles.size());
            for (std::uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle)
            {
                const std::size_t byteOffset = static_cast<std::size_t>(meshlet.triangleOffset) + triangle * 3;
                const std::uint32_t packed = static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset]) |
                                             (static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset + 1]) << 8) |
                                             (static_cast<std::uint32_t>(lod.meshletTriangles[byteOffset + 2]) << 16);
                packedTriangles.push_back(packed);
            }
            meshletTemplates.push_back({meshletVertexOffset, triangleOffset, meshlet.vertexCount, meshlet.triangleCount,
                                        0, {}, meshlet.boundingSphere, meshlet.normalCone,
                                        {material.baseColor.x, material.baseColor.y, material.baseColor.z, material.metallic},
                                        {material.roughness, 0.0f, 0.0f, 0.0f}});
        }
        rasterRange.meshletCount = static_cast<std::uint32_t>(meshletTemplates.size()) - rasterRange.firstMeshlet;
        for (std::uint32_t index : lod.indices)
            geometryIndices.push_back(index);
    }

    const auto appendInstance = [&](std::uint32_t meshIndex, const Mat4& transform) {
        if (meshIndex >= meshToGeometry.size() || meshToGeometry[meshIndex] == UINT32_MAX)
            return;
        const float scaleX = length(Vec3{transform.m[0], transform.m[1], transform.m[2]});
        const float scaleY = length(Vec3{transform.m[4], transform.m[5], transform.m[6]});
        const float scaleZ = length(Vec3{transform.m[8], transform.m[9], transform.m[10]});
        const float maxScale = std::max({scaleX, scaleY, scaleZ});
        const float minScale = std::min({scaleX, scaleY, scaleZ});
        const std::uint32_t gpuInstanceIndex = static_cast<std::uint32_t>(gpuInstances.size());
        const UploadedGeometry& geometry = uploadedGeometries_[meshToGeometry[meshIndex]];
        Material material{};
        const Mesh& mesh = scene_->meshes[meshIndex];
        if (mesh.materialIndex < scene_->materials.size())
            material = scene_->materials[mesh.materialIndex];
        gpuInstances.push_back({transform,
                                normalTransformMatrix(transform),
                                {maxScale, minScale, 0.0f, 0.0f},
                                {geometry.vertexOffset, geometry.indexOffset, geometry.indexCount, 0},
                                {material.baseColor.x, material.baseColor.y, material.baseColor.z, material.metallic},
                                {material.emissive.x, material.emissive.y, material.emissive.z, material.roughness}});
        const RasterMeshRange range = rasterRanges[meshIndex];
        for (std::uint32_t index = 0; index < range.meshletCount; ++index)
        {
            GpuMeshlet meshlet = meshletTemplates[range.firstMeshlet + index];
            meshlet.instanceIndex = gpuInstanceIndex;
            meshlets.push_back(meshlet);
        }
        uploadedInstances_.push_back({meshToGeometry[meshIndex], transform});
    };

    if (!scene_->instances.empty())
    {
        for (const Instance& instance : scene_->instances)
            appendInstance(instance.meshIndex, instance.transform);
    }
    else
    {
        const Mat4 identity = Mat4::identity();
        for (std::uint32_t meshIndex = 0; meshIndex < scene_->meshes.size(); ++meshIndex)
            appendInstance(meshIndex, identity);
    }
    if (vertices.empty() || meshlets.empty() || geometryIndices.empty() || gpuInstances.empty())
        return true;

    vertexBuffer_ = createDeviceLocalBuffer(vertices.size() * sizeof(GpuVertex),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    meshletBuffer_ = createDeviceLocalBuffer(meshlets.size() * sizeof(GpuMeshlet), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshletVertexBuffer_ = createDeviceLocalBuffer(meshletVertices.size() * sizeof(std::uint32_t),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshletTriangleBuffer_ = createDeviceLocalBuffer(packedTriangles.size() * sizeof(std::uint32_t),
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    geometryIndexBuffer_ = createDeviceLocalBuffer(geometryIndices.size() * sizeof(std::uint32_t),
                                                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    sceneInstanceBuffer_ = createDeviceLocalBuffer(gpuInstances.size() * sizeof(GpuInstance),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    environmentBuffer_ = createDeviceLocalBuffer(environmentPixels.size_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const std::array uploads{
        BufferUpload{&vertexBuffer_, vertices.data(), vertices.size() * sizeof(GpuVertex)},
        BufferUpload{&meshletBuffer_, meshlets.data(), meshlets.size() * sizeof(GpuMeshlet)},
        BufferUpload{&meshletVertexBuffer_, meshletVertices.data(), meshletVertices.size() * sizeof(std::uint32_t)},
        BufferUpload{&meshletTriangleBuffer_, packedTriangles.data(), packedTriangles.size() * sizeof(std::uint32_t)},
        BufferUpload{&geometryIndexBuffer_, geometryIndices.data(), geometryIndices.size() * sizeof(std::uint32_t)},
        BufferUpload{&sceneInstanceBuffer_, gpuInstances.data(), gpuInstances.size() * sizeof(GpuInstance)},
        BufferUpload{&environmentBuffer_, environmentPixels.data(), environmentPixels.size_bytes()},
    };
    if (!uploadDeviceLocalBuffers(uploads))
        throw std::runtime_error("Failed to stage Vulkan scene geometry into device-local memory");
    uploadedVertexCount_ = static_cast<std::uint32_t>(vertices.size());
    uploadedTriangleCount_ = static_cast<std::uint32_t>(geometryIndices.size() / 3);
    if (!buildAccelerationStructures())
        throw std::runtime_error("Failed to build Vulkan BLAS/TLAS");

    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = sceneDescriptorPool_;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &sceneDescriptorSetLayout_;
    check(vkAllocateDescriptorSets(device_, &allocateInfo, &sceneDescriptorSet_), "vkAllocateDescriptorSets");
    const std::array<VkDescriptorBufferInfo, 7> bufferInfos{{
        {vertexBuffer_.buffer, 0, vertexBuffer_.size},
        {meshletBuffer_.buffer, 0, meshletBuffer_.size},
        {meshletVertexBuffer_.buffer, 0, meshletVertexBuffer_.size},
        {meshletTriangleBuffer_.buffer, 0, meshletTriangleBuffer_.size},
        {sceneInstanceBuffer_.buffer, 0, sceneInstanceBuffer_.size},
        {geometryIndexBuffer_.buffer, 0, geometryIndexBuffer_.size},
        {environmentBuffer_.buffer, 0, environmentBuffer_.size},
    }};
    std::array<VkWriteDescriptorSet, 8> writes{};
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
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = sceneDescriptorSet_;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = &bufferInfos[4];
    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = sceneDescriptorSet_;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &bufferInfos[5];
    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = sceneDescriptorSet_;
    writes[7].dstBinding = 7;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[7].pBufferInfo = &bufferInfos[6];
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    uploadedMeshletCount_ = static_cast<std::uint32_t>(meshlets.size());
    stats_.totalMeshlets = uploadedMeshletCount_;
    log(LogLevel::Info, "Vulkan scene upload: " + std::to_string(uploadedVertexCount_) + " unique vertices, " +
                            std::to_string(uploadedTriangleCount_) + " unique triangles, " +
                            std::to_string(uploadedGeometries_.size()) + " BLAS geometries, " +
                            std::to_string(uploadedInstances_.size()) + " TLAS instances, " +
                            std::to_string(uploadedMeshletCount_) + " rendered meshlets");
    return true;
}

bool VulkanRenderer::buildAccelerationStructures()
{
    try
    {
        destroyAccelerationStructures();
        if (uploadedGeometries_.empty() || uploadedInstances_.empty())
            return false;

        const auto deviceAddress = [&](VkBuffer buffer) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = buffer;
            return vkGetBufferDeviceAddress(device_, &info);
        };
        const auto destroyBuffer = [&](GpuBuffer& buffer) {
            if (buffer.buffer)
                vkDestroyBuffer(device_, buffer.buffer, nullptr);
            if (buffer.memory)
                vkFreeMemory(device_, buffer.memory, nullptr);
            buffer = {};
        };

        struct BlasBuildRecord
        {
            VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            VkAccelerationStructureBuildGeometryInfoKHR build{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            VkAccelerationStructureBuildRangeInfoKHR range{};
            VkDeviceSize scratchOffset{};
        };
        std::vector<BlasBuildRecord> blasBuilds;
        blasBuilds.reserve(uploadedGeometries_.size());
        blases_.reserve(uploadedGeometries_.size());
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 physicalProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        physicalProperties.pNext = &accelerationProperties;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &physicalProperties);
        const VkDeviceSize scratchAlignment = accelerationProperties.minAccelerationStructureScratchOffsetAlignment;
        VkDeviceSize blasScratchSize = 0;
        const VkDeviceAddress vertexAddress = deviceAddress(vertexBuffer_.buffer);
        const VkDeviceAddress indexAddress = deviceAddress(geometryIndexBuffer_.buffer);
        constexpr VkDeviceSize vertexStride = sizeof(Vec4) * 4;
        for (const UploadedGeometry& uploaded : uploadedGeometries_)
        {
            const std::uint32_t primitiveCount = uploaded.indexCount / 3;
            blasBuilds.emplace_back();
            BlasBuildRecord& record = blasBuilds.back();
            auto& triangles = record.geometry.geometry.triangles;
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = vertexAddress + static_cast<VkDeviceSize>(uploaded.vertexOffset) * vertexStride;
            triangles.vertexStride = vertexStride;
            triangles.maxVertex = uploaded.vertexCount - 1;
            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress = indexAddress + static_cast<VkDeviceSize>(uploaded.indexOffset) * sizeof(std::uint32_t);
            record.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            record.geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            record.build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            record.build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            record.build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            record.build.geometryCount = 1;
            record.build.pGeometries = &record.geometry;
            record.range.primitiveCount = primitiveCount;

            VkAccelerationStructureBuildSizesInfoKHR sizes{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            getAccelerationStructureBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &record.build, &primitiveCount, &sizes);
            BlasResource blas{};
            blas.storage = createDeviceLocalBuffer(sizes.accelerationStructureSize,
                                                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            createInfo.buffer = blas.storage.buffer;
            createInfo.size = sizes.accelerationStructureSize;
            createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            check(createAccelerationStructure_(device_, &createInfo, nullptr, &blas.handle),
                  "vkCreateAccelerationStructureKHR(BLAS)");
            record.build.dstAccelerationStructure = blas.handle;
            blasScratchSize = (blasScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);
            record.scratchOffset = blasScratchSize;
            blasScratchSize += sizes.buildScratchSize;
            blases_.push_back(blas);
        }

        std::vector<VkAccelerationStructureInstanceKHR> accelerationInstances;
        accelerationInstances.reserve(uploadedInstances_.size());
        for (std::uint32_t instanceIndex = 0; instanceIndex < uploadedInstances_.size(); ++instanceIndex)
        {
            const UploadedInstance& uploaded = uploadedInstances_[instanceIndex];
            if (uploaded.geometryIndex >= blases_.size())
                continue;
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            addressInfo.accelerationStructure = blases_[uploaded.geometryIndex].handle;
            VkAccelerationStructureInstanceKHR instance{};
            const Mat4& transform = uploaded.transform;
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
            instance.instanceCustomIndex = instanceIndex;
            instance.mask = 0xff;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = getAccelerationStructureDeviceAddress_(device_, &addressInfo);
            accelerationInstances.push_back(instance);
        }
        if (accelerationInstances.empty())
            throw std::runtime_error("No valid TLAS instances were generated");
        accelerationInstanceBuffer_ = createDeviceLocalBuffer(
            accelerationInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        const BufferUpload instanceUpload{&accelerationInstanceBuffer_, accelerationInstances.data(),
                                          accelerationInstances.size() * sizeof(VkAccelerationStructureInstanceKHR)};
        if (!uploadDeviceLocalBuffers(std::span{&instanceUpload, 1}))
            throw std::runtime_error("Failed to stage TLAS instances into device-local memory");

        VkAccelerationStructureGeometryInstancesDataKHR instances{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        instances.data.deviceAddress = deviceAddress(accelerationInstanceBuffer_.buffer);
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
        const std::uint32_t instanceCount = static_cast<std::uint32_t>(accelerationInstances.size());
        VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getAccelerationStructureBuildSizes_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuild,
                                            &instanceCount, &tlasSizes);
        tlasStorage_ = createDeviceLocalBuffer(tlasSizes.accelerationStructureSize,
                                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR tlasCreate{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        tlasCreate.buffer = tlasStorage_.buffer;
        tlasCreate.size = tlasSizes.accelerationStructureSize;
        tlasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        check(createAccelerationStructure_(device_, &tlasCreate, nullptr, &tlas_), "vkCreateAccelerationStructureKHR(TLAS)");
        const VkDeviceSize scratchSize = std::max(blasScratchSize, tlasSizes.buildScratchSize);
        GpuBuffer scratch = createDeviceLocalBuffer(scratchSize,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        const VkDeviceAddress scratchAddress = deviceAddress(scratch.buffer);
        for (BlasBuildRecord& record : blasBuilds)
            record.build.scratchData.deviceAddress = scratchAddress + record.scratchOffset;
        tlasBuild.dstAccelerationStructure = tlas_;
        tlasBuild.scratchData.deviceAddress = scratchAddress;
        VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
        tlasRange.primitiveCount = instanceCount;

        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = commandPool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(device_, &allocation, &command), "vkAllocateCommandBuffers(AS batch)");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(AS batch)");
        VkMemoryBarrier2 buildBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        buildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        buildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        buildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        buildBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        VkDependencyInfo buildDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        buildDependency.memoryBarrierCount = 1;
        buildDependency.pMemoryBarriers = &buildBarrier;
        for (BlasBuildRecord& record : blasBuilds)
        {
            const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&record.range};
            cmdBuildAccelerationStructures_(command, 1, &record.build, ranges);
        }
        vkCmdPipelineBarrier2(command, &buildDependency);
        const VkAccelerationStructureBuildRangeInfoKHR* tlasRanges[] = {&tlasRange};
        cmdBuildAccelerationStructures_(command, 1, &tlasBuild, tlasRanges);
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer(AS batch)");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence buildFence = VK_NULL_HANDLE;
        check(vkCreateFence(device_, &fenceInfo, nullptr, &buildFence), "vkCreateFence(AS batch)");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(graphicsQueue_, 1, &submit, buildFence), "vkQueueSubmit(AS batch)");
        check(vkWaitForFences(device_, 1, &buildFence, VK_TRUE, UINT64_MAX), "vkWaitForFences(AS batch)");
        vkDestroyFence(device_, buildFence, nullptr);
        vkFreeCommandBuffers(device_, commandPool_, 1, &command);
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
    tlas_ = VK_NULL_HANDLE;
    const auto destroyBuffer = [&](GpuBuffer& buffer) {
        if (buffer.buffer)
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory)
            vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    };
    for (BlasResource& blas : blases_)
    {
        if (blas.handle && destroyAccelerationStructure_)
            destroyAccelerationStructure_(device_, blas.handle, nullptr);
        destroyBuffer(blas.storage);
    }
    blases_.clear();
    destroyBuffer(accelerationInstanceBuffer_);
    destroyBuffer(tlasStorage_);
}

VulkanRenderer::GpuBuffer VulkanRenderer::createExternalBuffer(
    VkDeviceSize byteSize,
    VkDeviceSize& allocationSize,
    HANDLE& memoryHandle) const
{
    GpuBuffer result{};
    try
    {
        result.size = byteSize;
        VkExternalMemoryBufferCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.pNext = &externalInfo;
        bufferInfo.size = byteSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer(CUDA interop)");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        allocationSize = requirements.size;
        VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocationInfo.pNext = &exportInfo;
        allocationInfo.allocationSize = requirements.size;
        allocationInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &allocationInfo, nullptr, &result.memory), "vkAllocateMemory(CUDA interop)");
        check(vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory(CUDA interop)");

        const auto getMemoryHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
        if (!getMemoryHandle)
            throw std::runtime_error("vkGetMemoryWin32HandleKHR is unavailable");
        VkMemoryGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        handleInfo.memory = result.memory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        check(getMemoryHandle(device_, &handleInfo, &memoryHandle), "vkGetMemoryWin32HandleKHR");
        return result;
    }
    catch (...)
    {
        if (memoryHandle)
            CloseHandle(memoryHandle);
        memoryHandle = nullptr;
        if (result.memory)
            vkFreeMemory(device_, result.memory, nullptr);
        if (result.buffer)
            vkDestroyBuffer(device_, result.buffer, nullptr);
        throw;
    }
}

VkSemaphore VulkanRenderer::createExternalSemaphore(HANDLE& semaphoreHandle) const
{
    VkSemaphore semaphore = VK_NULL_HANDLE;
    try
    {
        VkExportSemaphoreCreateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        createInfo.pNext = &exportInfo;
        check(vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore), "vkCreateSemaphore(CUDA interop)");
        const auto getSemaphoreHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device_, "vkGetSemaphoreWin32HandleKHR"));
        if (!getSemaphoreHandle)
            throw std::runtime_error("vkGetSemaphoreWin32HandleKHR is unavailable");
        VkSemaphoreGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        handleInfo.semaphore = semaphore;
        handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        check(getSemaphoreHandle(device_, &handleInfo, &semaphoreHandle), "vkGetSemaphoreWin32HandleKHR");
        return semaphore;
    }
    catch (...)
    {
        if (semaphoreHandle)
            CloseHandle(semaphoreHandle);
        semaphoreHandle = nullptr;
        if (semaphore)
            vkDestroySemaphore(device_, semaphore, nullptr);
        throw;
    }
}

VulkanRenderer::GpuBuffer VulkanRenderer::createDeviceLocalBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage) const
{
    if (size == 0)
        throw std::runtime_error("Cannot create an empty device-local buffer");
    GpuBuffer result{};
    try
    {
        result.size = size;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer(device local)");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocationInfo.allocationSize = requirements.size;
        allocationInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateFlagsInfo allocationFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0)
        {
            allocationFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            allocationInfo.pNext = &allocationFlags;
        }
        check(vkAllocateMemory(device_, &allocationInfo, nullptr, &result.memory), "vkAllocateMemory(device local)");
        check(vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory(device local)");
        return result;
    }
    catch (...)
    {
        if (result.memory)
            vkFreeMemory(device_, result.memory, nullptr);
        if (result.buffer)
            vkDestroyBuffer(device_, result.buffer, nullptr);
        throw;
    }
}

bool VulkanRenderer::uploadDeviceLocalBuffers(std::span<const BufferUpload> uploads)
{
    if (uploads.empty())
        return true;
    GpuBuffer staging{};
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    try
    {
        std::vector<VkDeviceSize> offsets(uploads.size());
        VkDeviceSize stagingSize = 0;
        for (std::size_t index = 0; index < uploads.size(); ++index)
        {
            const BufferUpload& upload = uploads[index];
            if (!upload.destination || !upload.destination->buffer || !upload.data || upload.size == 0 ||
                upload.size > upload.destination->size)
                throw std::runtime_error("Invalid device-local buffer upload");
            stagingSize = (stagingSize + 15u) & ~VkDeviceSize{15u};
            offsets[index] = stagingSize;
            stagingSize += upload.size;
        }

        staging.size = stagingSize;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = stagingSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bufferInfo, nullptr, &staging.buffer), "vkCreateBuffer(staging batch)");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, staging.buffer, &requirements);
        VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocationInfo.allocationSize = requirements.size;
        allocationInfo.memoryTypeIndex = findMemoryType(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device_, &allocationInfo, nullptr, &staging.memory), "vkAllocateMemory(staging batch)");
        check(vkBindBufferMemory(device_, staging.buffer, staging.memory, 0), "vkBindBufferMemory(staging batch)");
        void* mapped = nullptr;
        check(vkMapMemory(device_, staging.memory, 0, stagingSize, 0, &mapped), "vkMapMemory(staging batch)");
        for (std::size_t index = 0; index < uploads.size(); ++index)
            std::memcpy(static_cast<std::byte*>(mapped) + offsets[index], uploads[index].data,
                        static_cast<std::size_t>(uploads[index].size));
        vkUnmapMemory(device_, staging.memory);

        VkCommandBufferAllocateInfo commandAllocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commandAllocation.commandPool = commandPool_;
        commandAllocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocation.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &commandAllocation, &command),
              "vkAllocateCommandBuffers(staging batch)");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(staging batch)");
        for (std::size_t index = 0; index < uploads.size(); ++index)
        {
            VkBufferCopy copy{offsets[index], 0, uploads[index].size};
            vkCmdCopyBuffer(command, staging.buffer, uploads[index].destination->buffer, 1, &copy);
        }
        std::vector<VkBufferMemoryBarrier2> barriers;
        barriers.reserve(uploads.size());
        for (const BufferUpload& upload : uploads)
        {
            VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            barrier.buffer = upload.destination->buffer;
            barrier.size = upload.size;
            barriers.push_back(barrier);
        }
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(command, &dependency);
        check(vkEndCommandBuffer(command), "vkEndCommandBuffer(staging batch)");

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence(staging batch)");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(graphicsQueue_, 1, &submit, fence), "vkQueueSubmit(staging batch)");
        check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences(staging batch)");
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, commandPool_, 1, &command);
        vkDestroyBuffer(device_, staging.buffer, nullptr);
        vkFreeMemory(device_, staging.memory, nullptr);
        return true;
    }
    catch (const std::exception& error)
    {
        if (fence)
            vkDestroyFence(device_, fence, nullptr);
        if (command)
            vkFreeCommandBuffers(device_, commandPool_, 1, &command);
        if (staging.buffer)
            vkDestroyBuffer(device_, staging.buffer, nullptr);
        if (staging.memory)
            vkFreeMemory(device_, staging.memory, nullptr);
        setError(error.what());
        return false;
    }
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
    VkImageMemoryBarrier2 depthBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    depthBarrier.srcStageMask = depthImageInitialized_
                                    ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                                    : VK_PIPELINE_STAGE_2_NONE;
    depthBarrier.srcAccessMask = depthImageInitialized_ ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0;
    depthBarrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthBarrier.oldLayout = depthImageInitialized_ ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthBarrier.image = depthImage_;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.layerCount = 1;
    const std::array targetBarriers{toTarget, depthBarrier};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(targetBarriers.size());
    dependency.pImageMemoryBarriers = targetBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    depthImageInitialized_ = true;

    if (externalBuffer)
    {
        VkBufferMemoryBarrier2 acquireInterop{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        acquireInterop.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acquireInterop.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        acquireInterop.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        acquireInterop.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        acquireInterop.dstQueueFamilyIndex = graphicsQueueFamily_;
        acquireInterop.buffer = externalBuffer->buffer;
        acquireInterop.size = externalBuffer->size;
        VkDependencyInfo acquireDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        acquireDependency.bufferMemoryBarrierCount = 1;
        acquireDependency.pBufferMemoryBarriers = &acquireInterop;
        vkCmdPipelineBarrier2(commandBuffer, &acquireDependency);

        VkBufferImageCopy copy{};
        copy.bufferOffset = gpuInteropSurfaceInfo_.displayOffset;
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
        VkBufferMemoryBarrier2 releaseInterop{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        releaseInterop.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        releaseInterop.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        releaseInterop.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        releaseInterop.srcQueueFamilyIndex = graphicsQueueFamily_;
        releaseInterop.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        releaseInterop.buffer = externalBuffer->buffer;
        releaseInterop.size = externalBuffer->size;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &toAttachment;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &releaseInterop;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
        dependency.bufferMemoryBarrierCount = 0;
        dependency.pBufferMemoryBarriers = nullptr;
    }

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = swapchainImageViews_[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = externalBuffer ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.008f, 0.012f, 0.022f, 1.0f}};
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImageView_;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea.extent = swapchainExtent_;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
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
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout_, 0, 1,
                                &sceneDescriptorSet_, 0, nullptr);
        constexpr VkShaderStageFlags allPushConstantStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT |
                                                              VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
        vkCmdPushConstants(commandBuffer, meshPipelineLayout_, allPushConstantStages,
                           0, sizeof(FramePushConstants), &framePushConstants_);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout_, 0, 1,
                                &sceneDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, meshPipelineLayout_, allPushConstantStages, 0,
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
    dependency.imageMemoryBarrierCount = 1;
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
