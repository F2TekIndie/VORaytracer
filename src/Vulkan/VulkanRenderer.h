#pragma once

#include "Core/Renderer.h"

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vor
{
class VulkanRenderer final : public IRenderBackend
{
public:
    VulkanRenderer() = default;
    ~VulkanRenderer() override;

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    bool initialize(GLFWwindow* window) override;
    void shutdown() override;
    void setScene(const Scene* scene) override;
    void resize(std::uint32_t width, std::uint32_t height) override;
    void resetAccumulation() override;
    bool render(const Camera& camera, const RenderSettings& settings) override;
    [[nodiscard]] bool available() const override { return initialized_; }
    [[nodiscard]] const char* unavailableReason() const override { return unavailableReason_.c_str(); }
    [[nodiscard]] const RendererStats& stats() const override { return stats_; }

    void beginUiFrame();
    void clearExternalImage();
    bool createGpuInteropSurface(std::uint32_t width, std::uint32_t height);
    void destroyGpuInteropSurface();
    [[nodiscard]] const GpuInteropSurface& gpuInteropSurface() const { return gpuInteropSurfaceInfo_; }
    void setGpuInteropFrameReady(bool ready) { gpuInteropFrameReady_ = ready; }
    [[nodiscard]] bool rayQueryAvailable() const { return rayQueryAvailable_; }
    [[nodiscard]] bool taskShaderAvailable() const { return taskShaderAvailable_; }

private:
    static constexpr std::uint32_t kFramesInFlight = 2;

    struct FrameResources
    {
        VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
        VkSemaphore imageAvailable{VK_NULL_HANDLE};
        VkSemaphore renderFinished{VK_NULL_HANDLE};
        VkFence inFlight{VK_NULL_HANDLE};
    };

    struct GpuBuffer
    {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        VkDeviceSize size{};
    };

    struct FramePushConstants
    {
        Mat4 viewProjection{Mat4::identity()};
        Mat4 model{Mat4::identity()};
        Vec4 cameraPosition{};
        Vec4 lightPosition{};
        Vec4 lightColorAndIntensity{};
        std::uint32_t rayTracedShadows{};
        float exposure{};
        std::uint32_t meshletCount{};
        std::uint32_t padding{};
    };

    bool createInstance();
    bool createSurface();
    bool selectPhysicalDevice();
    bool createDevice();
    bool createSwapchain();
    bool createCommands();
    bool createSceneDescriptors();
    bool createMeshPipeline();
    bool initializeImGui();
    bool recreateSwapchain();

    void destroySwapchain();
    void destroyMeshPipeline();
    void destroySceneResources();
    bool uploadSceneResources();
    bool buildAccelerationStructures();
    void destroyAccelerationStructures();
    GpuBuffer createUploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) const;
    std::uint32_t findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags properties) const;
    void recordCommands(VkCommandBuffer commandBuffer, std::uint32_t imageIndex, const GpuBuffer* externalBuffer);
    GpuBuffer createExternalBuffer(VkDeviceSize byteSize, VkDeviceSize& allocationSize, HANDLE& memoryHandle) const;
    VkSemaphore createExternalSemaphore(HANDLE& semaphoreHandle) const;
    VkShaderModule loadShaderModule(const wchar_t* relativePath) const;
    void setError(std::string message);

    GLFWwindow* window_{};
    const Scene* scene_{};
    VkInstance instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugMessenger_{VK_NULL_HANDLE};
    VkSurfaceKHR surface_{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue graphicsQueue_{VK_NULL_HANDLE};
    std::uint32_t graphicsQueueFamily_{UINT32_MAX};

    VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
    VkFormat swapchainFormat_{VK_FORMAT_UNDEFINED};
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<bool> swapchainImageInitialized_;
    VkFormat depthFormat_{VK_FORMAT_UNDEFINED};
    VkImage depthImage_{VK_NULL_HANDLE};
    VkDeviceMemory depthImageMemory_{VK_NULL_HANDLE};
    VkImageView depthImageView_{VK_NULL_HANDLE};
    bool depthImageInitialized_{};

    VkCommandPool commandPool_{VK_NULL_HANDLE};
    std::array<FrameResources, kFramesInFlight> frames_{};
    std::uint32_t frameSlot_{};

    VkPipelineLayout meshPipelineLayout_{VK_NULL_HANDLE};
    VkPipeline meshPipeline_{VK_NULL_HANDLE};
    PFN_vkCmdDrawMeshTasksEXT cmdDrawMeshTasks_{};
    VkDescriptorSetLayout sceneDescriptorSetLayout_{VK_NULL_HANDLE};
    VkDescriptorPool sceneDescriptorPool_{VK_NULL_HANDLE};
    VkDescriptorSet sceneDescriptorSet_{VK_NULL_HANDLE};
    GpuBuffer vertexBuffer_{};
    GpuBuffer meshletBuffer_{};
    GpuBuffer meshletVertexBuffer_{};
    GpuBuffer meshletTriangleBuffer_{};
    GpuBuffer geometryIndexBuffer_{};
    GpuBuffer blasStorage_{};
    GpuBuffer tlasStorage_{};
    GpuBuffer instanceBuffer_{};
    VkAccelerationStructureKHR blas_{VK_NULL_HANDLE};
    VkAccelerationStructureKHR tlas_{VK_NULL_HANDLE};
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure_{};
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure_{};
    PFN_vkGetAccelerationStructureBuildSizesKHR getAccelerationStructureBuildSizes_{};
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures_{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress_{};
    std::uint32_t uploadedMeshletCount_{};
    std::uint32_t uploadedVertexCount_{};
    std::uint32_t uploadedTriangleCount_{};
    FramePushConstants framePushConstants_{};
    GpuBuffer gpuInteropBuffer_{};
    VkSemaphore cudaReadySemaphore_{VK_NULL_HANDLE};
    VkSemaphore vulkanCompleteSemaphore_{VK_NULL_HANDLE};
    GpuInteropSurface gpuInteropSurfaceInfo_{};
    bool gpuInteropFrameReady_{};
    std::uint64_t gpuInteropGeneration_{};

    bool initialized_{};
    bool imguiInitialized_{};
    bool resizePending_{};
    bool rayQueryAvailable_{};
    bool taskShaderAvailable_{};
    std::uint32_t requestedWidth_{};
    std::uint32_t requestedHeight_{};
    std::string unavailableReason_;
    RendererStats stats_{};
};
} // namespace vor
