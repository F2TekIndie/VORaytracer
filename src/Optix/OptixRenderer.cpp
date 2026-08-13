#include "Optix/OptixRenderer.h"

#include "Core/Log.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <Windows.h>

namespace vor
{
namespace
{
void checkCuda(CUresult result, const char* operation)
{
    if (result == CUDA_SUCCESS)
        return;
    const char* name = "unknown";
    const char* description = "unknown";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    throw std::runtime_error(std::string(operation) + " failed: " + name + " (" + description + ")");
}

void checkOptix(OptixResult result, const char* operation)
{
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with OptixResult " + std::to_string(result));
}

std::filesystem::path executableDirectory()
{
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open OptiX PTX: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
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
    if (std::abs(determinant) < 1.0e-8f)
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

Vec3 transformDirection(const Mat4& transform, Vec3 direction)
{
    return normalize({transform.m[0] * direction.x + transform.m[4] * direction.y + transform.m[8] * direction.z,
                      transform.m[1] * direction.x + transform.m[5] * direction.y + transform.m[9] * direction.z,
                      transform.m[2] * direction.x + transform.m[6] * direction.y + transform.m[10] * direction.z});
}

float transformHandedness(const Mat4& transform)
{
    const float determinant = transform.m[0] * (transform.m[5] * transform.m[10] - transform.m[9] * transform.m[6]) -
                              transform.m[4] * (transform.m[1] * transform.m[10] - transform.m[9] * transform.m[2]) +
                              transform.m[8] * (transform.m[1] * transform.m[6] - transform.m[5] * transform.m[2]);
    return determinant < 0.0f ? -1.0f : 1.0f;
}

struct SlangStructuredBuffer
{
    CUdeviceptr data;
    std::size_t count;
};

struct alignas(16) GpuEmissiveTriangle
{
    Vec4 position0;
    Vec4 position1;
    Vec4 position2;
    Vec4 emissionAndArea;
    Vec4 cdfAndPower;
    Vec4 uv0Uv1;
    Vec4 uv2AndPadding;
    std::array<std::uint32_t, 4> materialData;
};
static_assert(sizeof(GpuEmissiveTriangle) == 128);

struct alignas(16) GpuOptixInstance
{
    Mat4 transform;
    Mat4 normalTransform;
    std::uint32_t geometry[4];
    std::uint32_t materialIndex;
    float handedness;
    std::uint32_t padding[2];
};
static_assert(sizeof(GpuOptixInstance) == 160);

struct LaunchParameters
{
    SlangStructuredBuffer output;
    SlangStructuredBuffer denoisedOutput;
    SlangStructuredBuffer albedoGuide;
    SlangStructuredBuffer normalGuide;
    SlangStructuredBuffer displayOutput;
    SlangStructuredBuffer positions;
    SlangStructuredBuffer normals;
    SlangStructuredBuffer tangents;
    SlangStructuredBuffer uvs;
    SlangStructuredBuffer indices;
    SlangStructuredBuffer instances;
    SlangStructuredBuffer materials;
    SlangStructuredBuffer emissiveTriangles;
    SlangStructuredBuffer environmentPixels;
    SlangStructuredBuffer environmentConditionalCdf;
    SlangStructuredBuffer environmentMarginalCdf;
    CUdeviceptr materialTextureTable;
    OptixTraversableHandle scene;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t frameIndex;
    std::uint32_t maxBounces;
    std::uint32_t samplesPerFrame;
    std::uint32_t accumulatedSamples;
    std::uint32_t rayTracedShadows;
    std::uint32_t rayTracedReflections;
    std::uint32_t displayBgra;
    float exposure;
    std::uint32_t writeDisplay;
    std::uint32_t materialOverrideId;
    std::uint32_t debugView;
    std::uint32_t padding;
    std::uint32_t indirectLighting;
    std::uint32_t globalLightMode;
    std::uint32_t environmentWidth;
    std::uint32_t environmentHeight;
    float environmentIntensity;
    float environmentRotation;
    std::uint32_t environmentVisible;
    float environmentImportanceTotal;
    float emissiveLightPower;
    std::uint32_t importancePadding[2];
    Vec4 cameraPositionAndFov;
    Vec4 cameraTargetAndAspect;
    Vec4 cameraUp;
    Vec4 lightPosition;
    Vec4 lightColorAndIntensity;
    Vec4 lightDirection;
    Vec4 skyZenith;
    Vec4 skyHorizon;
    Vec4 skyGround;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header{};
    LaunchParameters parameters{};
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord
{
    std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header{};
};

static_assert(sizeof(LaunchParameters) % alignof(CUdeviceptr) == 0);
static_assert(sizeof(RaygenRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0);
static_assert(sizeof(EmptyRecord) == OPTIX_SBT_RECORD_HEADER_SIZE);
} // namespace

OptixRenderer::~OptixRenderer()
{
    shutdown();
}

bool OptixRenderer::initialize(GLFWwindow* window)
{
    try
    {
        checkCuda(cuInit(0), "cuInit");
        int deviceCount = 0;
        checkCuda(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
        if (deviceCount == 0)
            throw std::runtime_error("No CUDA device found");
        checkCuda(cuDeviceGet(&cudaDevice_, 0), "cuDeviceGet");
        checkCuda(cuCtxCreate(&cudaContext_, nullptr, 0, cudaDevice_), "cuCtxCreate");
        checkCuda(cuStreamCreate(&cudaStream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        checkCuda(cuEventCreate(&launchParameterCopyComplete_, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        checkCuda(cuEventCreate(&launchTimingStart_, CU_EVENT_DEFAULT), "cuEventCreate(launch timing start)");
        checkCuda(cuEventCreate(&launchTimingEnd_, CU_EVENT_DEFAULT), "cuEventCreate(launch timing end)");
        checkCuda(cuMemHostAlloc(&launchParametersHost_, sizeof(LaunchParameters), CU_MEMHOSTALLOC_PORTABLE),
                  "cuMemHostAlloc(launch parameters)");
        checkOptix(optixInit(), "optixInit");

        OptixDeviceContextOptions options{};
        options.logCallbackFunction = contextLog;
        options.logCallbackLevel = VOR_DEBUG ? 4 : 2;
        checkOptix(optixDeviceContextCreate(cudaContext_, &options, &optixContext_), "optixDeviceContextCreate");

        if (!createPipeline())
            return false;

        char deviceName[256]{};
        checkCuda(cuDeviceGetName(deviceName, sizeof(deviceName), cudaDevice_), "cuDeviceGetName");
        stats_.deviceName = deviceName;
        int framebufferWidth = 1;
        int framebufferHeight = 1;
        if (window)
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        width_ = static_cast<std::uint32_t>(std::max(framebufferWidth, 1));
        height_ = static_cast<std::uint32_t>(std::max(framebufferHeight, 1));
        if (!resizeOutput())
            return false;
        available_ = true;
        stats_.frameIndex = 0;
        stats_.accumulatedSamples = 0;
        stats_.tracedRays = 0;
        stats_.frameMilliseconds = 0.0f;
        log(LogLevel::Info, "OptiX 9.1 context initialized on " + stats_.deviceName);
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        shutdown();
        return false;
    }
}

void OptixRenderer::shutdown()
{
    if (cudaContext_)
        cuCtxSetCurrent(cudaContext_);
    if (cudaStream_)
        cuStreamSynchronize(cudaStream_);
    clearGpuInteropSurface();
    destroySceneAcceleration();
    destroyPipeline();
    destroyDenoiser();
    destroyOutputBuffers();
    if (optixContext_)
        optixDeviceContextDestroy(optixContext_);
    optixContext_ = nullptr;
    if (launchParametersHost_)
        cuMemFreeHost(launchParametersHost_);
    launchParametersHost_ = nullptr;
    if (launchParameterCopyComplete_)
        cuEventDestroy(launchParameterCopyComplete_);
    launchParameterCopyComplete_ = nullptr;
    if (launchTimingStart_)
        cuEventDestroy(launchTimingStart_);
    if (launchTimingEnd_)
        cuEventDestroy(launchTimingEnd_);
    launchTimingStart_ = nullptr;
    launchTimingEnd_ = nullptr;
    launchTimingPending_ = false;
    if (cudaStream_)
        cuStreamDestroy(cudaStream_);
    cudaStream_ = nullptr;
    if (cudaContext_)
        cuCtxDestroy(cudaContext_);
    cudaContext_ = nullptr;
    available_ = false;
}

void OptixRenderer::setScene(const Scene* scene)
{
    scene_ = scene;
    stats_.totalMeshlets = scene ? static_cast<std::uint32_t>(scene->meshletCount()) : 0;
    if (available_ && !buildSceneAcceleration())
        log(LogLevel::Error, "OptiX scene acceleration build failed: " + unavailableReason_);
    resetAccumulation();
}

bool OptixRenderer::updateMaterial(std::uint32_t materialIndex)
{
    if (!available_ || !scene_ || !materialBuffer_ || scene_->materials.size() != materialCount_ ||
        materialIndex >= scene_->materials.size())
        return false;
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent(material update)");
        const GpuMaterial material = scene_->materials[materialIndex].toGpu();
        checkCuda(cuMemcpyHtoDAsync(materialBuffer_ + materialIndex * sizeof(GpuMaterial), &material,
                                    sizeof(GpuMaterial), cudaStream_),
                  "cuMemcpyHtoDAsync(OptiX materials)");
        checkCuda(cuStreamSynchronize(cudaStream_), "cuStreamSynchronize(material update)");
        resetAccumulation();
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

void OptixRenderer::resize(std::uint32_t width, std::uint32_t height)
{
    const std::uint32_t newWidth = std::max(width, 1u);
    const std::uint32_t newHeight = std::max(height, 1u);
    if (width_ == newWidth && height_ == newHeight)
        return;
    width_ = newWidth;
    height_ = newHeight;
    if (available_)
        resizeOutput();
}

bool OptixRenderer::setGpuInteropSurface(const GpuInteropSurface& surface)
{
    if (!surface || !cudaContext_)
        return false;
    if (interopGeneration_ == surface.generation && interopOutputBuffer_)
        return true;
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent");
        clearGpuInteropSurface();

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memoryDesc{};
        memoryDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        memoryDesc.handle.win32.handle = surface.memoryHandle;
        memoryDesc.size = surface.allocationSize;
        checkCuda(cuImportExternalMemory(&externalMemory_, &memoryDesc), "cuImportExternalMemory(Vulkan)");
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc{};
        bufferDesc.size = surface.bufferByteSize;
        checkCuda(cuExternalMemoryGetMappedBuffer(&interopBaseBuffer_, externalMemory_, &bufferDesc),
                  "cuExternalMemoryGetMappedBuffer(Vulkan)");
        interopInputBuffer_ = interopBaseBuffer_ + surface.inputOffset;
        interopDenoisedBuffer_ = interopBaseBuffer_ + surface.denoisedOffset;
        interopOutputBuffer_ = interopBaseBuffer_ + surface.displayOffset;

        const auto importSemaphore = [](void* handle, CUexternalSemaphore& semaphore) {
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphoreDesc{};
            semaphoreDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
            semaphoreDesc.handle.win32.handle = handle;
            checkCuda(cuImportExternalSemaphore(&semaphore, &semaphoreDesc), "cuImportExternalSemaphore(Vulkan)");
        };
        importSemaphore(surface.cudaReadySemaphoreHandle, cudaReadySemaphore_);
        importSemaphore(surface.vulkanCompleteSemaphoreHandle, vulkanCompleteSemaphore_);
        interopGeneration_ = surface.generation;
        interopBgra_ = surface.bgra;
        firstInteropLaunch_ = true;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        clearGpuInteropSurface();
        return false;
    }
}

void OptixRenderer::clearGpuInteropSurface()
{
    if (cudaStream_)
        cuStreamSynchronize(cudaStream_);
    if (interopBaseBuffer_)
        cuMemFree(interopBaseBuffer_);
    interopBaseBuffer_ = 0;
    interopInputBuffer_ = 0;
    interopDenoisedBuffer_ = 0;
    interopOutputBuffer_ = 0;
    if (cudaReadySemaphore_)
        cuDestroyExternalSemaphore(cudaReadySemaphore_);
    if (vulkanCompleteSemaphore_)
        cuDestroyExternalSemaphore(vulkanCompleteSemaphore_);
    cudaReadySemaphore_ = nullptr;
    vulkanCompleteSemaphore_ = nullptr;
    if (externalMemory_)
        cuDestroyExternalMemory(externalMemory_);
    externalMemory_ = nullptr;
    interopGeneration_ = 0;
    firstInteropLaunch_ = true;
}

void OptixRenderer::resetAccumulation()
{
    stats_.accumulatedSamples = 0;
}

bool OptixRenderer::render(const Camera& camera, const RenderSettings& settings)
{
    if (!available_ || !gasHandle_ || !interopOutputBuffer_)
        return false;
    const auto begin = std::chrono::steady_clock::now();
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent");
        if (!updateShaderBindingTable(camera, settings))
            return false;
        if (launchTimingPending_ && cuEventQuery(launchTimingEnd_) == CUDA_SUCCESS)
        {
            float elapsedMilliseconds = 0.0f;
            checkCuda(cuEventElapsedTime(&elapsedMilliseconds, launchTimingStart_, launchTimingEnd_),
                      "cuEventElapsedTime(OptiX launch)");
            stats_.gpuMilliseconds = elapsedMilliseconds;
            launchTimingPending_ = false;
        }
        const bool recordLaunchTiming = !launchTimingPending_;
        OptixShaderBindingTable shaderBindingTable{};
        shaderBindingTable.raygenRecord = raygenRecord_;
        shaderBindingTable.missRecordBase = missRecord_;
        shaderBindingTable.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        shaderBindingTable.missRecordCount = 1;
        shaderBindingTable.hitgroupRecordBase = hitRecord_;
        shaderBindingTable.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
        shaderBindingTable.hitgroupRecordCount = 1;
        if (!firstInteropLaunch_)
        {
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams{};
            checkCuda(cuWaitExternalSemaphoresAsync(&vulkanCompleteSemaphore_, &waitParams, 1, cudaStream_),
                      "cuWaitExternalSemaphoresAsync(Vulkan complete)");
        }
        if (recordLaunchTiming)
            checkCuda(cuEventRecord(launchTimingStart_, cudaStream_), "cuEventRecord(OptiX launch start)");
        checkOptix(optixLaunch(pipeline_, cudaStream_, 0, 0, &shaderBindingTable, width_, height_, 1), "optixLaunch");
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams{};
        checkCuda(cuSignalExternalSemaphoresAsync(&cudaReadySemaphore_, &signalParams, 1, cudaStream_),
                  "cuSignalExternalSemaphoresAsync(CUDA ready)");
        if (recordLaunchTiming)
        {
            checkCuda(cuEventRecord(launchTimingEnd_, cudaStream_), "cuEventRecord(OptiX launch end)");
            launchTimingPending_ = true;
        }
        firstInteropLaunch_ = false;
        ++stats_.frameIndex;
        stats_.accumulatedSamples += settings.samplesPerFrame;
        const std::uint64_t pathDepth = settings.rayTracedReflections ? std::max(settings.maxBounces, 1u) : 1u;
        const std::uint64_t shadowFactor = settings.rayTracedShadows ? 2u : 1u;
        stats_.tracedRays = static_cast<std::uint64_t>(width_) * height_ * settings.samplesPerFrame * pathDepth *
                            shadowFactor;
        stats_.frameMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

bool OptixRenderer::denoiseVulkanFrame(float exposure)
{
    if (!available_ || !interopInputBuffer_ || !interopDenoisedBuffer_ || !interopOutputBuffer_)
        return false;
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent(Vulkan denoiser)");
        if (!ensureDenoiserResources())
            return false;
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams{};
        checkCuda(cuWaitExternalSemaphoresAsync(&vulkanCompleteSemaphore_, &waitParams, 1, cudaStream_),
                  "cuWaitExternalSemaphoresAsync(Vulkan denoiser input)");
        if (!invokeDenoiser())
            return false;
        if (launchParameterCopyPending_)
            checkCuda(cuEventSynchronize(launchParameterCopyComplete_),
                      "cuEventSynchronize(Vulkan tone map parameters)");
        auto& parameters = *static_cast<LaunchParameters*>(launchParametersHost_);
        parameters = {};
        parameters.denoisedOutput = {interopDenoisedBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.displayOutput = {interopOutputBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.width = width_;
        parameters.height = height_;
        parameters.displayBgra = interopBgra_ ? 1u : 0u;
        parameters.exposure = exposure;
        checkCuda(cuMemcpyHtoDAsync(toneMapRecord_ + offsetof(RaygenRecord, parameters), &parameters,
                                    sizeof(parameters), cudaStream_),
                  "cuMemcpyHtoDAsync(Vulkan tone map parameters)");
        OptixShaderBindingTable toneMapBindingTable{};
        toneMapBindingTable.raygenRecord = toneMapRecord_;
        checkOptix(optixLaunch(pipeline_, cudaStream_, 0, 0, &toneMapBindingTable, width_, height_, 1),
                   "optixLaunch(Vulkan post-denoiser tone map)");
        checkCuda(cuEventRecord(launchParameterCopyComplete_, cudaStream_),
                  "cuEventRecord(Vulkan tone map parameters)");
        launchParameterCopyPending_ = true;
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams{};
        checkCuda(cuSignalExternalSemaphoresAsync(&cudaReadySemaphore_, &signalParams, 1, cudaStream_),
                  "cuSignalExternalSemaphoresAsync(Vulkan denoised)");
        firstInteropLaunch_ = false;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

void OptixRenderer::contextLog(unsigned int level, const char* tag, const char* message, void*)
{
    const LogLevel logLevel = level <= 1 ? LogLevel::Error : level == 2 ? LogLevel::Warning : LogLevel::Info;
    log(logLevel, std::string("OptiX[") + (tag ? tag : "") + "] " + (message ? message : ""));
}

bool OptixRenderer::resizeOutput()
{
    try
    {
        if (cudaStream_)
            checkCuda(cuStreamSynchronize(cudaStream_), "cuStreamSynchronize(resize output)");
        destroyOutputBuffers();
        const std::size_t byteCount = static_cast<std::size_t>(width_) * height_ * 4 * sizeof(float);
        checkCuda(cuMemAlloc(&outputBuffer_, byteCount), "cuMemAlloc(beauty output)");
        resetAccumulation();
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

bool OptixRenderer::ensureDenoiserResources()
{
    if (denoiserState_ && denoiserScratch_)
        return true;
    try
    {
        if (!denoiser_ && !createDenoiser())
            return false;
        if ((denoiserState_ || denoiserScratch_) && cudaStream_)
            checkCuda(cuStreamSynchronize(cudaStream_), "cuStreamSynchronize(rebuild denoiser resources)");
        destroyDenoiserResources();
        if (!resizeDenoiser())
            throw std::runtime_error(unavailableReason_.empty() ? "OptiX denoiser setup failed" : unavailableReason_);
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyDenoiserResources();
        return false;
    }
}

bool OptixRenderer::createDenoiser()
{
    try
    {
        log(LogLevel::Info, "Creating OptiX AI denoiser");
        OptixDenoiserOptions options{};
        options.guideAlbedo = 0;
        options.guideNormal = 0;
        options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        checkOptix(optixDenoiserCreate(optixContext_, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &denoiser_),
                   "optixDenoiserCreate");
        log(LogLevel::Info, "OptiX AI denoiser created");
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyDenoiser();
        return false;
    }
}

bool OptixRenderer::resizeDenoiser()
{
    try
    {
        if (!denoiser_)
            return false;
        if (denoiserState_)
            checkCuda(cuMemFree(denoiserState_), "cuMemFree(denoiser state)");
        if (denoiserScratch_)
            checkCuda(cuMemFree(denoiserScratch_), "cuMemFree(denoiser scratch)");
        denoiserState_ = 0;
        denoiserScratch_ = 0;
        denoiserStateSize_ = 0;
        denoiserScratchSize_ = 0;

        OptixDenoiserSizes sizes{};
        checkOptix(optixDenoiserComputeMemoryResources(denoiser_, width_, height_, &sizes),
                   "optixDenoiserComputeMemoryResources");
        denoiserStateSize_ = sizes.stateSizeInBytes;
        denoiserScratchSize_ = sizes.withoutOverlapScratchSizeInBytes;
        log(LogLevel::Info, "OptiX denoiser setup " + std::to_string(width_) + "x" + std::to_string(height_) +
                                ": state " + std::to_string(denoiserStateSize_) + " bytes, scratch " +
                                std::to_string(denoiserScratchSize_) + " bytes");
        checkCuda(cuMemAlloc(&denoiserState_, denoiserStateSize_), "cuMemAlloc(denoiser state)");
        checkCuda(cuMemAlloc(&denoiserScratch_, denoiserScratchSize_), "cuMemAlloc(denoiser scratch)");
        checkOptix(optixDenoiserSetup(denoiser_, cudaStream_, width_, height_, denoiserState_, denoiserStateSize_,
                                      denoiserScratch_, denoiserScratchSize_),
                   "optixDenoiserSetup");
        log(LogLevel::Info, "OptiX denoiser setup complete");
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

bool OptixRenderer::invokeDenoiser()
{
    try
    {
        const auto image = [this](CUdeviceptr buffer) {
            constexpr unsigned int kHalf4PixelByteSize = sizeof(std::uint16_t) * 4;
            OptixImage2D result{};
            result.data = buffer;
            result.width = width_;
            result.height = height_;
            result.rowStrideInBytes = width_ * kHalf4PixelByteSize;
            result.pixelStrideInBytes = kHalf4PixelByteSize;
            result.format = OPTIX_PIXEL_FORMAT_HALF4;
            return result;
        };
        OptixDenoiserGuideLayer guides{};
        OptixDenoiserLayer layer{};
        layer.input = image(interopInputBuffer_);
        layer.output = image(interopDenoisedBuffer_);
        layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
        OptixDenoiserParams parameters{};
        parameters.blendFactor = 0.0f;
        checkOptix(optixDenoiserInvoke(denoiser_, cudaStream_, &parameters, denoiserState_, denoiserStateSize_,
                                       &guides, &layer, 1, 0, 0, denoiserScratch_, denoiserScratchSize_),
                   "optixDenoiserInvoke");
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

void OptixRenderer::destroyDenoiser()
{
    destroyDenoiserResources();
    if (denoiser_)
        optixDenoiserDestroy(denoiser_);
    denoiser_ = nullptr;
}

void OptixRenderer::destroyDenoiserResources()
{
    if (denoiserScratch_)
        cuMemFree(denoiserScratch_);
    if (denoiserState_)
        cuMemFree(denoiserState_);
    denoiserScratch_ = 0;
    denoiserState_ = 0;
    denoiserScratchSize_ = 0;
    denoiserStateSize_ = 0;
}

void OptixRenderer::destroyOutputBuffers()
{
    destroyDenoiserResources();
    if (outputBuffer_)
        cuMemFree(outputBuffer_);
    outputBuffer_ = 0;
}

bool OptixRenderer::uploadTextureResources()
{
    TextureReference fallback{};
    fallback.width = 1;
    fallback.height = 1;
    fallback.mipCount = 1;
    fallback.mipOffsets = {0};
    fallback.rgba8Pixels = {255, 255, 255, 255};
    std::vector<const TextureReference*> sources;
    if (scene_ && !scene_->textures.empty())
    {
        if (scene_->textures.size() > 1024)
            throw std::runtime_error("Scene exceeds the OptiX material texture limit");
        sources.reserve(scene_->textures.size());
        for (const TextureReference& texture : scene_->textures)
            sources.push_back(texture.valid() ? &texture : &fallback);
    }
    else
        sources.push_back(&fallback);

    textureArrays_.reserve(sources.size());
    textureObjects_.reserve(sources.size());
    for (const TextureReference* sourcePointer : sources)
    {
        const TextureReference& source = *sourcePointer;
        CUDA_ARRAY3D_DESCRIPTOR arrayDescription{};
        arrayDescription.Width = source.width;
        arrayDescription.Height = source.height;
        arrayDescription.Depth = 0;
        arrayDescription.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        arrayDescription.NumChannels = 4;
        CUmipmappedArray mipmappedArray{};
        checkCuda(cuMipmappedArrayCreate(&mipmappedArray, &arrayDescription, source.mipCount),
                  "cuMipmappedArrayCreate(material texture)");
        textureArrays_.push_back(mipmappedArray);
        std::uint32_t width = source.width;
        std::uint32_t height = source.height;
        for (std::uint32_t mip = 0; mip < source.mipCount; ++mip)
        {
            CUarray level{};
            checkCuda(cuMipmappedArrayGetLevel(&level, mipmappedArray, mip),
                      "cuMipmappedArrayGetLevel(material texture)");
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = source.rgba8Pixels.data() + source.mipOffsets[mip];
            copy.srcPitch = static_cast<std::size_t>(width) * 4;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = level;
            copy.WidthInBytes = static_cast<std::size_t>(width) * 4;
            copy.Height = height;
            checkCuda(cuMemcpy2D(&copy), "cuMemcpy2D(material texture)");
            width = std::max(width / 2, 1u);
            height = std::max(height / 2, 1u);
        }

        CUDA_RESOURCE_DESC resource{};
        resource.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        resource.res.mipmap.hMipmappedArray = mipmappedArray;
        CUDA_TEXTURE_DESC textureDescription{};
        textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
        textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
        textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
        textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR;
        textureDescription.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
        textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES | (source.srgb ? CU_TRSF_SRGB : 0u);
        textureDescription.maxMipmapLevelClamp = static_cast<float>(source.mipCount - 1);
        CUtexObject textureObject{};
        checkCuda(cuTexObjectCreate(&textureObject, &resource, &textureDescription, nullptr),
                  "cuTexObjectCreate(material texture)");
        textureObjects_.push_back(textureObject);
    }
    constexpr std::size_t textureTableEntries = 1024;
    std::vector<CUtexObject> textureTable(textureTableEntries + 1, 0);
    std::copy(textureObjects_.begin(), textureObjects_.end(), textureTable.begin());
    checkCuda(cuMemAlloc(&textureTableBuffer_, textureTable.size() * sizeof(CUtexObject)),
              "cuMemAlloc(material texture table)");
    checkCuda(cuMemcpyHtoD(textureTableBuffer_, textureTable.data(),
                           textureTable.size() * sizeof(CUtexObject)),
              "cuMemcpyHtoD(material texture table)");
    return true;
}

void OptixRenderer::destroyTextureResources()
{
    if (textureTableBuffer_)
        cuMemFree(textureTableBuffer_);
    textureTableBuffer_ = 0;
    for (CUtexObject texture : textureObjects_)
        cuTexObjectDestroy(texture);
    textureObjects_.clear();
    for (CUmipmappedArray array : textureArrays_)
        cuMipmappedArrayDestroy(array);
    textureArrays_.clear();
}

bool OptixRenderer::buildSceneAcceleration()
{
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent");
        destroySceneAcceleration();
        if (!scene_ || scene_->meshes.empty())
            throw std::runtime_error("Scene contains no geometry for OptiX");

        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<Vec4> tangents;
        std::vector<Vec2> uvs;
        std::vector<std::uint32_t> indices;
        struct MeshGeometry
        {
            std::uint32_t vertexOffset{};
            std::uint32_t vertexCount{};
            std::uint32_t triangleOffset{};
            std::uint32_t triangleCount{};
            std::uint32_t materialIndex{};
            bool valid{};
        };
        std::vector<MeshGeometry> meshGeometries(scene_->meshes.size());
        for (std::uint32_t meshIndex = 0; meshIndex < scene_->meshes.size(); ++meshIndex)
        {
            const Mesh& mesh = scene_->meshes[meshIndex];
            if (mesh.vertices.empty() || mesh.lods.empty() || mesh.lods.front().indices.empty())
                continue;
            MeshGeometry& geometry = meshGeometries[meshIndex];
            geometry.vertexOffset = static_cast<std::uint32_t>(positions.size());
            geometry.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
            geometry.triangleOffset = static_cast<std::uint32_t>(indices.size() / 3);
            geometry.triangleCount = static_cast<std::uint32_t>(mesh.lods.front().indices.size() / 3);
            geometry.materialIndex = mesh.materialIndex < scene_->materials.size() ? mesh.materialIndex : 0u;
            geometry.valid = true;
            for (const Vertex& vertex : mesh.vertices)
            {
                positions.push_back(vertex.position);
                normals.push_back(vertex.normal);
                tangents.push_back(vertex.tangent);
                uvs.push_back(vertex.uv);
            }
            for (std::uint32_t index : mesh.lods.front().indices)
                indices.push_back(index);
        }

        std::vector<GpuOptixInstance> gpuInstances;
        std::vector<std::uint32_t> instanceMeshIndices;
        const auto appendInstance = [&](std::uint32_t meshIndex, const Mat4& transform) {
            if (meshIndex >= meshGeometries.size() || !meshGeometries[meshIndex].valid)
                return;
            const MeshGeometry& geometry = meshGeometries[meshIndex];
            gpuInstances.push_back({transform, normalTransformMatrix(transform),
                                    {geometry.vertexOffset, geometry.triangleOffset, geometry.triangleCount, 0u},
                                    geometry.materialIndex, transformHandedness(transform), {}});
            instanceMeshIndices.push_back(meshIndex);
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
        if (positions.empty() || indices.size() < 3 || gpuInstances.empty())
            throw std::runtime_error("Scene contains no indexed triangles for OptiX");

        const std::size_t vertexBytes = positions.size() * sizeof(Vec3);
        const std::size_t normalBytes = normals.size() * sizeof(Vec3);
        const std::size_t tangentBytes = tangents.size() * sizeof(Vec4);
        const std::size_t uvBytes = uvs.size() * sizeof(Vec2);
        const std::size_t indexBytes = indices.size() * sizeof(std::uint32_t);
        const std::size_t instanceBytes = gpuInstances.size() * sizeof(GpuOptixInstance);
        std::vector<GpuMaterial> materials;
        materials.reserve(std::max<std::size_t>(scene_->materials.size(), 1));
        if (scene_->materials.empty())
            materials.push_back(Material{}.toGpu());
        else
        {
            for (const Material& material : scene_->materials)
            {
                materials.push_back(material.toGpu());
            }
        }
        const std::size_t materialBytes = materials.size() * sizeof(GpuMaterial);
        std::vector<GpuEmissiveTriangle> emissiveTriangles;
        float emissiveTotalPower = 0.0f;
        for (std::size_t instanceIndex = 0; instanceIndex < gpuInstances.size(); ++instanceIndex)
        {
            const GpuOptixInstance& instance = gpuInstances[instanceIndex];
            const std::uint32_t materialIndex = instance.materialIndex;
            if (materialIndex >= scene_->materials.size())
                continue;
            const Material& material = scene_->materials[materialIndex];
            const float luminance = material.emissive.x * 0.2126f + material.emissive.y * 0.7152f +
                                    material.emissive.z * 0.0722f;
            if (luminance <= 1.0e-6f)
                continue;
            const auto worldPoint = [&](Vec3 point) {
                return Vec3{instance.transform.m[0] * point.x + instance.transform.m[4] * point.y +
                                instance.transform.m[8] * point.z + instance.transform.m[12],
                            instance.transform.m[1] * point.x + instance.transform.m[5] * point.y +
                                instance.transform.m[9] * point.z + instance.transform.m[13],
                            instance.transform.m[2] * point.x + instance.transform.m[6] * point.y +
                                instance.transform.m[10] * point.z + instance.transform.m[14]};
            };
            for (std::uint32_t triangleIndex = 0; triangleIndex < instance.geometry[2]; ++triangleIndex)
            {
                const std::uint32_t triangleBase = (instance.geometry[1] + triangleIndex) * 3;
                const Vec3 position0 = worldPoint(positions[instance.geometry[0] + indices[triangleBase + 0]]);
                const Vec3 position1 = worldPoint(positions[instance.geometry[0] + indices[triangleBase + 1]]);
                const Vec3 position2 = worldPoint(positions[instance.geometry[0] + indices[triangleBase + 2]]);
                const Vec2 uv0 = uvs[instance.geometry[0] + indices[triangleBase + 0]];
                const Vec2 uv1 = uvs[instance.geometry[0] + indices[triangleBase + 1]];
                const Vec2 uv2 = uvs[instance.geometry[0] + indices[triangleBase + 2]];
                const float area = 0.5f * length(cross(position1 - position0, position2 - position0));
                const float power = luminance * area;
                if (power <= 1.0e-10f)
                    continue;
                const float previousPower = emissiveTotalPower;
                emissiveTotalPower += power;
                emissiveTriangles.push_back({{position0.x, position0.y, position0.z, 1.0f},
                                             {position1.x, position1.y, position1.z, 1.0f},
                                             {position2.x, position2.y, position2.z, 1.0f},
                                             {material.emissive.x, material.emissive.y, material.emissive.z, area},
                                             {previousPower, emissiveTotalPower, power, 0.0f},
                                             {uv0.x, uv0.y, uv1.x, uv1.y},
                                             {uv2.x, uv2.y, 0.0f, 0.0f},
                                             {material.emissiveTexture >= 0
                                                  ? static_cast<std::uint32_t>(material.emissiveTexture)
                                                  : kInvalidTextureId,
                                              material.doubleSided ? 1u : 0u, materialIndex, 0u}});
            }
        }
        if (emissiveTriangles.empty())
            emissiveTriangles.push_back({});
        else
        {
            for (GpuEmissiveTriangle& triangle : emissiveTriangles)
            {
                triangle.cdfAndPower.x /= emissiveTotalPower;
                triangle.cdfAndPower.y /= emissiveTotalPower;
            }
        }
        const std::size_t emissiveTriangleBytes = emissiveTriangles.size() * sizeof(GpuEmissiveTriangle);
        const Vec4 fallbackEnvironment{0.0f, 0.0f, 0.0f, 1.0f};
        const Vec4* environmentData = scene_->environment.hasHdr() ? scene_->environment.hdrPixels.data()
                                                                   : &fallbackEnvironment;
        environmentPixelCount_ = scene_->environment.hasHdr() ? scene_->environment.hdrPixels.size() : 1;
        const std::size_t environmentBytes = environmentPixelCount_ * sizeof(Vec4);
        constexpr std::array<float, 2> fallbackCdf{0.0f, 1.0f};
        const bool validImportance = scene_->environment.hasHdr() &&
                                     scene_->environment.hdrConditionalCdf.size() ==
                                         static_cast<std::size_t>(scene_->environment.hdrHeight) *
                                             (scene_->environment.hdrWidth + 1u) &&
                                     scene_->environment.hdrMarginalCdf.size() == scene_->environment.hdrHeight + 1u;
        const float* conditionalCdfData = validImportance ? scene_->environment.hdrConditionalCdf.data()
                                                          : fallbackCdf.data();
        const float* marginalCdfData = validImportance ? scene_->environment.hdrMarginalCdf.data()
                                                       : fallbackCdf.data();
        environmentConditionalCdfCount_ = validImportance ? scene_->environment.hdrConditionalCdf.size() : 2;
        environmentMarginalCdfCount_ = validImportance ? scene_->environment.hdrMarginalCdf.size() : 2;
        uploadTextureResources();
        checkCuda(cuMemAlloc(&vertexBuffer_, vertexBytes), "cuMemAlloc(OptiX vertices)");
        checkCuda(cuMemAlloc(&normalBuffer_, normalBytes), "cuMemAlloc(OptiX normals)");
        checkCuda(cuMemAlloc(&tangentBuffer_, tangentBytes), "cuMemAlloc(OptiX tangents)");
        checkCuda(cuMemAlloc(&uvBuffer_, uvBytes), "cuMemAlloc(OptiX UVs)");
        checkCuda(cuMemAlloc(&indexBuffer_, indexBytes), "cuMemAlloc(OptiX indices)");
        checkCuda(cuMemAlloc(&instanceBuffer_, instanceBytes), "cuMemAlloc(OptiX instances)");
        checkCuda(cuMemAlloc(&materialBuffer_, materialBytes), "cuMemAlloc(OptiX materials)");
        checkCuda(cuMemAlloc(&emissiveTriangleBuffer_, emissiveTriangleBytes),
                  "cuMemAlloc(OptiX emissive triangles)");
        checkCuda(cuMemAlloc(&environmentBuffer_, environmentBytes), "cuMemAlloc(OptiX HDR environment)");
        checkCuda(cuMemAlloc(&environmentConditionalCdfBuffer_, environmentConditionalCdfCount_ * sizeof(float)),
                  "cuMemAlloc(OptiX HDR conditional CDF)");
        checkCuda(cuMemAlloc(&environmentMarginalCdfBuffer_, environmentMarginalCdfCount_ * sizeof(float)),
                  "cuMemAlloc(OptiX HDR marginal CDF)");
        checkCuda(cuMemcpyHtoD(vertexBuffer_, positions.data(), vertexBytes), "cuMemcpyHtoD(OptiX vertices)");
        checkCuda(cuMemcpyHtoD(normalBuffer_, normals.data(), normalBytes), "cuMemcpyHtoD(OptiX normals)");
        checkCuda(cuMemcpyHtoD(tangentBuffer_, tangents.data(), tangentBytes), "cuMemcpyHtoD(OptiX tangents)");
        checkCuda(cuMemcpyHtoD(uvBuffer_, uvs.data(), uvBytes), "cuMemcpyHtoD(OptiX UVs)");
        checkCuda(cuMemcpyHtoD(indexBuffer_, indices.data(), indexBytes), "cuMemcpyHtoD(OptiX indices)");
        checkCuda(cuMemcpyHtoD(instanceBuffer_, gpuInstances.data(), instanceBytes),
                  "cuMemcpyHtoD(OptiX instances)");
        checkCuda(cuMemcpyHtoD(materialBuffer_, materials.data(), materialBytes), "cuMemcpyHtoD(OptiX materials)");
        checkCuda(cuMemcpyHtoD(emissiveTriangleBuffer_, emissiveTriangles.data(), emissiveTriangleBytes),
                  "cuMemcpyHtoD(OptiX emissive triangles)");
        checkCuda(cuMemcpyHtoD(environmentBuffer_, environmentData, environmentBytes),
                  "cuMemcpyHtoD(OptiX HDR environment)");
        checkCuda(cuMemcpyHtoD(environmentConditionalCdfBuffer_, conditionalCdfData,
                               environmentConditionalCdfCount_ * sizeof(float)),
                  "cuMemcpyHtoD(OptiX HDR conditional CDF)");
        checkCuda(cuMemcpyHtoD(environmentMarginalCdfBuffer_, marginalCdfData,
                               environmentMarginalCdfCount_ * sizeof(float)),
                  "cuMemcpyHtoD(OptiX HDR marginal CDF)");
        vertexCount_ = positions.size();
        triangleCount_ = indices.size() / 3;
        materialCount_ = materials.size();
        instanceCount_ = gpuInstances.size();
        emissiveTriangleCount_ = emissiveTotalPower > 0.0f ? emissiveTriangles.size() : 0;
        emissiveLightPower_ = emissiveTotalPower;

        const std::uint32_t geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
        OptixAccelBuildOptions buildOptions{};
        buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        struct GasBuildRecord
        {
            std::uint32_t meshIndex{};
            CUdeviceptr vertexData{};
            OptixBuildInput input{};
            OptixAccelBufferSizes sizes{};
        };
        std::vector<GasBuildRecord> gasBuilds;
        gasBuilds.reserve(scene_->meshes.size());
        std::size_t maximumGasTemporarySize = 0;
        for (std::uint32_t meshIndex = 0; meshIndex < meshGeometries.size(); ++meshIndex)
        {
            const MeshGeometry& geometry = meshGeometries[meshIndex];
            if (!geometry.valid)
                continue;
            gasBuilds.push_back({});
            GasBuildRecord& record = gasBuilds.back();
            record.meshIndex = meshIndex;
            record.vertexData = vertexBuffer_ + static_cast<CUdeviceptr>(geometry.vertexOffset) * sizeof(Vec3);
            record.input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            record.input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            record.input.triangleArray.vertexStrideInBytes = sizeof(Vec3);
            record.input.triangleArray.numVertices = geometry.vertexCount;
            record.input.triangleArray.vertexBuffers = &record.vertexData;
            record.input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            record.input.triangleArray.indexStrideInBytes = 3 * sizeof(std::uint32_t);
            record.input.triangleArray.numIndexTriplets = geometry.triangleCount;
            record.input.triangleArray.indexBuffer = indexBuffer_ +
                static_cast<CUdeviceptr>(geometry.triangleOffset) * 3 * sizeof(std::uint32_t);
            record.input.triangleArray.flags = &geometryFlags;
            record.input.triangleArray.numSbtRecords = 1;
            checkOptix(optixAccelComputeMemoryUsage(optixContext_, &buildOptions, &record.input, 1,
                                                    &record.sizes),
                       "optixAccelComputeMemoryUsage(GAS)");
            maximumGasTemporarySize = std::max(maximumGasTemporarySize, record.sizes.tempSizeInBytes);
        }

        CUdeviceptr temporaryBuffer = 0;
        checkCuda(cuMemAlloc(&temporaryBuffer, maximumGasTemporarySize), "cuMemAlloc(OptiX GAS temporary)");
        std::vector<OptixTraversableHandle> meshHandles(scene_->meshes.size(), 0);
        std::size_t accelerationBytes = 0;
        for (GasBuildRecord& record : gasBuilds)
        {
            CUdeviceptr output = 0;
            checkCuda(cuMemAlloc(&output, record.sizes.outputSizeInBytes), "cuMemAlloc(OptiX mesh GAS)");
            meshGasBuffers_.push_back(output);
            OptixTraversableHandle handle = 0;
            checkOptix(optixAccelBuild(optixContext_, cudaStream_, &buildOptions, &record.input, 1,
                                       temporaryBuffer, record.sizes.tempSizeInBytes, output,
                                       record.sizes.outputSizeInBytes, &handle, nullptr, 0),
                       "optixAccelBuild(mesh GAS)");
            meshHandles[record.meshIndex] = handle;
            accelerationBytes += record.sizes.outputSizeInBytes;
        }
        cuMemFree(temporaryBuffer);
        temporaryBuffer = 0;

        std::vector<OptixInstance> optixInstances(gpuInstances.size());
        for (std::uint32_t instanceIndex = 0; instanceIndex < gpuInstances.size(); ++instanceIndex)
        {
            const Mat4& transform = gpuInstances[instanceIndex].transform;
            OptixInstance& instance = optixInstances[instanceIndex];
            instance.transform[0] = transform.m[0];
            instance.transform[1] = transform.m[4];
            instance.transform[2] = transform.m[8];
            instance.transform[3] = transform.m[12];
            instance.transform[4] = transform.m[1];
            instance.transform[5] = transform.m[5];
            instance.transform[6] = transform.m[9];
            instance.transform[7] = transform.m[13];
            instance.transform[8] = transform.m[2];
            instance.transform[9] = transform.m[6];
            instance.transform[10] = transform.m[10];
            instance.transform[11] = transform.m[14];
            instance.instanceId = instanceIndex;
            instance.sbtOffset = 0;
            instance.visibilityMask = 0xff;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = meshHandles[instanceMeshIndices[instanceIndex]];
        }
        const std::size_t iasInstanceBytes = optixInstances.size() * sizeof(OptixInstance);
        checkCuda(cuMemAlloc(&iasInstanceBuffer_, iasInstanceBytes), "cuMemAlloc(OptiX IAS instances)");
        checkCuda(cuMemcpyHtoDAsync(iasInstanceBuffer_, optixInstances.data(), iasInstanceBytes, cudaStream_),
                  "cuMemcpyHtoDAsync(OptiX IAS instances)");
        OptixBuildInput iasInput{};
        iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        iasInput.instanceArray.instances = iasInstanceBuffer_;
        iasInput.instanceArray.numInstances = static_cast<std::uint32_t>(optixInstances.size());
        OptixAccelBufferSizes iasSizes{};
        checkOptix(optixAccelComputeMemoryUsage(optixContext_, &buildOptions, &iasInput, 1, &iasSizes),
                   "optixAccelComputeMemoryUsage(IAS)");
        checkCuda(cuMemAlloc(&temporaryBuffer, iasSizes.tempSizeInBytes), "cuMemAlloc(OptiX IAS temporary)");
        checkCuda(cuMemAlloc(&gasBuffer_, iasSizes.outputSizeInBytes), "cuMemAlloc(OptiX IAS)");
        checkOptix(optixAccelBuild(optixContext_, cudaStream_, &buildOptions, &iasInput, 1,
                                   temporaryBuffer, iasSizes.tempSizeInBytes, gasBuffer_,
                                   iasSizes.outputSizeInBytes, &gasHandle_, nullptr, 0),
                   "optixAccelBuild(IAS)");
        checkCuda(cuStreamSynchronize(cudaStream_), "cuStreamSynchronize(OptiX GAS/IAS)");
        cuMemFree(temporaryBuffer);
        accelerationBytes += iasSizes.outputSizeInBytes + iasInstanceBytes;
        stats_.materialBytes = materialBytes;
        stats_.residentMaterials = static_cast<std::uint32_t>(materials.size());
        stats_.residentTextures = static_cast<std::uint32_t>(textureObjects_.size());
        stats_.descriptorCapacity = 1024;
        stats_.textureBytes = 0;
        if (scene_ && !scene_->textures.empty())
        {
            for (const TextureReference& texture : scene_->textures)
                stats_.textureBytes += texture.rgba8Pixels.size();
        }
        else
            stats_.textureBytes = 4;
        stats_.gpuSceneBytes = vertexBytes + normalBytes + tangentBytes + uvBytes + indexBytes +
                               materialBytes + emissiveTriangleBytes + environmentBytes +
                               environmentConditionalCdfCount_ * sizeof(float) +
                               environmentMarginalCdfCount_ * sizeof(float) + instanceBytes + accelerationBytes +
                               stats_.textureBytes;
        log(LogLevel::Info, "OptiX scene GAS/IAS: " + std::to_string(positions.size()) + " unique vertices, " +
                                std::to_string(indices.size() / 3) + " unique triangles, " +
                                std::to_string(gasBuilds.size()) + " GAS, " +
                                std::to_string(gpuInstances.size()) + " IAS instances, " +
                                std::to_string(emissiveTriangleCount_) + " emissive lights");
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroySceneAcceleration();
        return false;
    }
}

void OptixRenderer::destroySceneAcceleration()
{
    destroyTextureResources();
    if (gasBuffer_)
        cuMemFree(gasBuffer_);
    for (CUdeviceptr meshGas : meshGasBuffers_)
        cuMemFree(meshGas);
    meshGasBuffers_.clear();
    if (iasInstanceBuffer_)
        cuMemFree(iasInstanceBuffer_);
    if (indexBuffer_)
        cuMemFree(indexBuffer_);
    if (materialBuffer_)
        cuMemFree(materialBuffer_);
    if (emissiveTriangleBuffer_)
        cuMemFree(emissiveTriangleBuffer_);
    if (environmentBuffer_)
        cuMemFree(environmentBuffer_);
    if (environmentConditionalCdfBuffer_)
        cuMemFree(environmentConditionalCdfBuffer_);
    if (environmentMarginalCdfBuffer_)
        cuMemFree(environmentMarginalCdfBuffer_);
    if (instanceBuffer_)
        cuMemFree(instanceBuffer_);
    if (normalBuffer_)
        cuMemFree(normalBuffer_);
    if (tangentBuffer_)
        cuMemFree(tangentBuffer_);
    if (uvBuffer_)
        cuMemFree(uvBuffer_);
    if (vertexBuffer_)
        cuMemFree(vertexBuffer_);
    gasBuffer_ = 0;
    iasInstanceBuffer_ = 0;
    indexBuffer_ = 0;
    materialBuffer_ = 0;
    emissiveTriangleBuffer_ = 0;
    environmentBuffer_ = 0;
    environmentConditionalCdfBuffer_ = 0;
    environmentMarginalCdfBuffer_ = 0;
    instanceBuffer_ = 0;
    normalBuffer_ = 0;
    tangentBuffer_ = 0;
    uvBuffer_ = 0;
    vertexBuffer_ = 0;
    vertexCount_ = 0;
    triangleCount_ = 0;
    materialCount_ = 0;
    instanceCount_ = 0;
    emissiveTriangleCount_ = 0;
    emissiveLightPower_ = 0.0f;
    environmentPixelCount_ = 0;
    environmentConditionalCdfCount_ = 0;
    environmentMarginalCdfCount_ = 0;
    gasHandle_ = 0;
    stats_.gpuSceneBytes = 0;
    stats_.textureBytes = 0;
    stats_.materialBytes = 0;
    stats_.residentTextures = 0;
    stats_.residentMaterials = 0;
}

bool OptixRenderer::createPipeline()
{
    try
    {
        const std::string ptx = readTextFile(executableDirectory() / L"Shaders\\PathTracer.ptx");
        OptixModuleCompileOptions moduleOptions{};
#if VOR_DEBUG
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        OptixPipelineCompileOptions pipelineOptions{};
        pipelineOptions.usesMotionBlur = false;
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        // Primitive, instance, barycentrics, distance and hit state lower to seven payload registers.
        pipelineOptions.numPayloadValues = 7;
        pipelineOptions.numAttributeValues = 2;
        pipelineOptions.exceptionFlags = VOR_DEBUG ? OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW : OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.pipelineLaunchParamsVariableName = nullptr;
        pipelineOptions.pipelineLaunchParamsSizeInBytes = 0;
        pipelineOptions.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        std::array<char, 4096> logBuffer{};
        std::size_t logSize = logBuffer.size();
        const OptixResult moduleResult = optixModuleCreate(optixContext_, &moduleOptions, &pipelineOptions,
                                                           ptx.data(), ptx.size(), logBuffer.data(), &logSize, &module_);
        if (logSize > 1)
            log(moduleResult == OPTIX_SUCCESS ? LogLevel::Info : LogLevel::Error,
                std::string_view(logBuffer.data(), std::min(logSize, logBuffer.size())));
        checkOptix(moduleResult, "optixModuleCreate");

        std::array<OptixProgramGroupDesc, 4> programGroupDescriptions{};
        programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDescriptions[0].raygen.module = module_;
        programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__RayGen";
        programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDescriptions[1].raygen.module = module_;
        programGroupDescriptions[1].raygen.entryFunctionName = "__raygen__ToneMap";
        programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDescriptions[2].miss.module = module_;
        programGroupDescriptions[2].miss.entryFunctionName = "__miss__Miss";
        programGroupDescriptions[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        programGroupDescriptions[3].hitgroup.moduleCH = module_;
        programGroupDescriptions[3].hitgroup.entryFunctionNameCH = "__closesthit__ClosestHit";
        OptixProgramGroupOptions programGroupOptions{};
        std::array<OptixProgramGroup, 4> programGroups{};
        logSize = logBuffer.size();
        const OptixResult programResult = optixProgramGroupCreate(optixContext_, programGroupDescriptions.data(),
                                                                  static_cast<unsigned int>(programGroupDescriptions.size()),
                                                                  &programGroupOptions, logBuffer.data(), &logSize,
                                                                  programGroups.data());
        if (logSize > 1)
            log(programResult == OPTIX_SUCCESS ? LogLevel::Info : LogLevel::Error,
                std::string_view(logBuffer.data(), std::min(logSize, logBuffer.size())));
        checkOptix(programResult, "optixProgramGroupCreate");
        raygenProgramGroup_ = programGroups[0];
        toneMapProgramGroup_ = programGroups[1];
        missProgramGroup_ = programGroups[2];
        hitProgramGroup_ = programGroups[3];

        OptixPipelineLinkOptions linkOptions{};
        linkOptions.maxTraceDepth = 1;
        logSize = logBuffer.size();
        const OptixResult pipelineResult = optixPipelineCreate(optixContext_, &pipelineOptions, &linkOptions,
                                                               programGroups.data(), static_cast<unsigned int>(programGroups.size()),
                                                               logBuffer.data(), &logSize,
                                                               &pipeline_);
        if (logSize > 1)
            log(pipelineResult == OPTIX_SUCCESS ? LogLevel::Info : LogLevel::Error,
                std::string_view(logBuffer.data(), std::min(logSize, logBuffer.size())));
        checkOptix(pipelineResult, "optixPipelineCreate");
        OptixStackSizes stackSizes{};
        for (OptixProgramGroup group : programGroups)
            checkOptix(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline_), "optixUtilAccumulateStackSizes");
        std::uint32_t directFromTraversal = 0;
        std::uint32_t directFromState = 0;
        std::uint32_t continuation = 0;
        checkOptix(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0, &directFromTraversal, &directFromState,
                                              &continuation),
                   "optixUtilComputeStackSizes");
        checkOptix(optixPipelineSetStackSize(pipeline_, directFromTraversal, directFromState, continuation, 2),
                   "optixPipelineSetStackSize");

        checkCuda(cuMemAlloc(&raygenRecord_, sizeof(RaygenRecord)), "cuMemAlloc(raygen SBT)");
        checkCuda(cuMemAlloc(&toneMapRecord_, sizeof(RaygenRecord)), "cuMemAlloc(tone map SBT)");
        checkCuda(cuMemAlloc(&missRecord_, OPTIX_SBT_RECORD_HEADER_SIZE), "cuMemAlloc(miss SBT)");
        checkCuda(cuMemAlloc(&hitRecord_, OPTIX_SBT_RECORD_HEADER_SIZE), "cuMemAlloc(hit SBT)");
        RaygenRecord raygenRecord{};
        RaygenRecord toneMapRecord{};
        EmptyRecord missRecord{};
        EmptyRecord hitRecord{};
        checkOptix(optixSbtRecordPackHeader(raygenProgramGroup_, &raygenRecord), "optixSbtRecordPackHeader(raygen)");
        checkOptix(optixSbtRecordPackHeader(toneMapProgramGroup_, &toneMapRecord),
                   "optixSbtRecordPackHeader(tone map)");
        checkOptix(optixSbtRecordPackHeader(missProgramGroup_, &missRecord), "optixSbtRecordPackHeader(miss)");
        checkOptix(optixSbtRecordPackHeader(hitProgramGroup_, &hitRecord), "optixSbtRecordPackHeader(hit)");
        checkCuda(cuMemcpyHtoD(raygenRecord_, &raygenRecord, sizeof(raygenRecord)), "cuMemcpyHtoD(raygen SBT)");
        checkCuda(cuMemcpyHtoD(toneMapRecord_, &toneMapRecord, sizeof(toneMapRecord)),
                  "cuMemcpyHtoD(tone map SBT)");
        checkCuda(cuMemcpyHtoD(missRecord_, &missRecord, sizeof(missRecord)), "cuMemcpyHtoD(miss SBT)");
        checkCuda(cuMemcpyHtoD(hitRecord_, &hitRecord, sizeof(hitRecord)), "cuMemcpyHtoD(hit SBT)");
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        destroyPipeline();
        return false;
    }
}

void OptixRenderer::destroyPipeline()
{
    if (toneMapRecord_)
        cuMemFree(toneMapRecord_);
    if (hitRecord_)
        cuMemFree(hitRecord_);
    if (missRecord_)
        cuMemFree(missRecord_);
    if (raygenRecord_)
        cuMemFree(raygenRecord_);
    hitRecord_ = 0;
    missRecord_ = 0;
    raygenRecord_ = 0;
    toneMapRecord_ = 0;
    if (pipeline_)
        optixPipelineDestroy(pipeline_);
    pipeline_ = nullptr;
    if (raygenProgramGroup_)
        optixProgramGroupDestroy(raygenProgramGroup_);
    if (toneMapProgramGroup_)
        optixProgramGroupDestroy(toneMapProgramGroup_);
    if (missProgramGroup_)
        optixProgramGroupDestroy(missProgramGroup_);
    if (hitProgramGroup_)
        optixProgramGroupDestroy(hitProgramGroup_);
    raygenProgramGroup_ = nullptr;
    toneMapProgramGroup_ = nullptr;
    missProgramGroup_ = nullptr;
    hitProgramGroup_ = nullptr;
    if (module_)
        optixModuleDestroy(module_);
    module_ = nullptr;
}

bool OptixRenderer::updateShaderBindingTable(const Camera& camera, const RenderSettings& settings)
{
    try
    {
        if (launchParameterCopyPending_)
            checkCuda(cuEventSynchronize(launchParameterCopyComplete_), "cuEventSynchronize(launch parameter copy)");
        auto& parameters = *static_cast<LaunchParameters*>(launchParametersHost_);
        parameters = {};
        parameters.output = {outputBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.denoisedOutput = {};
        parameters.albedoGuide = {};
        parameters.normalGuide = {};
        parameters.displayOutput = {interopOutputBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.positions = {vertexBuffer_, vertexCount_};
        parameters.normals = {normalBuffer_, vertexCount_};
        parameters.tangents = {tangentBuffer_, vertexCount_};
        parameters.uvs = {uvBuffer_, vertexCount_};
        parameters.indices = {indexBuffer_, triangleCount_};
        parameters.instances = {instanceBuffer_, instanceCount_};
        parameters.materials = {materialBuffer_, materialCount_};
        parameters.emissiveTriangles = {emissiveTriangleBuffer_, emissiveTriangleCount_};
        parameters.environmentPixels = {environmentBuffer_, environmentPixelCount_};
        parameters.environmentConditionalCdf = {environmentConditionalCdfBuffer_, environmentConditionalCdfCount_};
        parameters.environmentMarginalCdf = {environmentMarginalCdfBuffer_, environmentMarginalCdfCount_};
        parameters.materialTextureTable = textureTableBuffer_;
        parameters.scene = gasHandle_;
        parameters.width = width_;
        parameters.height = height_;
        parameters.frameIndex = static_cast<std::uint32_t>(stats_.frameIndex);
        parameters.maxBounces = settings.maxBounces;
        parameters.samplesPerFrame = settings.samplesPerFrame;
        parameters.accumulatedSamples = static_cast<std::uint32_t>(stats_.accumulatedSamples);
        parameters.rayTracedShadows = settings.rayTracedShadows ? 1u : 0u;
        parameters.rayTracedReflections = settings.rayTracedReflections ? 1u : 0u;
        parameters.displayBgra = interopBgra_ ? 1u : 0u;
        parameters.exposure = settings.exposure;
        parameters.writeDisplay = 1u;
        parameters.materialOverrideId = scene_ && scene_->materialOverrideId < scene_->materials.size()
                                            ? scene_->materialOverrideId
                                            : kInvalidMaterialId;
        parameters.debugView = static_cast<std::uint32_t>(settings.debugView);
        const Environment fallbackEnvironment{};
        const Environment& environment = scene_ ? scene_->environment : fallbackEnvironment;
        parameters.indirectLighting = settings.indirectLighting ? 1u : 0u;
        parameters.globalLightMode = static_cast<std::uint32_t>(environment.mode);
        parameters.environmentWidth = environment.hdrWidth;
        parameters.environmentHeight = environment.hdrHeight;
        parameters.environmentIntensity = environment.intensity;
        parameters.environmentRotation = environment.rotationRadians;
        parameters.environmentVisible = environment.visibleBackground ? 1u : 0u;
        parameters.environmentImportanceTotal = environment.hdrImportanceTotal;
        parameters.emissiveLightPower = parameters.materialOverrideId == kInvalidMaterialId
                                            ? emissiveLightPower_
                                            : 0.0f;
        parameters.cameraPositionAndFov = {camera.position.x, camera.position.y, camera.position.z,
                                           camera.verticalFovDegrees * kPi / 180.0f};
        parameters.cameraTargetAndAspect = {camera.target.x, camera.target.y, camera.target.z,
                                            static_cast<float>(width_) / static_cast<float>(std::max(height_, 1u))};
        parameters.cameraUp = {camera.up.x, camera.up.y, camera.up.z, 0.0f};
        const Light light = scene_ && !scene_->lights.empty() ? scene_->lights.front() : Light{};
        parameters.lightPosition = {light.position.x, light.position.y, light.position.z, 1.0f};
        parameters.lightColorAndIntensity = {light.color.x, light.color.y, light.color.z, light.intensity};
        parameters.lightDirection = {light.direction.x, light.direction.y, light.direction.z, 0.0f};
        parameters.skyZenith = {environment.zenithColor.x, environment.zenithColor.y,
                                environment.zenithColor.z, 0.0f};
        parameters.skyHorizon = {environment.horizonColor.x, environment.horizonColor.y,
                                 environment.horizonColor.z, 0.0f};
        parameters.skyGround = {environment.groundColor.x, environment.groundColor.y,
                                environment.groundColor.z, 0.0f};

        checkCuda(cuMemcpyHtoDAsync(raygenRecord_ + offsetof(RaygenRecord, parameters), &parameters,
                                    sizeof(parameters), cudaStream_),
                  "cuMemcpyHtoDAsync(launch parameters)");
        checkCuda(cuEventRecord(launchParameterCopyComplete_, cudaStream_), "cuEventRecord(launch parameter copy)");
        launchParameterCopyPending_ = true;
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
}

void OptixRenderer::setError(std::string message)
{
    unavailableReason_ = std::move(message);
    log(LogLevel::Error, unavailableReason_);
}
} // namespace vor
