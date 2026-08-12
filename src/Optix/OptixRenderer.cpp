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

struct SlangStructuredBuffer
{
    CUdeviceptr data;
    std::size_t count;
};

struct LaunchParameters
{
    SlangStructuredBuffer output;
    SlangStructuredBuffer denoisedOutput;
    SlangStructuredBuffer albedoGuide;
    SlangStructuredBuffer normalGuide;
    SlangStructuredBuffer displayOutput;
    SlangStructuredBuffer normals;
    SlangStructuredBuffer indices;
    SlangStructuredBuffer triangleMaterialIndices;
    SlangStructuredBuffer materials;
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
    std::uint32_t padding[3];
    Vec4 cameraPositionAndFov;
    Vec4 cameraTargetAndAspect;
    Vec4 cameraUp;
    Vec4 lightPosition;
    Vec4 lightColorAndIntensity;
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

static_assert(sizeof(LaunchParameters) == 288);
static_assert(sizeof(RaygenRecord) == 320);
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
        checkCuda(cuMemHostAlloc(&launchParametersHost_, sizeof(LaunchParameters), CU_MEMHOSTALLOC_PORTABLE),
                  "cuMemHostAlloc(launch parameters)");
        checkOptix(optixInit(), "optixInit");

        OptixDeviceContextOptions options{};
        options.logCallbackFunction = contextLog;
        options.logCallbackLevel = VOR_DEBUG ? 4 : 2;
        checkOptix(optixDeviceContextCreate(cudaContext_, &options, &optixContext_), "optixDeviceContextCreate");

        if (!createPipeline() || !createDenoiser())
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
        bufferDesc.size = surface.pixelByteSize;
        checkCuda(cuExternalMemoryGetMappedBuffer(&interopOutputBuffer_, externalMemory_, &bufferDesc),
                  "cuExternalMemoryGetMappedBuffer(Vulkan)");

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
    if (interopOutputBuffer_)
        cuMemFree(interopOutputBuffer_);
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
        if (settings.denoiser && !ensureDenoiserResources())
            return false;
        if (!updateShaderBindingTable(camera, settings))
            return false;
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
        checkOptix(optixLaunch(pipeline_, cudaStream_, 0, 0, &shaderBindingTable, width_, height_, 1), "optixLaunch");
        if (settings.denoiser)
        {
            if (!invokeDenoiser())
                return false;
            OptixShaderBindingTable toneMapBindingTable{};
            toneMapBindingTable.raygenRecord = toneMapRecord_;
            checkOptix(optixLaunch(pipeline_, cudaStream_, 0, 0, &toneMapBindingTable, width_, height_, 1),
                       "optixLaunch(tone map)");
        }
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams{};
        checkCuda(cuSignalExternalSemaphoresAsync(&cudaReadySemaphore_, &signalParams, 1, cudaStream_),
                  "cuSignalExternalSemaphoresAsync(CUDA ready)");
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
    if (denoisedOutputBuffer_ && albedoGuideBuffer_ && normalGuideBuffer_ && denoiserState_ && denoiserScratch_)
        return true;
    try
    {
        if ((denoisedOutputBuffer_ || albedoGuideBuffer_ || normalGuideBuffer_ || denoiserState_ || denoiserScratch_) &&
            cudaStream_)
            checkCuda(cuStreamSynchronize(cudaStream_), "cuStreamSynchronize(rebuild denoiser resources)");
        destroyDenoiserResources();
        const std::size_t byteCount = static_cast<std::size_t>(width_) * height_ * sizeof(Vec4);
        checkCuda(cuMemAlloc(&denoisedOutputBuffer_, byteCount), "cuMemAlloc(denoised output)");
        checkCuda(cuMemAlloc(&albedoGuideBuffer_, byteCount), "cuMemAlloc(albedo guide)");
        checkCuda(cuMemAlloc(&normalGuideBuffer_, byteCount), "cuMemAlloc(normal guide)");
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
        options.guideAlbedo = 1;
        options.guideNormal = 1;
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
            OptixImage2D result{};
            result.data = buffer;
            result.width = width_;
            result.height = height_;
            result.rowStrideInBytes = width_ * static_cast<unsigned int>(sizeof(Vec4));
            result.pixelStrideInBytes = sizeof(Vec4);
            result.format = OPTIX_PIXEL_FORMAT_FLOAT4;
            return result;
        };
        OptixDenoiserGuideLayer guides{};
        guides.albedo = image(albedoGuideBuffer_);
        guides.normal = image(normalGuideBuffer_);
        OptixDenoiserLayer layer{};
        layer.input = image(outputBuffer_);
        layer.output = image(denoisedOutputBuffer_);
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
    if (normalGuideBuffer_)
        cuMemFree(normalGuideBuffer_);
    if (albedoGuideBuffer_)
        cuMemFree(albedoGuideBuffer_);
    if (denoisedOutputBuffer_)
        cuMemFree(denoisedOutputBuffer_);
    normalGuideBuffer_ = 0;
    albedoGuideBuffer_ = 0;
    denoisedOutputBuffer_ = 0;
}

void OptixRenderer::destroyOutputBuffers()
{
    destroyDenoiserResources();
    if (outputBuffer_)
        cuMemFree(outputBuffer_);
    outputBuffer_ = 0;
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
        std::vector<std::uint32_t> indices;
        std::vector<std::uint32_t> triangleMaterialIndices;
        const auto appendInstance = [&](const Mesh& mesh, const Mat4& transform) {
            if (mesh.vertices.empty() || mesh.lods.empty() || mesh.lods.front().indices.empty())
                return;
            const std::uint32_t vertexBase = static_cast<std::uint32_t>(positions.size());
            for (const Vertex& vertex : mesh.vertices)
            {
                const Vec3& p = vertex.position;
                positions.push_back({transform.m[0] * p.x + transform.m[4] * p.y + transform.m[8] * p.z + transform.m[12],
                                     transform.m[1] * p.x + transform.m[5] * p.y + transform.m[9] * p.z + transform.m[13],
                                     transform.m[2] * p.x + transform.m[6] * p.y + transform.m[10] * p.z + transform.m[14]});
                normals.push_back(transformNormal(transform, vertex.normal));
            }
            for (std::uint32_t index : mesh.lods.front().indices)
                indices.push_back(vertexBase + index);
            const std::uint32_t materialIndex = mesh.materialIndex < scene_->materials.size() ? mesh.materialIndex : 0u;
            triangleMaterialIndices.insert(triangleMaterialIndices.end(), mesh.lods.front().indices.size() / 3,
                                           materialIndex);
        };
        if (!scene_->instances.empty())
        {
            for (const Instance& instance : scene_->instances)
            {
                if (instance.meshIndex < scene_->meshes.size())
                    appendInstance(scene_->meshes[instance.meshIndex], instance.transform);
            }
        }
        else
        {
            const Mat4 identity = Mat4::identity();
            for (const Mesh& mesh : scene_->meshes)
                appendInstance(mesh, identity);
        }
        if (positions.empty() || indices.size() < 3)
            throw std::runtime_error("Scene contains no indexed triangles for OptiX");

        const std::size_t vertexBytes = positions.size() * sizeof(Vec3);
        const std::size_t normalBytes = normals.size() * sizeof(Vec3);
        const std::size_t indexBytes = indices.size() * sizeof(std::uint32_t);
        const std::size_t triangleMaterialBytes = triangleMaterialIndices.size() * sizeof(std::uint32_t);
        struct alignas(16) GpuMaterial
        {
            Vec4 baseColorAndMetallic;
            Vec4 emissiveAndRoughness;
        };
        std::vector<GpuMaterial> materials;
        materials.reserve(std::max<std::size_t>(scene_->materials.size(), 1));
        if (scene_->materials.empty())
            materials.push_back({{1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}});
        else
        {
            for (const Material& material : scene_->materials)
            {
                materials.push_back({{material.baseColor.x, material.baseColor.y, material.baseColor.z, material.metallic},
                                     {material.emissive.x, material.emissive.y, material.emissive.z, material.roughness}});
            }
        }
        const std::size_t materialBytes = materials.size() * sizeof(GpuMaterial);
        checkCuda(cuMemAlloc(&vertexBuffer_, vertexBytes), "cuMemAlloc(OptiX vertices)");
        checkCuda(cuMemAlloc(&normalBuffer_, normalBytes), "cuMemAlloc(OptiX normals)");
        checkCuda(cuMemAlloc(&indexBuffer_, indexBytes), "cuMemAlloc(OptiX indices)");
        checkCuda(cuMemAlloc(&triangleMaterialIndexBuffer_, triangleMaterialBytes),
                  "cuMemAlloc(OptiX triangle materials)");
        checkCuda(cuMemAlloc(&materialBuffer_, materialBytes), "cuMemAlloc(OptiX materials)");
        checkCuda(cuMemcpyHtoD(vertexBuffer_, positions.data(), vertexBytes), "cuMemcpyHtoD(OptiX vertices)");
        checkCuda(cuMemcpyHtoD(normalBuffer_, normals.data(), normalBytes), "cuMemcpyHtoD(OptiX normals)");
        checkCuda(cuMemcpyHtoD(indexBuffer_, indices.data(), indexBytes), "cuMemcpyHtoD(OptiX indices)");
        checkCuda(cuMemcpyHtoD(triangleMaterialIndexBuffer_, triangleMaterialIndices.data(), triangleMaterialBytes),
                  "cuMemcpyHtoD(OptiX triangle materials)");
        checkCuda(cuMemcpyHtoD(materialBuffer_, materials.data(), materialBytes), "cuMemcpyHtoD(OptiX materials)");
        vertexCount_ = positions.size();
        triangleCount_ = indices.size() / 3;
        materialCount_ = materials.size();

        const std::uint32_t geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;
        CUdeviceptr vertexBuffers[] = {vertexBuffer_};
        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vec3);
        buildInput.triangleArray.numVertices = static_cast<std::uint32_t>(positions.size());
        buildInput.triangleArray.vertexBuffers = vertexBuffers;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = 3 * sizeof(std::uint32_t);
        buildInput.triangleArray.numIndexTriplets = static_cast<std::uint32_t>(indices.size() / 3);
        buildInput.triangleArray.indexBuffer = indexBuffer_;
        buildInput.triangleArray.flags = &geometryFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions buildOptions{};
        buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        checkOptix(optixAccelComputeMemoryUsage(optixContext_, &buildOptions, &buildInput, 1, &sizes),
                   "optixAccelComputeMemoryUsage");

        CUdeviceptr temporaryBuffer = 0;
        checkCuda(cuMemAlloc(&temporaryBuffer, sizes.tempSizeInBytes), "cuMemAlloc(OptiX GAS temporary)");
        checkCuda(cuMemAlloc(&gasBuffer_, sizes.outputSizeInBytes), "cuMemAlloc(OptiX GAS)");
        const OptixResult buildResult = optixAccelBuild(optixContext_, nullptr, &buildOptions, &buildInput, 1,
                                                        temporaryBuffer, sizes.tempSizeInBytes, gasBuffer_,
                                                        sizes.outputSizeInBytes, &gasHandle_, nullptr, 0);
        cuMemFree(temporaryBuffer);
        checkOptix(buildResult, "optixAccelBuild");
        checkCuda(cuCtxSynchronize(), "cuCtxSynchronize(OptiX GAS)");
        log(LogLevel::Info, "OptiX scene GAS: " + std::to_string(positions.size()) + " vertices, " +
                                std::to_string(indices.size() / 3) + " triangles");
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
    if (gasBuffer_)
        cuMemFree(gasBuffer_);
    if (indexBuffer_)
        cuMemFree(indexBuffer_);
    if (materialBuffer_)
        cuMemFree(materialBuffer_);
    if (triangleMaterialIndexBuffer_)
        cuMemFree(triangleMaterialIndexBuffer_);
    if (normalBuffer_)
        cuMemFree(normalBuffer_);
    if (vertexBuffer_)
        cuMemFree(vertexBuffer_);
    gasBuffer_ = 0;
    indexBuffer_ = 0;
    materialBuffer_ = 0;
    triangleMaterialIndexBuffer_ = 0;
    normalBuffer_ = 0;
    vertexBuffer_ = 0;
    vertexCount_ = 0;
    triangleCount_ = 0;
    materialCount_ = 0;
    gasHandle_ = 0;
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
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        // Slang lowers the payload struct to six OptiX registers because float2 keeps its CUDA alignment.
        pipelineOptions.numPayloadValues = 6;
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
        checkOptix(optixPipelineSetStackSize(pipeline_, directFromTraversal, directFromState, continuation, 1),
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
        parameters.denoisedOutput = {denoisedOutputBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.albedoGuide = {albedoGuideBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.normalGuide = {normalGuideBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.displayOutput = {interopOutputBuffer_, static_cast<std::size_t>(width_) * height_};
        parameters.normals = {normalBuffer_, vertexCount_};
        parameters.indices = {indexBuffer_, triangleCount_};
        parameters.triangleMaterialIndices = {triangleMaterialIndexBuffer_, triangleCount_};
        parameters.materials = {materialBuffer_, materialCount_};
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
        parameters.writeDisplay = settings.denoiser ? 0u : 1u;
        parameters.cameraPositionAndFov = {camera.position.x, camera.position.y, camera.position.z,
                                           camera.verticalFovDegrees * kPi / 180.0f};
        parameters.cameraTargetAndAspect = {camera.target.x, camera.target.y, camera.target.z,
                                            static_cast<float>(width_) / static_cast<float>(std::max(height_, 1u))};
        parameters.cameraUp = {camera.up.x, camera.up.y, camera.up.z, 0.0f};
        const Light light = scene_ && !scene_->lights.empty() ? scene_->lights.front() : Light{};
        parameters.lightPosition = {light.position.x, light.position.y, light.position.z, 1.0f};
        parameters.lightColorAndIntensity = {light.color.x, light.color.y, light.color.z, light.intensity};

        checkCuda(cuMemcpyHtoDAsync(raygenRecord_ + offsetof(RaygenRecord, parameters), &parameters,
                                    sizeof(parameters), cudaStream_),
                  "cuMemcpyHtoDAsync(launch parameters)");
        if (settings.denoiser)
        {
            checkCuda(cuMemcpyHtoDAsync(toneMapRecord_ + offsetof(RaygenRecord, parameters), &parameters,
                                        sizeof(parameters), cudaStream_),
                      "cuMemcpyHtoDAsync(tone map parameters)");
        }
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
