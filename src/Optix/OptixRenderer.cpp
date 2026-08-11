#include "Optix/OptixRenderer.h"

#include "Core/Log.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

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
} // namespace

OptixRenderer::~OptixRenderer()
{
    shutdown();
}

bool OptixRenderer::initialize(GLFWwindow*)
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
    destroySceneAcceleration();
    destroyPipeline();
    if (outputBuffer_)
        cuMemFree(outputBuffer_);
    outputBuffer_ = 0;
    if (optixContext_)
        optixDeviceContextDestroy(optixContext_);
    optixContext_ = nullptr;
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
    width_ = std::max(width, 1u);
    height_ = std::max(height, 1u);
    if (available_)
        resizeOutput();
}

void OptixRenderer::resetAccumulation()
{
    stats_.accumulatedSamples = 0;
}

bool OptixRenderer::render(const Camera& camera, const RenderSettings& settings)
{
    if (!available_ || !gasHandle_)
        return false;
    const auto begin = std::chrono::steady_clock::now();
    try
    {
        checkCuda(cuCtxSetCurrent(cudaContext_), "cuCtxSetCurrent");
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
        checkOptix(optixLaunch(pipeline_, nullptr, 0, 0, &shaderBindingTable, width_, height_, 1), "optixLaunch");
        checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");
        checkCuda(cuMemcpyDtoH(hostOutput_.data(), outputBuffer_, hostOutput_.size() * sizeof(float)),
                  "cuMemcpyDtoH(OptiX output)");

        const float exposure = std::exp2(settings.exposure);
        const auto toSrgb8 = [exposure](float value) -> std::uint32_t {
            value = std::max(value * exposure, 0.0f);
            value = value / (1.0f + value);
            value = value <= 0.0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
            return static_cast<std::uint32_t>(std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
        };
        for (std::size_t pixel = 0; pixel < displayPixels_.size(); ++pixel)
        {
            const std::uint32_t r = toSrgb8(hostOutput_[pixel * 4 + 0]);
            const std::uint32_t g = toSrgb8(hostOutput_[pixel * 4 + 1]);
            const std::uint32_t b = toSrgb8(hostOutput_[pixel * 4 + 2]);
            displayPixels_[pixel] = r | (g << 8u) | (b << 16u) | 0xff000000u;
        }
        ++stats_.frameIndex;
        stats_.accumulatedSamples += settings.samplesPerFrame;
        stats_.tracedRays = static_cast<std::uint64_t>(width_) * height_ * settings.samplesPerFrame;
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
        if (outputBuffer_)
            checkCuda(cuMemFree(outputBuffer_), "cuMemFree");
        outputBuffer_ = 0;
        const std::size_t byteCount = static_cast<std::size_t>(width_) * height_ * 4 * sizeof(float);
        checkCuda(cuMemAlloc(&outputBuffer_, byteCount), "cuMemAlloc");
        hostOutput_.resize(static_cast<std::size_t>(width_) * height_ * 4);
        displayPixels_.resize(static_cast<std::size_t>(width_) * height_);
        resetAccumulation();
        return true;
    }
    catch (const std::exception& error)
    {
        setError(error.what());
        return false;
    }
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
        std::vector<std::uint32_t> indices;
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
            }
            for (std::uint32_t index : mesh.lods.front().indices)
                indices.push_back(vertexBase + index);
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
        const std::size_t indexBytes = indices.size() * sizeof(std::uint32_t);
        checkCuda(cuMemAlloc(&vertexBuffer_, vertexBytes), "cuMemAlloc(OptiX vertices)");
        checkCuda(cuMemAlloc(&indexBuffer_, indexBytes), "cuMemAlloc(OptiX indices)");
        checkCuda(cuMemcpyHtoD(vertexBuffer_, positions.data(), vertexBytes), "cuMemcpyHtoD(OptiX vertices)");
        checkCuda(cuMemcpyHtoD(indexBuffer_, indices.data(), indexBytes), "cuMemcpyHtoD(OptiX indices)");

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
    if (vertexBuffer_)
        cuMemFree(vertexBuffer_);
    gasBuffer_ = 0;
    indexBuffer_ = 0;
    vertexBuffer_ = 0;
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
        pipelineOptions.numPayloadValues = 8;
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

        std::array<OptixProgramGroupDesc, 3> programGroupDescriptions{};
        programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDescriptions[0].raygen.module = module_;
        programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__RayGen";
        programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDescriptions[1].miss.module = module_;
        programGroupDescriptions[1].miss.entryFunctionName = "__miss__Miss";
        programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        programGroupDescriptions[2].hitgroup.moduleCH = module_;
        programGroupDescriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__ClosestHit";
        OptixProgramGroupOptions programGroupOptions{};
        std::array<OptixProgramGroup, 3> programGroups{};
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
        missProgramGroup_ = programGroups[1];
        hitProgramGroup_ = programGroups[2];

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

        checkCuda(cuMemAlloc(&raygenRecord_, 160), "cuMemAlloc(raygen SBT)");
        checkCuda(cuMemAlloc(&missRecord_, OPTIX_SBT_RECORD_HEADER_SIZE), "cuMemAlloc(miss SBT)");
        checkCuda(cuMemAlloc(&hitRecord_, OPTIX_SBT_RECORD_HEADER_SIZE), "cuMemAlloc(hit SBT)");
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
    if (hitRecord_)
        cuMemFree(hitRecord_);
    if (missRecord_)
        cuMemFree(missRecord_);
    if (raygenRecord_)
        cuMemFree(raygenRecord_);
    hitRecord_ = 0;
    missRecord_ = 0;
    raygenRecord_ = 0;
    if (pipeline_)
        optixPipelineDestroy(pipeline_);
    pipeline_ = nullptr;
    if (raygenProgramGroup_)
        optixProgramGroupDestroy(raygenProgramGroup_);
    if (missProgramGroup_)
        optixProgramGroupDestroy(missProgramGroup_);
    if (hitProgramGroup_)
        optixProgramGroupDestroy(hitProgramGroup_);
    raygenProgramGroup_ = nullptr;
    missProgramGroup_ = nullptr;
    hitProgramGroup_ = nullptr;
    if (module_)
        optixModuleDestroy(module_);
    module_ = nullptr;
}

bool OptixRenderer::updateShaderBindingTable(const Camera& camera, const RenderSettings& settings)
{
    struct SlangStructuredBuffer
    {
        CUdeviceptr data;
        std::size_t count;
    };
    struct LaunchParameters
    {
        SlangStructuredBuffer output;
        OptixTraversableHandle scene;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t frameIndex;
        std::uint32_t maxBounces;
        std::uint32_t samplesPerFrame;
        std::uint32_t accumulatedSamples;
        Vec4 cameraPositionAndFov;
        Vec4 cameraTargetAndAspect;
        Vec4 cameraUp;
        Vec4 baseColorAndMetallic;
        float roughness{};
        float materialPadding[3]{};
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
    static_assert(sizeof(LaunchParameters) == 128);
    static_assert(sizeof(RaygenRecord) == 160);
    static_assert(sizeof(EmptyRecord) == OPTIX_SBT_RECORD_HEADER_SIZE);

    try
    {
        RaygenRecord record{};
        checkOptix(optixSbtRecordPackHeader(raygenProgramGroup_, &record), "optixSbtRecordPackHeader");
        record.parameters.output = {outputBuffer_, static_cast<std::size_t>(width_) * height_};
        record.parameters.scene = gasHandle_;
        record.parameters.width = width_;
        record.parameters.height = height_;
        record.parameters.frameIndex = static_cast<std::uint32_t>(stats_.frameIndex);
        record.parameters.maxBounces = settings.maxBounces;
        record.parameters.samplesPerFrame = settings.samplesPerFrame;
        record.parameters.accumulatedSamples = static_cast<std::uint32_t>(stats_.accumulatedSamples);
        record.parameters.cameraPositionAndFov = {camera.position.x, camera.position.y, camera.position.z,
                                                  camera.verticalFovDegrees * kPi / 180.0f};
        record.parameters.cameraTargetAndAspect = {camera.target.x, camera.target.y, camera.target.z,
                                                   static_cast<float>(width_) / static_cast<float>(std::max(height_, 1u))};
        record.parameters.cameraUp = {camera.up.x, camera.up.y, camera.up.z, 0.0f};
        Material material{};
        if (scene_ && !scene_->meshes.empty() && scene_->meshes.front().materialIndex < scene_->materials.size())
            material = scene_->materials[scene_->meshes.front().materialIndex];
        record.parameters.baseColorAndMetallic = {material.baseColor.x, material.baseColor.y, material.baseColor.z,
                                                  material.metallic};
        record.parameters.roughness = material.roughness;

        EmptyRecord missRecord{};
        EmptyRecord hitRecord{};
        checkOptix(optixSbtRecordPackHeader(missProgramGroup_, &missRecord), "optixSbtRecordPackHeader(miss)");
        checkOptix(optixSbtRecordPackHeader(hitProgramGroup_, &hitRecord), "optixSbtRecordPackHeader(hit)");
        checkCuda(cuMemcpyHtoD(raygenRecord_, &record, sizeof(record)), "cuMemcpyHtoD(raygen SBT)");
        checkCuda(cuMemcpyHtoD(missRecord_, &missRecord, sizeof(missRecord)), "cuMemcpyHtoD(miss SBT)");
        checkCuda(cuMemcpyHtoD(hitRecord_, &hitRecord, sizeof(hitRecord)), "cuMemcpyHtoD(hit SBT)");
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
