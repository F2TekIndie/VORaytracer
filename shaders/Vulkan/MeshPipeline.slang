import MaterialTypes;
import MaterialSampling;
import Pbr;
import Fresnel;
import Microfacet;
import BsdfTransmission;
import BsdfSubsurface;
import BsdfDiffuse;
import BsdfConductor;
import BsdfClearcoat;
import BsdfSheen;
import BsdfAnisotropic;

struct TaskPayload
{
    uint meshletIndices[32];
};

groupshared TaskPayload taskPayload;
groupshared uint visibleMeshletCount;

struct GpuVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 uv;
};

struct GpuMeshlet
{
    uint vertexOffset;
    uint triangleOffset;
    uint vertexCount;
    uint triangleCount;
    uint instanceIndex;
    uint materialIndex;
    float lodError;
    float nextLodError;
    float4 boundingSphere;
    float4 normalCone;
};

struct GpuInstance
{
    float4x4 transform;
    float4x4 normalTransform;
    float4 scaleAndFlags;
    uint4 geometry;
    uint materialIndex;
    float4 boundsCenter;
};

struct FramePushConstants
{
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 cameraForwardAndFov;
    float4 cameraRightAndAspect;
    float4 cameraUp;
    float4 lightColorAndIntensity;
    float4 lightDirectionAndMode;
    float4 environmentParameters;
    float4 skyZenithAndVisibility;
    float4 skyHorizonAndIndirect;
    float4 skyGround;
    uint rayTracedShadows;
    uint rayTracedReflections;
    float exposure;
    uint meshletCount;
    uint materialOverrideId;
    uint showMeshlets;
    uint renderFlags;
    uint environmentMipCount;
};

static const uint RENDER_FLAG_POST_PROCESS_INPUT = 1u << 0;
static const uint RENDER_FLAG_TRANSPARENT_PASS = 1u << 1;

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex> sceneVertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuMeshlet> sceneMeshlets;
[[vk::binding(2, 0)]] StructuredBuffer<uint> meshletVertexIndices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> meshletTriangleIndices;
[[vk::binding(4, 0)]] RaytracingAccelerationStructure sceneTlas;
[[vk::binding(5, 0)]] StructuredBuffer<GpuInstance> sceneInstances;
[[vk::binding(6, 0)]] StructuredBuffer<uint> geometryIndices;
[[vk::binding(7, 0)]] Texture2D<float4> environmentTexture;
[[vk::binding(8, 0)]] StructuredBuffer<GpuMaterial> sceneMaterials;
[[vk::binding(9, 0)]] Texture2D<float4> materialTextures[1024];
[[vk::binding(10, 0)]] SamplerState materialTextureSampler;
[[vk::binding(11, 0)]] SamplerState environmentSampler;
[[vk::push_constant]] ConstantBuffer<FramePushConstants> frame;

struct MeshVertex
{
    float4 position : SV_Position;
    float3 worldNormal : NORMAL0;
    float3 worldPosition : POSITION0;
    float2 uv : TEXCOORD0;
    float4 worldTangent : TANGENT0;
    nointerpolation uint materialId : TEXCOORD1;
    nointerpolation uint meshletIndex : TEXCOORD2;
    nointerpolation uint instanceIndex : TEXCOORD3;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
};

struct BackgroundVertexOutput
{
    float4 position : SV_Position;
    float3 direction : TEXCOORD0;
};

uint meshletHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float3 meshletDebugColor(uint meshletIndex)
{
    uint hash = meshletHash(meshletIndex + 1u);
    float3 color = float3(float(hash & 0xffu),
                          float((hash >> 8) & 0xffu),
                          float((hash >> 16) & 0xffu)) / 255.0;
    return 0.2 + color * 0.8;
}

float3 materialDebugColor(uint value)
{
    return meshletDebugColor(value * 747796405u + 2891336453u);
}

float3 surfaceDebugColor(uint debugView, SurfaceData surface, GpuMaterial material,
                         uint materialId, float3 geometricNormal, float3 viewDirection)
{
    float3 lightDirection = normalize(float3(0.45, 0.8, 0.35));
    if (debugView == 1u) return surface.baseColor;
    if (debugView == 2u) return float3(surface.metallic);
    if (debugView == 3u) return float3(surface.roughness);
    if (debugView == 4u) return surface.shadingNormal * 0.5 + 0.5;
    if (debugView == 5u) return geometricNormal * 0.5 + 0.5;
    if (debugView == 6u) return surface.tangent * 0.5 + 0.5;
    if (debugView == 7u) return surface.bitangent * 0.5 + 0.5;
    if (debugView == 8u) return float3(surface.occlusion);
    if (debugView == 9u) return surface.emissive;
    if (debugView == 10u)
        return evaluateLambert(surface.baseColor) * (1.0 - surface.metallic) * (1.0 - surface.transmission) * PI;
    if (debugView == 11u)
    {
        float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
        return evaluateConductorGgx(f0, surface.roughness, surface.shadingNormal, viewDirection, lightDirection) * PI;
    }
    if (debugView == 12u) return evaluateClearcoat(surface, viewDirection, lightDirection) * PI;
    if (debugView == 13u) return evaluateSheen(surface, viewDirection, lightDirection) * PI;
    if (debugView == 14u)
        return evaluateRoughTransmission(surface, viewDirection, -lightDirection, 1.0, surface.ior) * PI;
    if (debugView == 15u) return float3(evaluateBsdfPdf(surface, viewDirection, lightDirection));
    if (debugView == 16u) return materialDebugColor(materialId);
    if (debugView == 17u)
        return (material.textureIndices0.x == INVALID_TEXTURE_ID && material.textureIndices0.y == INVALID_TEXTURE_ID &&
                material.textureIndices0.z == INVALID_TEXTURE_ID && material.materialFlags.z == INVALID_TEXTURE_ID &&
                material.textureIndices3.x == INVALID_TEXTURE_ID)
                   ? float3(0.05) :
               materialDebugColor(material.textureIndices0.x ^ (material.textureIndices0.y * 31u) ^
                                  (material.textureIndices0.z * 131u) ^ (material.materialFlags.z * 521u) ^
                                  (material.textureIndices3.x * 8191u));
    if (debugView == 18u)
        return saturate(surface.volumeAbsorption * surface.volumeDensity +
                        surface.volumeScattering * surface.volumeDensity + surface.subsurfaceColor * surface.subsurface);
    if (debugView == 19u) return float3(0.0);
    return float3(0.0);
}

uint effectiveMaterialId(uint materialId)
{
    return frame.materialOverrideId != 0xffffffffu ? frame.materialOverrideId : materialId;
}

GpuMaterial loadMaterial(uint materialId)
{
    return sceneMaterials[effectiveMaterialId(materialId)];
}

float4 sampleMaterialTexture(uint textureId, float2 uv, float4 fallback)
{
    return textureId == INVALID_TEXTURE_ID
               ? fallback
               : materialTextures[NonUniformResourceIndex(textureId)].Sample(materialTextureSampler, uv);
}

float heightValue(float4 sampleValue)
{
    return dot(sampleValue.xyz, float3(0.2126, 0.7152, 0.0722));
}

float2 sampleHeightGradient(GpuMaterial material, float2 uv)
{
    uint textureId = material.materialFlags.z;
    uint packedDimensions = material.materialFlags.w;
    uint width = packedDimensions & 0xffffu;
    uint height = packedDimensions >> 16u;
    if (textureId == INVALID_TEXTURE_ID || width == 0u || height == 0u)
        return float2(0.0);

    float2 texelSize = 1.0 / float2(width, height);
    float2 uvDx = ddx(uv);
    float2 uvDy = ddy(uv);
    uint resourceIndex = NonUniformResourceIndex(textureId);
    float left = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSampler, uv - float2(texelSize.x, 0.0), uvDx, uvDy));
    float right = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSampler, uv + float2(texelSize.x, 0.0), uvDx, uvDy));
    float up = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSampler, uv - float2(0.0, texelSize.y), uvDx, uvDy));
    float down = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSampler, uv + float2(0.0, texelSize.y), uvDx, uvDy));
    return float2(left - right, up - down);
}

SurfaceData resolveMaterialSurface(GpuMaterial material, float3 position, float3 geometricNormal,
                                   float3 shadingNormal, float4 tangentAndSign, float2 uv)
{
    SurfaceData surface = makeSurfaceData(material, position, geometricNormal, shadingNormal, tangentAndSign, uv);
    MaterialTextureSamples samples = defaultMaterialTextureSamples();
    samples.baseColor = sampleMaterialTexture(material.textureIndices0.x, uv, samples.baseColor);
    samples.opacity = sampleMaterialTexture(material.textureIndices3.x, uv, samples.opacity);
    samples.metallicRoughness = sampleMaterialTexture(material.textureIndices0.y, uv, samples.metallicRoughness);
    samples.normal = sampleMaterialTexture(material.textureIndices0.z, uv, samples.normal);
    samples.heightGradient = sampleHeightGradient(material, uv);
    samples.emissive = sampleMaterialTexture(material.textureIndices0.w, uv, samples.emissive);
    samples.occlusion = sampleMaterialTexture(material.textureIndices1.x, uv, samples.occlusion);
    samples.transmission = sampleMaterialTexture(material.textureIndices1.y, uv, samples.transmission);
    samples.clearcoat = sampleMaterialTexture(material.textureIndices1.z, uv, samples.clearcoat);
    samples.clearcoatRoughness = sampleMaterialTexture(material.textureIndices1.w, uv,
                                                       samples.clearcoatRoughness);
    samples.clearcoatNormal = sampleMaterialTexture(material.textureIndices2.x, uv, samples.clearcoatNormal);
    samples.anisotropy = sampleMaterialTexture(material.textureIndices2.w, uv, samples.anisotropy);
    samples.sheenColor = sampleMaterialTexture(material.textureIndices2.y, uv, samples.sheenColor);
    samples.sheenRoughness = sampleMaterialTexture(material.textureIndices2.z, uv, samples.sheenRoughness);
    return applyBaseMaterialTextures(material, surface, samples);
}

struct CandidateMaterial
{
    float opacity;
    uint alphaMode;
    bool accepted;
    bool doubleSided;
};

CandidateMaterial resolveCandidateMaterial(uint instanceIndex, uint primitiveIndex, float2 barycentrics)
{
    CandidateMaterial result;
    GpuInstance instance = sceneInstances[instanceIndex];
    GpuMaterial material = loadMaterial(instance.materialIndex);
    uint triangleOffset = instance.geometry.y + primitiveIndex * 3;
    uint3 triangle = uint3(geometryIndices[triangleOffset], geometryIndices[triangleOffset + 1],
                           geometryIndices[triangleOffset + 2]);
    float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
    float2 uv = sceneVertices[instance.geometry.x + triangle.x].uv.xy * barycentric0 +
                sceneVertices[instance.geometry.x + triangle.y].uv.xy * barycentrics.x +
                sceneVertices[instance.geometry.x + triangle.z].uv.xy * barycentrics.y;
    float textureAlpha = material.textureIndices0.x == INVALID_TEXTURE_ID
                             ? 1.0
                             : materialTextures[NonUniformResourceIndex(material.textureIndices0.x)].SampleLevel(
                                   materialTextureSampler, uv, 0.0).w;
    float opacityTexture = material.textureIndices3.x == INVALID_TEXTURE_ID
                               ? 1.0
                               : materialTextures[NonUniformResourceIndex(material.textureIndices3.x)].SampleLevel(
                                     materialTextureSampler, uv, 0.0).x;
    result.opacity = saturate(material.baseColorFactor.w * textureAlpha * opacityTexture);
    result.alphaMode = material.materialFlags.y;
    result.accepted = result.alphaMode != ALPHA_MODE_MASK || result.opacity >= material.surfaceParameters.w;
    result.doubleSided = materialHasFlag(material, MATERIAL_FLAG_DOUBLE_SIDED);
    return result;
}

float3 evaluateRealtimeSurfaceBsdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return evaluateBsdfWithSubsurfaceMix(surface, viewDirection, lightDirection,
                                         entering ? 1.0 : surface.ior,
                                         entering ? surface.ior : 1.0);
}

float effectiveEnvironmentRoughness(SurfaceData surface, float3 reflectionDirection)
{
    if (abs(surface.anisotropy) <= 1e-4)
        return surface.roughness;
    float3 tangent;
    float3 bitangent;
    anisotropicBasis(surface, tangent, bitangent);
    float2 alphas = anisotropicAlphas(surface);
    float tangentProjection = dot(reflectionDirection, tangent);
    float bitangentProjection = dot(reflectionDirection, bitangent);
    float projectionLength = tangentProjection * tangentProjection + bitangentProjection * bitangentProjection;
    if (projectionLength <= 1e-8)
        return surface.roughness;
    float directionalAlpha = sqrt((tangentProjection * tangentProjection * alphas.x * alphas.x +
                                   bitangentProjection * bitangentProjection * alphas.y * alphas.y) /
                                  projectionLength);
    return clamp(sqrt(max(directionalAlpha, 0.001)), 0.02, 1.0);
}

float3 rotateEnvironmentDirection(float3 direction)
{
    float angle = frame.environmentParameters.w;
    float sine = sin(angle);
    float cosine = cos(angle);
    float3 yawRotated = float3(cosine * direction.x - sine * direction.z,
                               direction.y,
                               sine * direction.x + cosine * direction.z);
    float pitch = asin(clamp(frame.lightDirectionAndMode.y, -1.0, 1.0));
    float pitchSine = sin(pitch);
    float pitchCosine = cos(pitch);
    return float3(yawRotated.x,
                  pitchCosine * yawRotated.y + pitchSine * yawRotated.z,
                  -pitchSine * yawRotated.y + pitchCosine * yawRotated.z);
}

float3 sampleHdrEnvironmentLod(float3 direction, float mipLevel)
{
    if (frame.environmentMipCount == 0)
        return float3(0.0);
    float3 rotatedDirection = rotateEnvironmentDirection(normalize(direction));
    float clampedLevel = clamp(mipLevel, 0.0, float(frame.environmentMipCount - 1));
    float2 uv = float2(atan2(rotatedDirection.z, rotatedDirection.x) / (2.0 * PI) + 0.5,
                            acos(clamp(rotatedDirection.y, -1.0, 1.0)) / PI);
    return environmentTexture.SampleLevel(environmentSampler, uv, clampedLevel).xyz;
}

float3 sampleHdrEnvironment(float3 direction)
{
    return sampleHdrEnvironmentLod(direction, 0.0);
}

float3 sampleGlobalEnvironment(float3 direction)
{
    uint mode = uint(frame.lightDirectionAndMode.w + 0.5);
    if (mode == 1)
        return sampleHdrEnvironment(direction) * frame.environmentParameters.z;
    if (mode == 2)
    {
        float3 normalizedDirection = normalize(direction);
        float upperBlend = pow(saturate(normalizedDirection.y), 0.45);
        float lowerBlend = pow(saturate(-normalizedDirection.y), 0.35);
        float3 sky = lerp(frame.skyHorizonAndIndirect.xyz, frame.skyZenithAndVisibility.xyz, upperBlend);
        sky = lerp(sky, frame.skyGround.xyz, lowerBlend);
        float sun = pow(saturate(dot(normalizedDirection, normalize(-frame.lightDirectionAndMode.xyz))), 512.0);
        return (sky + frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * sun) *
               frame.environmentParameters.z;
    }
    return float3(0.0);
}

float3 sampleDiffuseEnvironment(float3 normal)
{
    uint mode = uint(frame.lightDirectionAndMode.w + 0.5);
    if (mode == 1)
    {
        float diffuseLod = max(float(frame.environmentMipCount) - 5.0, 0.0);
        return sampleHdrEnvironmentLod(normal, diffuseLod) * frame.environmentParameters.z;
    }
    if (mode == 0)
        return float3(0.0);
    return sampleGlobalEnvironment(normal);
}

float3 samplePrefilteredEnvironment(float3 reflectionDirection, float roughness)
{
    reflectionDirection = normalize(reflectionDirection);
    roughness = saturate(roughness);
    if (uint(frame.lightDirectionAndMode.w + 0.5) == 1)
    {
        float specularLod = roughness * float(max(frame.environmentMipCount, 1u) - 1u);
        return sampleHdrEnvironmentLod(reflectionDirection, specularLod) * frame.environmentParameters.z;
    }
    return sampleGlobalEnvironment(reflectionDirection);
}

[shader("vertex")]
BackgroundVertexOutput BackgroundVertex(uint vertexIndex : SV_VertexID)
{
    BackgroundVertexOutput output;
    float2 position = vertexIndex == 0 ? float2(-1.0, -1.0) :
                      vertexIndex == 1 ? float2(-1.0, 3.0) : float2(3.0, -1.0);
    output.position = float4(position, 0.0, 1.0);
    output.direction = normalize(frame.cameraForwardAndFov.xyz +
                                 frame.cameraRightAndAspect.xyz * position.x *
                                     frame.cameraRightAndAspect.w * frame.cameraForwardAndFov.w -
                                 frame.cameraUp.xyz * position.y * frame.cameraForwardAndFov.w);
    return output;
}

[shader("fragment")]
FragmentOutput BackgroundFragment(BackgroundVertexOutput input)
{
    FragmentOutput output;
    float3 color = frame.skyZenithAndVisibility.w != 0.0
                       ? sampleGlobalEnvironment(input.direction) : float3(0.0);
    if ((frame.renderFlags & RENDER_FLAG_POST_PROCESS_INPUT) == 0u)
    {
        color = max(color * exp2(frame.exposure), 0.0);
        color = color / (1.0 + color);
    }
    output.color = float4(color, 1.0);
    return output;
}

[shader("amplification")]
[numthreads(32, 1, 1)]
void TaskMain(
    uint3 groupIndex : SV_GroupID,
    uint threadIndex : SV_GroupThreadID)
{
    if (threadIndex == 0)
        visibleMeshletCount = 0;
    GroupMemoryBarrierWithGroupSync();

    uint meshletIndex = groupIndex.x * 32 + threadIndex;
    bool visible = meshletIndex < frame.meshletCount;
    if (visible)
    {
        GpuMeshlet meshlet = sceneMeshlets[meshletIndex];
        visible = meshlet.vertexCount <= 64 && meshlet.triangleCount <= 124;
        if (visible)
        {
            bool transparentMaterial = loadMaterial(meshlet.materialIndex).materialFlags.y == ALPHA_MODE_BLEND;
            bool transparentPass = (frame.renderFlags & RENDER_FLAG_TRANSPARENT_PASS) != 0u;
            visible = transparentMaterial == transparentPass;
        }
        if (visible)
        {
            GpuInstance instance = sceneInstances[meshlet.instanceIndex];
            float3 worldBoundsCenter = mul(instance.transform, float4(instance.boundsCenter.xyz, 1.0)).xyz;
            float4 boundsClip = mul(frame.viewProjection, float4(worldBoundsCenter, 1.0));
            float projectedRadius = instance.scaleAndFlags.w * instance.scaleAndFlags.x *
                                    abs(frame.viewProjection[1][1]) / max(abs(boundsClip.w), 1e-4);
            // Errors returned by meshoptimizer are relative to mesh extent. Convert them to NDC and
            // choose the coarsest LOD whose projected error stays below the pixel-derived tolerance.
            float projectedError = 2.0 * projectedRadius * meshlet.lodError;
            float nextProjectedError = 2.0 * projectedRadius * meshlet.nextLodError;
            visible = projectedError <= frame.environmentParameters.x &&
                      nextProjectedError > frame.environmentParameters.x;
            if (visible && meshlet.boundingSphere.w > 0.0)
            {
                float3 worldCenter = mul(instance.transform, float4(meshlet.boundingSphere.xyz, 1.0)).xyz;
                float4 clipCenter = mul(frame.viewProjection, float4(worldCenter, 1.0));
                float radius = meshlet.boundingSphere.w * instance.scaleAndFlags.x;
                float radiusX = radius * length(float3(frame.viewProjection[0][0], frame.viewProjection[0][1], frame.viewProjection[0][2]));
                float radiusY = radius * length(float3(frame.viewProjection[1][0], frame.viewProjection[1][1], frame.viewProjection[1][2]));
                float radiusZ = radius * length(float3(frame.viewProjection[2][0], frame.viewProjection[2][1], frame.viewProjection[2][2]));
                float radiusW = radius * length(float3(frame.viewProjection[3][0], frame.viewProjection[3][1], frame.viewProjection[3][2]));
                visible = clipCenter.x + radiusX + radiusW >= -clipCenter.w &&
                          clipCenter.x - radiusX - radiusW <= clipCenter.w &&
                          clipCenter.y + radiusY + radiusW >= -clipCenter.w &&
                          clipCenter.y - radiusY - radiusW <= clipCenter.w &&
                          clipCenter.z + radiusZ >= 0.0 &&
                          clipCenter.z - radiusZ - radiusW <= clipCenter.w;

                float coneLength = length(meshlet.normalCone.xyz);
                if (visible && coneLength > 0.5 && meshlet.normalCone.w <= 1.0 &&
                    instance.scaleAndFlags.y >= instance.scaleAndFlags.x * 0.999)
                {
                    float3 cameraToMeshlet = worldCenter - frame.cameraPosition.xyz;
                    float3 worldConeAxis = normalize(mul((float3x3)instance.normalTransform, meshlet.normalCone.xyz));
                    // meshoptimizer's center/radius variant is deliberately conservative. Omitting the
                    // radius term rejects silhouette meshlets and produces visible holes on curved meshes.
                    visible = dot(worldConeAxis, cameraToMeshlet) <
                              meshlet.normalCone.w * length(cameraToMeshlet) + radius;
                }
            }
        }
    }

    if (visible)
    {
        uint payloadIndex;
        InterlockedAdd(visibleMeshletCount, 1, payloadIndex);
        taskPayload.meshletIndices[payloadIndex] = meshletIndex;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex == 0)
        DispatchMesh(visibleMeshletCount, 1, 1, taskPayload);
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void MeshMain(
    uint3 groupIndex : SV_GroupID,
    uint threadIndex : SV_GroupThreadID,
    in payload TaskPayload taskPayload,
    out vertices MeshVertex vertices[64],
    out indices uint3 triangles[124])
{
    uint meshletIndex = taskPayload.meshletIndices[groupIndex.x];
    GpuMeshlet meshlet = sceneMeshlets[meshletIndex];
    GpuInstance instance = sceneInstances[meshlet.instanceIndex];
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);
    for (uint outputVertex = threadIndex; outputVertex < meshlet.vertexCount; outputVertex += 32)
    {
        uint vertexIndex = meshletVertexIndices[meshlet.vertexOffset + outputVertex];
        GpuVertex source = sceneVertices[vertexIndex];
        float4 worldPosition = mul(instance.transform, float4(source.position.xyz, 1.0));
        vertices[outputVertex].position = mul(frame.viewProjection, worldPosition);
        vertices[outputVertex].worldPosition = worldPosition.xyz;
        vertices[outputVertex].worldNormal = normalize(mul((float3x3)instance.normalTransform, source.normal.xyz));
        vertices[outputVertex].worldTangent = float4(
            normalize(mul((float3x3)instance.transform, source.tangent.xyz)),
            source.tangent.w * instance.scaleAndFlags.z);
        vertices[outputVertex].uv = source.uv.xy;
        vertices[outputVertex].materialId = meshlet.materialIndex;
        vertices[outputVertex].meshletIndex = meshletIndex;
        vertices[outputVertex].instanceIndex = meshlet.instanceIndex;
    }
    for (uint outputTriangle = threadIndex; outputTriangle < meshlet.triangleCount; outputTriangle += 32)
    {
        uint packed = meshletTriangleIndices[meshlet.triangleOffset + outputTriangle];
        triangles[outputTriangle] = uint3(packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff);
    }
}

[shader("fragment")]
FragmentOutput FragmentMain(MeshVertex input, bool frontFacing : SV_IsFrontFace)
{
    FragmentOutput output;
    if (frame.showMeshlets == 1u)
    {
        output.color = float4(meshletDebugColor(input.meshletIndex), 1.0);
        return output;
    }
    GpuMaterial material = loadMaterial(input.materialId);
    if (!frontFacing && !materialHasFlag(material, MATERIAL_FLAG_DOUBLE_SIDED))
        discard;
    float3 interpolatedNormal = normalize(input.worldNormal);
    float3 geometricNormal = cross(ddx(input.worldPosition), ddy(input.worldPosition));
    geometricNormal = dot(geometricNormal, geometricNormal) > 1e-12
                          ? normalize(geometricNormal) : interpolatedNormal;
    if (dot(geometricNormal, interpolatedNormal) < 0.0)
        geometricNormal = -geometricNormal;
    SurfaceData surface = resolveMaterialSurface(material, input.worldPosition, geometricNormal,
                                                 interpolatedNormal, input.worldTangent, input.uv);
    if (material.materialFlags.y == ALPHA_MODE_MASK && surface.alpha < material.surfaceParameters.w)
        discard;
    float3 normal = surface.shadingNormal;
    float3 viewDirection = normalize(frame.cameraPosition.xyz - input.worldPosition);
    if (dot(normal, viewDirection) < 0.0)
    {
        normal = -normal;
        surface.shadingNormal = normal;
        surface.clearcoatNormal = -surface.clearcoatNormal;
    }
    if (frame.showMeshlets > 1u)
    {
        output.color = float4(surfaceDebugColor(frame.showMeshlets - 1u, surface, material,
                                                effectiveMaterialId(input.materialId), geometricNormal,
                                                viewDirection), 1.0);
        return output;
    }
    uint globalLightMode = uint(frame.lightDirectionAndMode.w + 0.5);
    float lightDistance = 10000.0;
    float3 lightDirection = normalize(-frame.lightDirectionAndMode.xyz);
    float3 baseColor = surface.baseColor;
    float visibility = 1.0;
    if (frame.rayTracedShadows != 0 && globalLightMode != 1)
    {
        float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)), abs(input.worldPosition.z));
        float rayBias = max(0.0001, coordinateScale * 0.00001);
        RayDesc shadowRay;
        shadowRay.Origin = input.worldPosition + normal * rayBias;
        shadowRay.Direction = lightDirection;
        shadowRay.TMin = rayBias * 0.5;
        shadowRay.TMax = max(lightDistance - rayBias, rayBias);
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, shadowRay);
        while (query.Proceed())
        {
            CandidateMaterial candidate = resolveCandidateMaterial(query.CandidateInstanceID(),
                                                                    query.CandidatePrimitiveIndex(),
                                                                    query.CandidateTriangleBarycentrics());
            if (!candidate.accepted)
                continue;
            if (!candidate.doubleSided && !query.CandidateTriangleFrontFace())
                continue;
            if (candidate.alphaMode == ALPHA_MODE_BLEND)
            {
                visibility *= 1.0 - candidate.opacity;
                continue;
            }
            query.CommitNonOpaqueTriangleHit();
        }
        if (query.CommittedStatus() != COMMITTED_NOTHING)
            visibility = 0.0;
    }
    float3 directRadiance = globalLightMode == 1
                                ? float3(0.0)
                                : frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * visibility;
    float3 color = evaluateRealtimeSurfaceBsdf(surface, viewDirection, lightDirection) *
                   saturate(dot(normal, lightDirection)) * directRadiance;
    color += surface.emissive;
    if (surface.subsurface > 0.0)
    {
        float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)),
                                    abs(input.worldPosition.z));
        float rayBias = max(0.0001, coordinateScale * 0.00001);
        float thickness = surface.subsurfaceRadius;
        RayDesc thicknessRay;
        thicknessRay.Origin = input.worldPosition - normal * rayBias;
        thicknessRay.Direction = -normal;
        thicknessRay.TMin = rayBias * 0.5;
        thicknessRay.TMax = max(surface.subsurfaceRadius * 12.0, rayBias);
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> thicknessQuery;
        thicknessQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, thicknessRay);
        while (thicknessQuery.Proceed())
        {
            CandidateMaterial candidate = resolveCandidateMaterial(thicknessQuery.CandidateInstanceID(),
                                                                    thicknessQuery.CandidatePrimitiveIndex(),
                                                                    thicknessQuery.CandidateTriangleBarycentrics());
            if (thicknessQuery.CandidateInstanceID() == input.instanceIndex && candidate.accepted &&
                candidate.alphaMode != ALPHA_MODE_BLEND)
                thicknessQuery.CommitNonOpaqueTriangleHit();
        }
        if (thicknessQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            thickness = thicknessQuery.CommittedRayT();
        float3 subsurfaceRadiance = globalLightMode == 1 ? sampleDiffuseEnvironment(-normal) : directRadiance;
        color += evaluateSubsurfaceApproximation(surface, viewDirection, lightDirection, thickness) *
                 subsurfaceRadiance;
    }
    if (frame.skyHorizonAndIndirect.w != 0.0 && surface.metallic < 0.999)
    {
        float3 indirectRadiance = sampleDiffuseEnvironment(normal);
        float nDotV = saturate(dot(normal, viewDirection));
        float3 diffuseF0 = lerp(float3(dielectricF0(surface.ior)), baseColor, surface.metallic);
        float3 diffuseFresnel = fresnelSchlickRoughness(nDotV, diffuseF0, surface.roughness);
        float clearcoatAttenuation = 1.0 - surface.clearcoat *
            fresnelSchlick(nDotV, float3(0.04)).x;
        float3 diffuseWeight = (1.0 - diffuseFresnel) * (1.0 - surface.metallic) *
                               (1.0 - surface.transmission) * (1.0 - surface.subsurface) *
                               sheenEnergyScale(surface) * clearcoatAttenuation;
        color += indirectRadiance * baseColor * diffuseWeight * surface.occlusion;
    }
    if (frame.rayTracedReflections != 0)
    {
        float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)), abs(input.worldPosition.z));
        float rayBias = max(0.0001, coordinateScale * 0.00001);
        RayDesc reflectionRay;
        reflectionRay.Origin = input.worldPosition + normal * rayBias;
        reflectionRay.Direction = normalize(reflect(-viewDirection, normal));
        reflectionRay.TMin = rayBias * 0.5;
        reflectionRay.TMax = 10000.0;
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> reflectionQuery;
        reflectionQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, reflectionRay);
        while (reflectionQuery.Proceed())
        {
            CandidateMaterial candidate = resolveCandidateMaterial(reflectionQuery.CandidateInstanceID(),
                                                                    reflectionQuery.CandidatePrimitiveIndex(),
                                                                    reflectionQuery.CandidateTriangleBarycentrics());
            if (!candidate.accepted || (!candidate.doubleSided &&
                                        !reflectionQuery.CandidateTriangleFrontFace()))
                continue;
            if (candidate.alphaMode == ALPHA_MODE_BLEND)
            {
                uint coverageHash = meshletHash(reflectionQuery.CandidateInstanceID() * 0x9e3779b9u ^
                                                reflectionQuery.CandidatePrimitiveIndex() * 0x85ebca6bu ^
                                                uint(input.position.x) * 0xc2b2ae35u ^ uint(input.position.y));
                float coverageSample = float(coverageHash & 0x00ffffffu) / float(0x01000000u);
                if (candidate.opacity <= coverageSample)
                    continue;
            }
            reflectionQuery.CommitNonOpaqueTriangleHit();
        }

        float environmentRoughness = effectiveEnvironmentRoughness(surface, reflectionRay.Direction);
        float3 environmentReflection = samplePrefilteredEnvironment(reflectionRay.Direction,
                                                                     environmentRoughness);
        float3 reflectedRadiance = environmentReflection;
        if (reflectionQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint instanceIndex = reflectionQuery.CommittedInstanceID();
            GpuInstance hitInstance = sceneInstances[instanceIndex];
            GpuMaterial hitMaterial = loadMaterial(hitInstance.materialIndex);
            uint primitiveIndex = reflectionQuery.CommittedPrimitiveIndex();
            uint triangleOffset = hitInstance.geometry.y + primitiveIndex * 3;
            uint3 triangle = uint3(geometryIndices[triangleOffset],
                                   geometryIndices[triangleOffset + 1],
                                   geometryIndices[triangleOffset + 2]);
            float2 barycentrics = reflectionQuery.CommittedTriangleBarycentrics();
            float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
            float3 hitNormal = normalize(sceneVertices[hitInstance.geometry.x + triangle.x].normal.xyz * barycentric0 +
                                         sceneVertices[hitInstance.geometry.x + triangle.y].normal.xyz * barycentrics.x +
                                         sceneVertices[hitInstance.geometry.x + triangle.z].normal.xyz * barycentrics.y);
            hitNormal = normalize(mul((float3x3)hitInstance.normalTransform, hitNormal));
            float3 hitPosition0 = mul(hitInstance.transform,
                                      float4(sceneVertices[hitInstance.geometry.x + triangle.x].position.xyz, 1.0)).xyz;
            float3 hitPosition1 = mul(hitInstance.transform,
                                      float4(sceneVertices[hitInstance.geometry.x + triangle.y].position.xyz, 1.0)).xyz;
            float3 hitPosition2 = mul(hitInstance.transform,
                                      float4(sceneVertices[hitInstance.geometry.x + triangle.z].position.xyz, 1.0)).xyz;
            float3 hitGeometricNormal = normalize(cross(hitPosition1 - hitPosition0,
                                                        hitPosition2 - hitPosition0));
            if (dot(hitNormal, hitGeometricNormal) < 0.0)
                hitNormal = -hitNormal;
            float3 reflectedViewDirection = normalize(-reflectionRay.Direction);
            float3 hitPosition = reflectionRay.Origin + reflectionRay.Direction * reflectionQuery.CommittedRayT();
            float2 hitUv = sceneVertices[hitInstance.geometry.x + triangle.x].uv.xy * barycentric0 +
                           sceneVertices[hitInstance.geometry.x + triangle.y].uv.xy * barycentrics.x +
                           sceneVertices[hitInstance.geometry.x + triangle.z].uv.xy * barycentrics.y;
            float4 hitTangent = sceneVertices[hitInstance.geometry.x + triangle.x].tangent * barycentric0 +
                                sceneVertices[hitInstance.geometry.x + triangle.y].tangent * barycentrics.x +
                                sceneVertices[hitInstance.geometry.x + triangle.z].tangent * barycentrics.y;
            hitTangent.xyz = normalize(mul((float3x3)hitInstance.transform, hitTangent.xyz));
            hitTangent.w *= hitInstance.scaleAndFlags.z;
            SurfaceData hitSurface = resolveMaterialSurface(hitMaterial, hitPosition, hitGeometricNormal,
                                                            hitNormal, hitTangent, hitUv);
            hitNormal = hitSurface.shadingNormal;
            if (dot(hitNormal, reflectedViewDirection) < 0.0)
            {
                hitNormal = -hitNormal;
                hitSurface.shadingNormal = hitNormal;
                hitSurface.clearcoatNormal = -hitSurface.clearcoatNormal;
            }
            float hitLightDistance = 10000.0;
            float3 hitLightDirection = normalize(-frame.lightDirectionAndMode.xyz);
            float hitVisibility = 1.0;
            if (frame.rayTracedShadows != 0 && globalLightMode != 1)
            {
                float hitCoordinateScale = max(max(abs(hitPosition.x), abs(hitPosition.y)), abs(hitPosition.z));
                float hitRayBias = max(0.0001, hitCoordinateScale * 0.00001);
                RayDesc reflectedShadowRay;
                reflectedShadowRay.Origin = hitPosition + hitNormal * hitRayBias;
                reflectedShadowRay.Direction = hitLightDirection;
                reflectedShadowRay.TMin = hitRayBias * 0.5;
                reflectedShadowRay.TMax = max(hitLightDistance - hitRayBias, hitRayBias);
                RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                         RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> reflectedShadowQuery;
                reflectedShadowQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, reflectedShadowRay);
                while (reflectedShadowQuery.Proceed())
                {
                    CandidateMaterial candidate = resolveCandidateMaterial(
                        reflectedShadowQuery.CandidateInstanceID(),
                        reflectedShadowQuery.CandidatePrimitiveIndex(),
                        reflectedShadowQuery.CandidateTriangleBarycentrics());
                    if (!candidate.accepted)
                        continue;
                    if (!candidate.doubleSided && !reflectedShadowQuery.CandidateTriangleFrontFace())
                        continue;
                    if (candidate.alphaMode == ALPHA_MODE_BLEND)
                    {
                        hitVisibility *= 1.0 - candidate.opacity;
                        continue;
                    }
                    reflectedShadowQuery.CommitNonOpaqueTriangleHit();
                }
                if (reflectedShadowQuery.CommittedStatus() != COMMITTED_NOTHING)
                    hitVisibility = 0.0;
            }
            float3 hitDirectRadiance = globalLightMode == 1
                                           ? float3(0.0)
                                           : frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * hitVisibility;
            float3 localReflection = evaluateRealtimeSurfaceBsdf(hitSurface, reflectedViewDirection,
                                                                 hitLightDirection) *
                                     saturate(dot(hitNormal, hitLightDirection)) * hitDirectRadiance;
            if (frame.skyHorizonAndIndirect.w != 0.0)
                localReflection += sampleDiffuseEnvironment(hitNormal) * hitSurface.baseColor *
                                   (1.0 - hitSurface.metallic) * (1.0 - hitSurface.transmission) *
                                   (1.0 - hitSurface.subsurface) * hitSurface.occlusion;
            localReflection += hitSurface.emissive;
            // A single ray can only represent a sharp local reflection. Fade it into the
            // prefiltered environment once the material lobe is too wide for that sample.
            float localReflectionWeight = 1.0 - smoothstep(0.12, 0.58, environmentRoughness);
            reflectedRadiance = lerp(environmentReflection, localReflection, localReflectionWeight);
        }
        float3 reflectionF0 = lerp(float3(dielectricF0(surface.ior)), baseColor, surface.metallic);
        float3 reflectionWeight = fresnelSchlickRoughness(saturate(dot(normal, viewDirection)),
                                                           reflectionF0, environmentRoughness);
        float clearcoatBaseAttenuation = 1.0 - surface.clearcoat *
            fresnelSchlick(saturate(dot(surface.clearcoatNormal, viewDirection)), float3(0.04)).x;
        color += reflectedRadiance * reflectionWeight * clearcoatBaseAttenuation *
                 sheenEnergyScale(surface);

        if (surface.clearcoat > 0.0)
        {
            float3 clearcoatDirection = normalize(reflect(-viewDirection, surface.clearcoatNormal));
            float3 clearcoatEnvironment = samplePrefilteredEnvironment(clearcoatDirection,
                                                                        surface.clearcoatRoughness);
            float clearcoatWeight = surface.clearcoat *
                fresnelSchlick(saturate(dot(surface.clearcoatNormal, viewDirection)), float3(0.04)).x;
            color += clearcoatEnvironment * clearcoatWeight;
        }
        if (sheenMaximumColor(surface) > 0.0)
        {
            float3 sheenDirection = normalize(reflect(-viewDirection, normal));
            float3 sheenEnvironment = samplePrefilteredEnvironment(sheenDirection,
                                                                    surface.sheenRoughness);
            float grazing = 1.0 - saturate(dot(normal, viewDirection));
            float sheenWeight = 0.2 + 0.3 * surface.sheenRoughness;
            color += sheenEnvironment * surface.sheenColor * sheenWeight *
                     (0.25 + 0.75 * grazing * grazing);
        }
    }
    if (surface.transmission > 0.0)
    {
        float3 internalDirection;
        bool entered = refractDielectric(-viewDirection, normal, 1.0 / surface.ior, internalDirection);
        float3 transmittedRadiance = samplePrefilteredEnvironment(
            entered ? internalDirection : reflect(-viewDirection, normal), surface.roughness);
        float travelDistance = 0.0;
        if (entered)
        {
            float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)),
                                        abs(input.worldPosition.z));
            float rayBias = max(0.0001, coordinateScale * 0.00001);
            RayDesc transmissionRay;
            transmissionRay.Origin = input.worldPosition - normal * rayBias;
            transmissionRay.Direction = internalDirection;
            transmissionRay.TMin = rayBias * 0.5;
            transmissionRay.TMax = 10000.0;
            RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> transmissionQuery;
            transmissionQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, transmissionRay);
            while (transmissionQuery.Proceed())
            {
                CandidateMaterial candidate = resolveCandidateMaterial(transmissionQuery.CandidateInstanceID(),
                                                                        transmissionQuery.CandidatePrimitiveIndex(),
                                                                        transmissionQuery.CandidateTriangleBarycentrics());
                if (transmissionQuery.CandidateInstanceID() == input.instanceIndex && candidate.accepted &&
                    candidate.alphaMode != ALPHA_MODE_BLEND)
                    transmissionQuery.CommitNonOpaqueTriangleHit();
            }
            if (transmissionQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                travelDistance = transmissionQuery.CommittedRayT();
                GpuInstance exitInstance = sceneInstances[transmissionQuery.CommittedInstanceID()];
                uint exitOffset = exitInstance.geometry.y + transmissionQuery.CommittedPrimitiveIndex() * 3;
                uint3 exitTriangle = uint3(geometryIndices[exitOffset], geometryIndices[exitOffset + 1],
                                           geometryIndices[exitOffset + 2]);
                float2 exitBarycentrics = transmissionQuery.CommittedTriangleBarycentrics();
                float exitBarycentric0 = 1.0 - exitBarycentrics.x - exitBarycentrics.y;
                float3 exitPosition0 = mul(exitInstance.transform,
                    float4(sceneVertices[exitInstance.geometry.x + exitTriangle.x].position.xyz, 1.0)).xyz;
                float3 exitPosition1 = mul(exitInstance.transform,
                    float4(sceneVertices[exitInstance.geometry.x + exitTriangle.y].position.xyz, 1.0)).xyz;
                float3 exitPosition2 = mul(exitInstance.transform,
                    float4(sceneVertices[exitInstance.geometry.x + exitTriangle.z].position.xyz, 1.0)).xyz;
                float3 exitNormal = normalize(cross(exitPosition1 - exitPosition0,
                                                    exitPosition2 - exitPosition0));
                if (dot(-internalDirection, exitNormal) < 0.0)
                    exitNormal = -exitNormal;
                float3 exitDirection;
                if (!refractDielectric(internalDirection, exitNormal, surface.ior, exitDirection))
                    exitDirection = reflect(internalDirection, exitNormal);
                transmittedRadiance = samplePrefilteredEnvironment(exitDirection, surface.roughness);
            }
        }
        float fresnel = fresnelDielectric(saturate(dot(normal, viewDirection)), 1.0, surface.ior);
        float3 transmittance = beerLambert(absorptionCoefficient(surface), travelDistance);
        if (surface.volumeDensity > 0.0 && travelDistance > 0.0)
        {
            float3 sigmaT = (surface.volumeAbsorption + surface.volumeScattering) * surface.volumeDensity;
            float3 volumeTransmittance = exp(-sigmaT * travelDistance);
            float3 fogLighting = globalLightMode == 1 ? sampleDiffuseEnvironment(normal)
                                                      : frame.lightColorAndIntensity.xyz *
                                                            frame.lightColorAndIntensity.w;
            float3 scatteringAlbedo = surface.volumeScattering /
                                      max(surface.volumeAbsorption + surface.volumeScattering,
                                          float3(1e-5));
            transmittedRadiance = transmittedRadiance * volumeTransmittance +
                                  fogLighting * scatteringAlbedo * (1.0 - volumeTransmittance);
        }
        color = lerp(color, transmittedRadiance * transmittance * surface.baseColor,
                     surface.transmission * (1.0 - fresnel));
    }
    if ((frame.renderFlags & RENDER_FLAG_POST_PROCESS_INPUT) == 0u)
    {
        color *= exp2(frame.exposure);
        color = max(color, 0.0);
        color = color / (1.0 + color);
    }
    output.color = float4(color, material.materialFlags.y == ALPHA_MODE_BLEND ? surface.alpha : 1.0);
    return output;
}
