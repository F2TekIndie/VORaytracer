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
    float4x4 previousTransform;
    float4x4 normalTransform;
    float4 scaleAndFlags;
    uint4 geometry;
    uint materialIndex;
    float4 boundsCenter;
};

struct TemporalFrameData
{
    float4x4 previousViewProjection;
    uint historyValid;
    float2 outputSize;
    uint padding;
};

struct GpuLight
{
    float4 positionAndType;
    float4 directionAndRange;
    float4 colorAndIntensity;
    float4 coneAndArea;
};

struct GpuLightAlias
{
    float probability;
    uint alias;
    float selectionPdf;
    float padding;
};

struct GpuEmissiveTriangle
{
    float4 position0;
    float4 position1;
    float4 position2;
    float4 emissionAndArea;
    float4 cdfAndPower;
    float4 uv0Uv1;
    float4 uv2AndPadding;
    float4 uvSet1_0_1;
    float4 uvSet1_2AndPadding;
    uint4 materialData;
};

struct GpuTextureMetadata
{
    float4 scaleAndOffset;
    float4 rotationAndUvSet;
};

struct GpuDdgiProbe
{
    float4 positionAndValidity;
    float4 irradiance;
    float4 directionAndDistance;
};

struct DdgiSettings
{
    float4 gridMinimumAndHysteresis;
    float4 gridMaximumAndIntensity;
    uint4 gridDimensions;
    uint4 frameRaysAndFlags;
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
[[vk::binding(10, 0)]] SamplerState materialTextureSamplers[1024];
[[vk::binding(11, 0)]] SamplerState environmentSampler;
[[vk::binding(12, 0)]] Texture2D<float4> iblTexture;
[[vk::binding(13, 0)]] StructuredBuffer<GpuLight> sceneLights;
[[vk::binding(14, 0)]] StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;
[[vk::binding(15, 0)]] StructuredBuffer<GpuTextureMetadata> textureMetadata;
[[vk::binding(16, 0)]] StructuredBuffer<GpuLightAlias> lightAliases;
[[vk::binding(17, 0)]] StructuredBuffer<TemporalFrameData> temporalFrame;
[[vk::binding(18, 0)]] Texture2D<float4> oitAccumulationTexture;
[[vk::binding(19, 0)]] Texture2D<float> oitRevealageTexture;
[[vk::binding(20, 0)]] StructuredBuffer<GpuDdgiProbe> ddgiProbes;
[[vk::binding(21, 0)]] StructuredBuffer<DdgiSettings> ddgiSettingsBuffer;
[[vk::binding(22, 0)]] RWStructuredBuffer<GpuDdgiProbe> ddgiProbeUpdates;
[[vk::push_constant]] ConstantBuffer<FramePushConstants> frame;

struct MeshVertex
{
    float4 position : SV_Position;
    float3 worldNormal : NORMAL0;
    float3 worldPosition : POSITION0;
    float4 uv : TEXCOORD0;
    float4 worldTangent : TANGENT0;
    nointerpolation uint materialId : TEXCOORD1;
    nointerpolation uint meshletIndex : TEXCOORD2;
    nointerpolation uint instanceIndex : TEXCOORD3;
    float4 previousClipPosition : TEXCOORD4;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
};

struct DenoiserFragmentOutput
{
    float4 color : SV_Target0;
    float2 flow : SV_Target1;
    float4 albedo : SV_Target2;
    float4 normalDepth : SV_Target3;
};

struct OitFragmentOutput
{
    float4 accumulation : SV_Target0;
    float revealage : SV_Target1;
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

float2 transformedTextureUv(uint textureId, float4 uvSets)
{
    GpuTextureMetadata metadata = textureMetadata[textureId];
    float2 uv = metadata.rotationAndUvSet.z >= 0.5 ? uvSets.zw : uvSets.xy;
    uv = (uv - 0.5) * metadata.scaleAndOffset.xy;
    float cosine = metadata.rotationAndUvSet.x;
    float sine = metadata.rotationAndUvSet.y;
    uv = float2(cosine * uv.x - sine * uv.y, sine * uv.x + cosine * uv.y);
    return uv + 0.5 + metadata.scaleAndOffset.zw;
}

float4 sampleMaterialTexture(uint textureId, float4 uvSets, float4 fallback)
{
    if (textureId == INVALID_TEXTURE_ID)
        return fallback;
    uint resourceIndex = NonUniformResourceIndex(textureId);
    return materialTextures[resourceIndex].Sample(materialTextureSamplers[resourceIndex],
                                                   transformedTextureUv(textureId, uvSets));
}

float heightValue(float4 sampleValue)
{
    return dot(sampleValue.xyz, float3(0.2126, 0.7152, 0.0722));
}

float2 sampleHeightGradient(GpuMaterial material, float4 uvSets)
{
    uint textureId = material.materialFlags.z;
    uint packedDimensions = material.materialFlags.w;
    uint width = packedDimensions & 0xffffu;
    uint height = packedDimensions >> 16u;
    if (textureId == INVALID_TEXTURE_ID || width == 0u || height == 0u)
        return float2(0.0);

    float2 uv = transformedTextureUv(textureId, uvSets);
    float2 texelSize = 1.0 / float2(width, height);
    float2 uvDx = ddx(uv);
    float2 uvDy = ddy(uv);
    uint resourceIndex = NonUniformResourceIndex(textureId);
    float left = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSamplers[resourceIndex], uv - float2(texelSize.x, 0.0), uvDx, uvDy));
    float right = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSamplers[resourceIndex], uv + float2(texelSize.x, 0.0), uvDx, uvDy));
    float up = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSamplers[resourceIndex], uv - float2(0.0, texelSize.y), uvDx, uvDy));
    float down = heightValue(materialTextures[resourceIndex].SampleGrad(
        materialTextureSamplers[resourceIndex], uv + float2(0.0, texelSize.y), uvDx, uvDy));
    return float2(left - right, up - down);
}

SurfaceData resolveMaterialSurface(GpuMaterial material, float3 position, float3 geometricNormal,
                                   float3 shadingNormal, float4 tangentAndSign, float4 uvSets)
{
    SurfaceData surface = makeSurfaceData(material, position, geometricNormal, shadingNormal, tangentAndSign,
                                          uvSets.xy);
    MaterialTextureSamples samples = defaultMaterialTextureSamples();
    samples.baseColor = sampleMaterialTexture(material.textureIndices0.x, uvSets, samples.baseColor);
    samples.opacity = sampleMaterialTexture(material.textureIndices3.x, uvSets, samples.opacity);
    samples.metallicRoughness = sampleMaterialTexture(material.textureIndices0.y, uvSets,
                                                      samples.metallicRoughness);
    samples.normal = sampleMaterialTexture(material.textureIndices0.z, uvSets, samples.normal);
    samples.heightGradient = sampleHeightGradient(material, uvSets);
    samples.emissive = sampleMaterialTexture(material.textureIndices0.w, uvSets, samples.emissive);
    samples.occlusion = sampleMaterialTexture(material.textureIndices1.x, uvSets, samples.occlusion);
    samples.transmission = sampleMaterialTexture(material.textureIndices1.y, uvSets, samples.transmission);
    samples.clearcoat = sampleMaterialTexture(material.textureIndices1.z, uvSets, samples.clearcoat);
    samples.clearcoatRoughness = sampleMaterialTexture(material.textureIndices1.w, uvSets,
                                                       samples.clearcoatRoughness);
    samples.clearcoatNormal = sampleMaterialTexture(material.textureIndices2.x, uvSets, samples.clearcoatNormal);
    samples.anisotropy = sampleMaterialTexture(material.textureIndices2.w, uvSets, samples.anisotropy);
    samples.sheenColor = sampleMaterialTexture(material.textureIndices2.y, uvSets, samples.sheenColor);
    samples.sheenRoughness = sampleMaterialTexture(material.textureIndices2.z, uvSets,
                                                   samples.sheenRoughness);
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
    float4 uvSets = sceneVertices[instance.geometry.x + triangle.x].uv * barycentric0 +
                    sceneVertices[instance.geometry.x + triangle.y].uv * barycentrics.x +
                    sceneVertices[instance.geometry.x + triangle.z].uv * barycentrics.y;
    float textureAlpha = material.textureIndices0.x == INVALID_TEXTURE_ID
                             ? 1.0
                             : materialTextures[NonUniformResourceIndex(material.textureIndices0.x)].SampleLevel(
                                    materialTextureSamplers[NonUniformResourceIndex(material.textureIndices0.x)],
                                    transformedTextureUv(material.textureIndices0.x, uvSets), 0.0).w;
    float opacityTexture = material.textureIndices3.x == INVALID_TEXTURE_ID
                               ? 1.0
                               : materialTextures[NonUniformResourceIndex(material.textureIndices3.x)].SampleLevel(
                                      materialTextureSamplers[NonUniformResourceIndex(material.textureIndices3.x)],
                                      transformedTextureUv(material.textureIndices3.x, uvSets), 0.0).x;
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

float2 environmentUv(float3 direction)
{
    float3 rotatedDirection = rotateEnvironmentDirection(normalize(direction));
    return float2(atan2(rotatedDirection.z, rotatedDirection.x) / (2.0 * PI) + 0.5,
                  acos(clamp(rotatedDirection.y, -1.0, 1.0)) / PI);
}

float4 sampleIblAtlas(float2 uv, float2 offset, float2 dimensions)
{
    uv.x = frac(uv.x);
    uv.y = saturate(uv.y);
    float2 atlasUv = (offset + 0.5 + uv * max(dimensions - 1.0, 0.0)) / float2(512.0, 256.0);
    return iblTexture.SampleLevel(environmentSampler, atlasUv, 0.0);
}

float3 sampleIblSpecularLevel(float3 direction, uint level)
{
    level = min(level, 8u);
    uint width = max(256u >> level, 1u);
    uint height = max(128u >> level, 1u);
    uint offsetX = level == 0u ? 0u : 512u - (512u >> level);
    return sampleIblAtlas(environmentUv(direction), float2(offsetX, 0u),
                          float2(width, height)).xyz;
}

float2 sampleEnvironmentBrdf(float nDotView, float roughness)
{
    return sampleIblAtlas(float2(saturate(nDotView), saturate(roughness)),
                          float2(256.0, 128.0), float2(128.0, 128.0)).xy;
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
        return sampleIblAtlas(environmentUv(normal), float2(256.0, 64.0),
                              float2(64.0, 32.0)).xyz * frame.environmentParameters.z;
    if (mode == 0)
        return float3(0.0);
    return sampleGlobalEnvironment(normal);
}

float3 sampleDdgiIrradiance(float3 worldPosition, float3 normal)
{
    DdgiSettings settings = ddgiSettingsBuffer[0];
    if (settings.frameRaysAndFlags.z == 0u)
        return float3(0.0);
    float3 extent = max(settings.gridMaximumAndIntensity.xyz - settings.gridMinimumAndHysteresis.xyz,
                        float3(1e-4));
    float3 gridCoordinate = saturate((worldPosition - settings.gridMinimumAndHysteresis.xyz) / extent) *
                            float3(settings.gridDimensions.xyz - 1u);
    uint3 lower = uint3(floor(gridCoordinate));
    uint3 upper = min(lower + 1u, settings.gridDimensions.xyz - 1u);
    float3 blend = frac(gridCoordinate);
    float3 irradiance = float3(0.0);
    float3 dominantDirection = float3(0.0);
    float validity = 0.0;
    for (uint corner = 0; corner < 8u; ++corner)
    {
        uint3 coordinate = uint3((corner & 1u) != 0u ? upper.x : lower.x,
                                 (corner & 2u) != 0u ? upper.y : lower.y,
                                 (corner & 4u) != 0u ? upper.z : lower.z);
        float3 weight3 = float3((corner & 1u) != 0u ? blend.x : 1.0 - blend.x,
                                (corner & 2u) != 0u ? blend.y : 1.0 - blend.y,
                                (corner & 4u) != 0u ? blend.z : 1.0 - blend.z);
        float weight = weight3.x * weight3.y * weight3.z;
        uint index = coordinate.x + settings.gridDimensions.x *
                     (coordinate.y + settings.gridDimensions.y * coordinate.z);
        GpuDdgiProbe probe = ddgiProbes[index];
        float probeWeight = weight * probe.positionAndValidity.w;
        irradiance += probe.irradiance.xyz * probeWeight;
        dominantDirection += probe.directionAndDistance.xyz * probeWeight;
        validity += probeWeight;
    }
    if (validity <= 1e-5)
        return float3(0.0);
    irradiance /= validity;
    dominantDirection = normalize(dominantDirection / validity + float3(1e-6, 0.0, 0.0));
    float directionalWeight = 0.35 + 0.65 * saturate(dot(normal, dominantDirection));
    return irradiance * directionalWeight * settings.gridMaximumAndIntensity.w;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void DdgiUpdateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    DdgiSettings settings = ddgiSettingsBuffer[0];
    uint probeCount = settings.gridDimensions.x * settings.gridDimensions.y * settings.gridDimensions.z;
    uint probeIndex = dispatchThreadId.x;
    if (probeIndex >= probeCount || settings.frameRaysAndFlags.z == 0u)
        return;
    GpuDdgiProbe previous = ddgiProbeUpdates[probeIndex];
    uint rayCount = clamp(settings.frameRaysAndFlags.y, 1u, 32u);
    float3 accumulatedRadiance = float3(0.0);
    float3 accumulatedDirection = float3(0.0);
    float accumulatedDistance = 0.0;
    for (uint rayIndex = 0; rayIndex < rayCount; ++rayIndex)
    {
        uint seed = meshletHash(probeIndex * 0x9e3779b9u ^ rayIndex * 0x85ebca6bu ^
                                settings.frameRaysAndFlags.x * 0xc2b2ae35u);
        float u = (float(rayIndex) + 0.5) / float(rayCount);
        float v = float(seed & 0x00ffffffu) / float(0x01000000u);
        float z = 1.0 - 2.0 * u;
        float radius = sqrt(max(1.0 - z * z, 0.0));
        float angle = 2.0 * PI * v;
        float3 direction = float3(radius * cos(angle), z, radius * sin(angle));
        RayDesc ray;
        ray.Origin = previous.positionAndValidity.xyz;
        ray.Direction = direction;
        ray.TMin = 0.002;
        ray.TMax = length(settings.gridMaximumAndIntensity.xyz - settings.gridMinimumAndHysteresis.xyz) * 2.0;
        RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, ray);
        while (query.Proceed())
        {
            CandidateMaterial candidate = resolveCandidateMaterial(query.CandidateInstanceID(),
                                                                    query.CandidatePrimitiveIndex(),
                                                                    query.CandidateTriangleBarycentrics());
            if (candidate.accepted && candidate.alphaMode != ALPHA_MODE_BLEND &&
                (candidate.doubleSided || query.CandidateTriangleFrontFace()))
                query.CommitNonOpaqueTriangleHit();
        }
        float3 radiance;
        float distance = ray.TMax;
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint instanceIndex = query.CommittedInstanceID();
            GpuInstance instance = sceneInstances[instanceIndex];
            uint primitiveIndex = query.CommittedPrimitiveIndex();
            uint triangleOffset = instance.geometry.y + primitiveIndex * 3u;
            uint3 triangle = uint3(geometryIndices[triangleOffset], geometryIndices[triangleOffset + 1u],
                                   geometryIndices[triangleOffset + 2u]);
            float2 barycentrics = query.CommittedTriangleBarycentrics();
            float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
            float3 normal = normalize(sceneVertices[instance.geometry.x + triangle.x].normal.xyz * barycentric0 +
                                      sceneVertices[instance.geometry.x + triangle.y].normal.xyz * barycentrics.x +
                                      sceneVertices[instance.geometry.x + triangle.z].normal.xyz * barycentrics.y);
            normal = normalize(mul((float3x3)instance.normalTransform, normal));
            if (dot(normal, -direction) < 0.0) normal = -normal;
            float4 uvSets = sceneVertices[instance.geometry.x + triangle.x].uv * barycentric0 +
                            sceneVertices[instance.geometry.x + triangle.y].uv * barycentrics.x +
                            sceneVertices[instance.geometry.x + triangle.z].uv * barycentrics.y;
            GpuMaterial material = loadMaterial(instance.materialIndex);
            float4 baseSample = material.textureIndices0.x == INVALID_TEXTURE_ID ? float4(1.0) :
                materialTextures[NonUniformResourceIndex(material.textureIndices0.x)].SampleLevel(
                    materialTextureSamplers[NonUniformResourceIndex(material.textureIndices0.x)],
                    transformedTextureUv(material.textureIndices0.x, uvSets), 2.0);
            float3 baseColor = max(material.baseColorFactor.xyz * baseSample.xyz, 0.0);
            float metallic = saturate(material.emissiveAndMetallic.w);
            if (material.textureIndices0.y != INVALID_TEXTURE_ID)
                metallic *= materialTextures[NonUniformResourceIndex(material.textureIndices0.y)].SampleLevel(
                    materialTextureSamplers[NonUniformResourceIndex(material.textureIndices0.y)],
                    transformedTextureUv(material.textureIndices0.y, uvSets), 2.0).z;
            float3 emissiveSample = material.textureIndices0.w == INVALID_TEXTURE_ID ? float3(1.0) :
                materialTextures[NonUniformResourceIndex(material.textureIndices0.w)].SampleLevel(
                    materialTextureSamplers[NonUniformResourceIndex(material.textureIndices0.w)],
                    transformedTextureUv(material.textureIndices0.w, uvSets), 2.0).xyz;
            float3 emissive = material.emissiveAndMetallic.xyz * emissiveSample;
            float3 incoming = uint(frame.lightDirectionAndMode.w + 0.5) == 0u
                                  ? frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w *
                                        saturate(dot(normal, normalize(-frame.lightDirectionAndMode.xyz)))
                                  : sampleDiffuseEnvironment(normal);
            radiance = emissive + baseColor * (1.0 - metallic) * incoming;
            distance = query.CommittedRayT();
        }
        else
            radiance = sampleGlobalEnvironment(direction);
        accumulatedRadiance += radiance;
        accumulatedDirection += direction * max(dot(radiance, float3(0.2126, 0.7152, 0.0722)), 1e-4);
        accumulatedDistance += distance;
    }
    float inverseRayCount = 1.0 / float(rayCount);
    float3 irradiance = accumulatedRadiance * inverseRayCount;
    float3 dominantDirection = normalize(accumulatedDirection + float3(1e-6, 0.0, 0.0));
    float averageDistance = accumulatedDistance * inverseRayCount;
    float historyWeight = previous.positionAndValidity.w > 0.0
                              ? saturate(settings.gridMinimumAndHysteresis.w) : 0.0;
    GpuDdgiProbe updated = previous;
    updated.positionAndValidity.w = 1.0;
    updated.irradiance = float4(lerp(irradiance, previous.irradiance.xyz, historyWeight), 0.0);
    updated.directionAndDistance = float4(
        normalize(lerp(dominantDirection, previous.directionAndDistance.xyz, historyWeight)),
        lerp(averageDistance, previous.directionAndDistance.w, historyWeight));
    ddgiProbeUpdates[probeIndex] = updated;
}

float3 samplePrefilteredEnvironment(float3 reflectionDirection, float roughness)
{
    reflectionDirection = normalize(reflectionDirection);
    roughness = saturate(roughness);
    if (uint(frame.lightDirectionAndMode.w + 0.5) == 1)
    {
        float specularLevel = roughness * 8.0;
        uint lowerLevel = uint(floor(specularLevel));
        uint upperLevel = min(lowerLevel + 1u, 8u);
        return lerp(sampleIblSpecularLevel(reflectionDirection, lowerLevel),
                    sampleIblSpecularLevel(reflectionDirection, upperLevel), frac(specularLevel)) *
               frame.environmentParameters.z;
    }
    return sampleGlobalEnvironment(reflectionDirection);
}

float randomFromHash(uint seed)
{
    return (float(meshletHash(seed)) + 0.5) / 4294967296.0;
}

struct DirectLightSample
{
    float3 direction;
    float3 radiance;
    float distance;
};

DirectLightSample sampleAnalyticLight(GpuLight light, float3 position, uint seed)
{
    DirectLightSample sample;
    sample.direction = normalize(-light.directionAndRange.xyz);
    sample.radiance = max(light.colorAndIntensity.xyz, 0.0) * max(light.colorAndIntensity.w, 0.0);
    sample.distance = 10000.0;
    uint type = uint(light.positionAndType.w + 0.5);
    if (type == 0u)
        return sample;

    float3 lightPosition = light.positionAndType.xyz;
    if (type == 3u)
    {
        float3 lightNormal = normalize(light.directionAndRange.xyz);
        float3 helper = abs(lightNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 tangent = normalize(cross(helper, lightNormal));
        float3 bitangent = cross(lightNormal, tangent);
        float2 areaSample = float2(randomFromHash(seed), randomFromHash(seed ^ 0x9e3779b9u)) - 0.5;
        lightPosition += tangent * areaSample.x * light.coneAndArea.z +
                         bitangent * areaSample.y * light.coneAndArea.w;
    }
    float3 toLight = lightPosition - position;
    float distanceSquared = max(dot(toLight, toLight), 1e-8);
    sample.distance = sqrt(distanceSquared);
    sample.direction = toLight / sample.distance;
    float range = max(light.directionAndRange.w, 1e-4);
    float rangeAttenuation = saturate(1.0 - pow(sample.distance / range, 4.0));
    rangeAttenuation *= rangeAttenuation;
    sample.radiance *= rangeAttenuation / distanceSquared;
    if (type == 2u)
    {
        float coneCosine = dot(normalize(light.directionAndRange.xyz), -sample.direction);
        sample.radiance *= smoothstep(light.coneAndArea.y, light.coneAndArea.x, coneCosine);
    }
    else if (type == 3u)
    {
        float cosineAtLight = max(dot(normalize(light.directionAndRange.xyz), -sample.direction), 0.0);
        sample.radiance *= cosineAtLight * max(light.coneAndArea.z * light.coneAndArea.w, 1e-6);
    }
    return sample;
}

DirectLightSample sampleAnalyticLightSet(float3 position, uint seed)
{
    DirectLightSample sample;
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0);
    sample.distance = 0.0;
    uint lightCount;
    uint lightStride;
    uint aliasCount;
    uint aliasStride;
    sceneLights.GetDimensions(lightCount, lightStride);
    lightAliases.GetDimensions(aliasCount, aliasStride);
    if (lightCount == 0u || aliasCount == 0u)
        return sample;
    float scaled = min(randomFromHash(seed), 0.99999994) * float(aliasCount);
    uint column = min(uint(scaled), aliasCount - 1u);
    GpuLightAlias entry = lightAliases[column];
    uint lightIndex = frac(scaled) < entry.probability ? column : entry.alias;
    lightIndex = min(lightIndex, lightCount - 1u);
    float selectionPdf = lightAliases[lightIndex].selectionPdf;
    if (selectionPdf <= 0.0)
        return sample;
    sample = sampleAnalyticLight(sceneLights[lightIndex], position, seed ^ 0x7f4a7c15u);
    sample.radiance /= selectionPdf;
    return sample;
}

float traceDirectVisibility(float3 position, float3 normal, DirectLightSample lightSample)
{
    float coordinateScale = max(max(abs(position.x), abs(position.y)), abs(position.z));
    float rayBias = max(0.0001, coordinateScale * 0.00001);
    RayDesc shadowRay;
    shadowRay.Origin = position + (dot(normal, lightSample.direction) >= 0.0 ? normal : -normal) * rayBias;
    shadowRay.Direction = lightSample.direction;
    shadowRay.TMin = rayBias * 0.5;
    shadowRay.TMax = max(lightSample.distance - rayBias, rayBias);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, shadowRay);
    float visibility = 1.0;
    while (query.Proceed())
    {
        CandidateMaterial candidate = resolveCandidateMaterial(query.CandidateInstanceID(),
                                                                query.CandidatePrimitiveIndex(),
                                                                query.CandidateTriangleBarycentrics());
        if (!candidate.accepted || (!candidate.doubleSided && !query.CandidateTriangleFrontFace()))
            continue;
        if (candidate.alphaMode == ALPHA_MODE_BLEND)
        {
            visibility *= 1.0 - candidate.opacity;
            continue;
        }
        query.CommitNonOpaqueTriangleHit();
    }
    return query.CommittedStatus() == COMMITTED_NOTHING ? visibility : 0.0;
}

DirectLightSample sampleEmissiveLight(float3 position, uint seed)
{
    DirectLightSample sample;
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0);
    sample.distance = 0.0;
    uint count;
    uint stride;
    emissiveTriangles.GetDimensions(count, stride);
    if (count == 0u || emissiveTriangles[count - 1u].cdfAndPower.w <= 0.0)
        return sample;
    float selector = randomFromHash(seed);
    uint low = 0u;
    uint high = count;
    while (low < high)
    {
        uint middle = (low + high) / 2u;
        if (selector < emissiveTriangles[middle].cdfAndPower.y)
            high = middle;
        else
            low = middle + 1u;
    }
    GpuEmissiveTriangle triangle = emissiveTriangles[min(low, count - 1u)];
    float2 randomSample = float2(randomFromHash(seed ^ 0x85ebca6bu),
                                 randomFromHash(seed ^ 0xc2b2ae35u));
    float squareRoot = sqrt(randomSample.x);
    float barycentric0 = 1.0 - squareRoot;
    float barycentric1 = squareRoot * (1.0 - randomSample.y);
    float barycentric2 = squareRoot * randomSample.y;
    float3 lightPosition = triangle.position0.xyz * barycentric0 +
                           triangle.position1.xyz * barycentric1 +
                           triangle.position2.xyz * barycentric2;
    float3 toLight = lightPosition - position;
    float distanceSquared = max(dot(toLight, toLight), 1e-8);
    sample.distance = sqrt(distanceSquared);
    sample.direction = toLight / sample.distance;
    float3 lightNormal = normalize(cross(triangle.position1.xyz - triangle.position0.xyz,
                                         triangle.position2.xyz - triangle.position0.xyz));
    float signedCosine = dot(lightNormal, -sample.direction);
    float cosineAtLight = triangle.materialData.y != 0u ? abs(signedCosine) : max(signedCosine, 0.0);
    float selectionPdf = triangle.cdfAndPower.z / max(triangle.cdfAndPower.w, 1e-8);
    float solidAnglePdf = selectionPdf * distanceSquared /
                          max(triangle.emissionAndArea.w * cosineAtLight, 1e-8);
    float2 uv = triangle.uv0Uv1.xy * barycentric0 + triangle.uv0Uv1.zw * barycentric1 +
                triangle.uv2AndPadding.xy * barycentric2;
    float2 uv1 = triangle.uvSet1_0_1.xy * barycentric0 + triangle.uvSet1_0_1.zw * barycentric1 +
                 triangle.uvSet1_2AndPadding.xy * barycentric2;
    float3 emissiveTexture = sampleMaterialTexture(triangle.materialData.x, float4(uv, uv1), float4(1.0)).xyz;
    sample.radiance = triangle.emissionAndArea.xyz * max(emissiveTexture, 0.0) /
                      max(solidAnglePdf, 1e-8);
    return sample;
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

FragmentOutput shadeBackground(BackgroundVertexOutput input)
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

[shader("fragment")]
FragmentOutput BackgroundFragment(BackgroundVertexOutput input)
{
    return shadeBackground(input);
}

[shader("fragment")]
DenoiserFragmentOutput DenoiserBackgroundFragment(BackgroundVertexOutput input)
{
    FragmentOutput beauty = shadeBackground(input);
    TemporalFrameData temporal = temporalFrame[0];
    float4 previousClip = mul(temporal.previousViewProjection, float4(input.direction, 0.0));
    float2 previousNdc = previousClip.xy / max(abs(previousClip.w), 1e-6);
    float2 previousPixel = (previousNdc * 0.5 + 0.5) * temporal.outputSize;
    DenoiserFragmentOutput output;
    output.color = beauty.color;
    output.flow = temporal.historyValid != 0u ? input.position.xy - previousPixel : float2(0.0);
    output.albedo = float4(0.0);
    output.normalDepth = float4(0.0, 0.0, 0.0, 65504.0);
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
        float4 previousWorldPosition = mul(instance.previousTransform, float4(source.position.xyz, 1.0));
        vertices[outputVertex].previousClipPosition =
            mul(temporalFrame[0].previousViewProjection, previousWorldPosition);
        vertices[outputVertex].worldPosition = worldPosition.xyz;
        vertices[outputVertex].worldNormal = normalize(mul((float3x3)instance.normalTransform, source.normal.xyz));
        vertices[outputVertex].worldTangent = float4(
            normalize(mul((float3x3)instance.transform, source.tangent.xyz)),
            source.tangent.w * instance.scaleAndFlags.z);
        vertices[outputVertex].uv = source.uv;
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

FragmentOutput shadeFragment(MeshVertex input, bool frontFacing)
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
    float3 lightDirection = normalize(-frame.lightDirectionAndMode.xyz);
    float3 baseColor = surface.baseColor;
    float3 directRadiance = float3(0.0);
    float3 color = surface.emissive;
    uint pixelSeed = uint(input.position.x) * 0x9e3779b9u ^ uint(input.position.y) * 0x85ebca6bu;
    if (globalLightMode != 1)
    {
        DirectLightSample lightSample = sampleAnalyticLightSet(input.worldPosition, pixelSeed);
        float visibility = frame.rayTracedShadows != 0
                               ? traceDirectVisibility(input.worldPosition, normal, lightSample) : 1.0;
        float nDotLight = saturate(dot(normal, lightSample.direction));
        color += evaluateRealtimeSurfaceBsdf(surface, viewDirection, lightSample.direction) *
                 nDotLight * lightSample.radiance * visibility;
        lightDirection = lightSample.direction;
        directRadiance = lightSample.radiance * visibility;
    }
    DirectLightSample emissiveLight = sampleEmissiveLight(input.worldPosition, pixelSeed ^ 0x165667b1u);
    if (dot(emissiveLight.radiance, emissiveLight.radiance) > 0.0)
    {
        float visibility = frame.rayTracedShadows != 0
                               ? traceDirectVisibility(input.worldPosition, normal, emissiveLight) : 1.0;
        color += evaluateRealtimeSurfaceBsdf(surface, viewDirection, emissiveLight.direction) *
                 saturate(dot(normal, emissiveLight.direction)) * emissiveLight.radiance * visibility;
    }
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
        float3 indirectRadiance = sampleDiffuseEnvironment(normal) +
                                  sampleDdgiIrradiance(input.worldPosition, normal);
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
            float4 hitUv = sceneVertices[hitInstance.geometry.x + triangle.x].uv * barycentric0 +
                           sceneVertices[hitInstance.geometry.x + triangle.y].uv * barycentrics.x +
                           sceneVertices[hitInstance.geometry.x + triangle.z].uv * barycentrics.y;
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
            float3 localReflection = float3(0.0);
            if (globalLightMode != 1)
            {
                DirectLightSample hitLight = sampleAnalyticLightSet(hitPosition, pixelSeed ^ 0xb5297a4du);
                float hitVisibility = frame.rayTracedShadows != 0
                                          ? traceDirectVisibility(hitPosition, hitNormal, hitLight) : 1.0;
                localReflection += evaluateRealtimeSurfaceBsdf(hitSurface, reflectedViewDirection,
                                                               hitLight.direction) *
                                   saturate(dot(hitNormal, hitLight.direction)) *
                                   hitLight.radiance * hitVisibility;
            }
            DirectLightSample hitEmissiveLight = sampleEmissiveLight(hitPosition, pixelSeed ^ 0x68e31da4u);
            if (dot(hitEmissiveLight.radiance, hitEmissiveLight.radiance) > 0.0)
            {
                float hitVisibility = frame.rayTracedShadows != 0
                                          ? traceDirectVisibility(hitPosition, hitNormal, hitEmissiveLight) : 1.0;
                localReflection += evaluateRealtimeSurfaceBsdf(hitSurface, reflectedViewDirection,
                                                               hitEmissiveLight.direction) *
                                   saturate(dot(hitNormal, hitEmissiveLight.direction)) *
                                   hitEmissiveLight.radiance * hitVisibility;
            }
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
        float nDotView = saturate(dot(normal, viewDirection));
        float3 reflectionWeight;
        if (globalLightMode == 1)
        {
            float2 environmentBrdf = sampleEnvironmentBrdf(nDotView, environmentRoughness);
            reflectionWeight = reflectionF0 * environmentBrdf.x + environmentBrdf.y;
        }
        else
            reflectionWeight = fresnelSchlickRoughness(nDotView, reflectionF0, environmentRoughness);
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

[shader("fragment")]
FragmentOutput FragmentMain(MeshVertex input, bool frontFacing : SV_IsFrontFace)
{
    return shadeFragment(input, frontFacing);
}

[shader("fragment")]
OitFragmentOutput OitFragmentMain(MeshVertex input, bool frontFacing : SV_IsFrontFace)
{
    FragmentOutput shaded = shadeFragment(input, frontFacing);
    float alpha = saturate(shaded.color.a);
    float depthWeight = max(1.0 - input.position.z * 0.9, 0.1);
    float weight = clamp((alpha * 8.0 + 0.01) * depthWeight * depthWeight * depthWeight,
                         0.01, 8.0);
    OitFragmentOutput output;
    output.accumulation = float4(shaded.color.rgb * alpha, alpha) * weight;
    output.revealage = alpha;
    return output;
}

[shader("fragment")]
FragmentOutput OitCompositeFragment(BackgroundVertexOutput input)
{
    int2 pixel = int2(input.position.xy);
    float4 accumulation = oitAccumulationTexture.Load(int3(pixel, 0));
    float revealage = saturate(oitRevealageTexture.Load(int3(pixel, 0)));
    FragmentOutput output;
    output.color = float4(accumulation.rgb / max(accumulation.a, 1e-5), 1.0 - revealage);
    return output;
}

[shader("fragment")]
DenoiserFragmentOutput DenoiserFragmentMain(MeshVertex input, bool frontFacing : SV_IsFrontFace)
{
    FragmentOutput beauty = shadeFragment(input, frontFacing);
    GpuMaterial material = loadMaterial(input.materialId);
    float3 interpolatedNormal = normalize(input.worldNormal);
    float3 geometricNormal = cross(ddx(input.worldPosition), ddy(input.worldPosition));
    geometricNormal = dot(geometricNormal, geometricNormal) > 1e-12
                          ? normalize(geometricNormal) : interpolatedNormal;
    if (dot(geometricNormal, interpolatedNormal) < 0.0)
        geometricNormal = -geometricNormal;
    SurfaceData surface = resolveMaterialSurface(material, input.worldPosition, geometricNormal,
                                                 interpolatedNormal, input.worldTangent, input.uv);
    float3 viewDirection = normalize(frame.cameraPosition.xyz - input.worldPosition);
    float3 normal = dot(surface.shadingNormal, viewDirection) < 0.0
                        ? -surface.shadingNormal : surface.shadingNormal;
    TemporalFrameData temporal = temporalFrame[0];
    float2 previousNdc = input.previousClipPosition.xy /
                         max(abs(input.previousClipPosition.w), 1e-6);
    float2 previousPixel = (previousNdc * 0.5 + 0.5) * temporal.outputSize;
    float2 flow = temporal.historyValid != 0u ? input.position.xy - previousPixel : float2(0.0);
    DenoiserFragmentOutput output;
    output.color = beauty.color;
    output.flow = flow;
    output.albedo = float4(surface.baseColor * (1.0 - surface.metallic), 1.0);
    output.normalDepth = float4(normal, length(frame.cameraPosition.xyz - input.worldPosition));
    return output;
}
