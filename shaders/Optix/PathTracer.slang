import MaterialTypes;
import MaterialSampling;
import Pbr;
import Fresnel;
import BsdfTypes;
import BsdfTransmission;
import BsdfSubsurface;
import PhaseFunctions;
import Microfacet;
import BsdfDiffuse;
import BsdfConductor;
import BsdfClearcoat;
import BsdfSheen;

struct RadiancePayload
{
    uint primitiveIndex;
    uint instanceIndex;
    float2 barycentrics;
    float hitDistance;
    uint hit;
};

struct MaterialTextureTable
{
    Texture2D<float4> textures[1024];
    SamplerState sampler;
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

struct GpuLight
{
    float4 positionAndType;
    float4 directionAndRange;
    float4 colorAndIntensity;
    float4 coneAndArea;
};

struct GpuOptixInstance
{
    float4x4 transform;
    float4x4 normalTransform;
    uint4 geometry;
    uint materialIndex;
    float handedness;
    uint2 padding;
};

struct LaunchParameters
{
    RWStructuredBuffer<float4> output;
    RWStructuredBuffer<half4> denoisedOutput;
    RWStructuredBuffer<float4> albedoGuide;
    RWStructuredBuffer<float4> normalGuide;
    RWStructuredBuffer<uint> displayOutput;
    StructuredBuffer<float3> positions;
    StructuredBuffer<float3> normals;
    StructuredBuffer<float4> tangents;
    StructuredBuffer<float4> uvs;
    StructuredBuffer<uint3> indices;
    StructuredBuffer<GpuOptixInstance> instances;
    StructuredBuffer<GpuMaterial> materials;
    StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;
    StructuredBuffer<GpuLight> lights;
    StructuredBuffer<GpuTextureMetadata> textureMetadata;
    StructuredBuffer<float4> environmentPixels;
    StructuredBuffer<float> environmentConditionalCdf;
    StructuredBuffer<float> environmentMarginalCdf;
    ConstantBuffer<MaterialTextureTable> materialTextureTable;
    RaytracingAccelerationStructure scene;
    uint2 outputSize;
    uint frameIndex;
    uint maxBounces;
    uint samplesPerFrame;
    uint accumulatedSamples;
    uint rayTracedShadows;
    uint rayTracedReflections;
    uint displayBgra;
    float exposure;
    uint writeDisplay;
    uint materialOverrideId;
    uint debugView;
    uint padding;
    uint indirectLighting;
    uint globalLightMode;
    uint environmentWidth;
    uint environmentHeight;
    float environmentIntensity;
    float environmentRotation;
    uint environmentVisible;
    float environmentImportanceTotal;
    float emissiveLightPower;
    uint2 importancePadding;
    float4 cameraPositionAndFov;
    float4 cameraTargetAndAspect;
    float4 cameraUp;
    float4 lightPosition;
    float4 lightColorAndIntensity;
    float4 lightDirection;
    float4 skyZenith;
    float4 skyHorizon;
    float4 skyGround;
};

float2 transformedTextureUv(LaunchParameters parameters, uint textureId, float4 uvSets)
{
    GpuTextureMetadata metadata = parameters.textureMetadata[textureId];
    float2 uv = metadata.rotationAndUvSet.z >= 0.5 ? uvSets.zw : uvSets.xy;
    uv -= float2(0.5);
    uv = float2(uv.x * metadata.rotationAndUvSet.x - uv.y * metadata.rotationAndUvSet.y,
                uv.x * metadata.rotationAndUvSet.y + uv.y * metadata.rotationAndUvSet.x);
    return uv * metadata.scaleAndOffset.xy + float2(0.5) + metadata.scaleAndOffset.zw;
}

float4 sampleMaterialTexture(LaunchParameters parameters, uint textureId, float4 uvSets, float4 fallback,
                             float textureLod)
{
    return textureId == INVALID_TEXTURE_ID
               ? fallback
               : parameters.materialTextureTable.textures[textureId].SampleLevel(
                      parameters.materialTextureTable.sampler,
                      transformedTextureUv(parameters, textureId, uvSets), max(textureLod, 0.0));
}

float heightValue(float4 sampleValue)
{
    return dot(sampleValue.xyz, float3(0.2126, 0.7152, 0.0722));
}

float2 sampleHeightGradient(LaunchParameters parameters, GpuMaterial material, float4 uvSets, float textureLod)
{
    uint textureId = material.materialFlags.z;
    uint packedDimensions = material.materialFlags.w;
    uint width = packedDimensions & 0xffffu;
    uint height = packedDimensions >> 16u;
    if (textureId == INVALID_TEXTURE_ID || width == 0u || height == 0u)
        return float2(0.0);

    float lod = max(textureLod, 0.0);
    float mipScale = exp2(floor(lod));
    float2 texelSize = min(mipScale / float2(width, height), float2(0.5));
    float2 uv = transformedTextureUv(parameters, textureId, uvSets);
    float left = heightValue(parameters.materialTextureTable.textures[textureId].SampleLevel(
        parameters.materialTextureTable.sampler, uv - float2(texelSize.x, 0.0), lod));
    float right = heightValue(parameters.materialTextureTable.textures[textureId].SampleLevel(
        parameters.materialTextureTable.sampler, uv + float2(texelSize.x, 0.0), lod));
    float up = heightValue(parameters.materialTextureTable.textures[textureId].SampleLevel(
        parameters.materialTextureTable.sampler, uv - float2(0.0, texelSize.y), lod));
    float down = heightValue(parameters.materialTextureTable.textures[textureId].SampleLevel(
        parameters.materialTextureTable.sampler, uv + float2(0.0, texelSize.y), lod));
    return float2(left - right, up - down);
}

SurfaceData resolveMaterialSurface(LaunchParameters parameters, GpuMaterial material, float3 position,
                                   float3 geometricNormal, float3 shadingNormal,
                                    float4 tangentAndSign, float4 uvSets, float textureLod)
{
    SurfaceData surface = makeSurfaceData(material, position, geometricNormal, shadingNormal, tangentAndSign,
                                          uvSets.xy);
    MaterialTextureSamples samples = defaultMaterialTextureSamples();
    samples.baseColor = sampleMaterialTexture(parameters, material.textureIndices0.x, uvSets, samples.baseColor,
                                               textureLod);
    samples.opacity = sampleMaterialTexture(parameters, material.textureIndices3.x, uvSets, samples.opacity,
                                             textureLod);
    samples.metallicRoughness = sampleMaterialTexture(parameters, material.textureIndices0.y, uvSets,
                                                      samples.metallicRoughness, textureLod);
    samples.normal = sampleMaterialTexture(parameters, material.textureIndices0.z, uvSets, samples.normal, textureLod);
    samples.heightGradient = sampleHeightGradient(parameters, material, uvSets, textureLod);
    samples.emissive = sampleMaterialTexture(parameters, material.textureIndices0.w, uvSets, samples.emissive,
                                              textureLod);
    samples.occlusion = sampleMaterialTexture(parameters, material.textureIndices1.x, uvSets, samples.occlusion,
                                               textureLod);
    samples.transmission = sampleMaterialTexture(parameters, material.textureIndices1.y, uvSets, samples.transmission,
                                                  textureLod);
    samples.clearcoat = sampleMaterialTexture(parameters, material.textureIndices1.z, uvSets, samples.clearcoat,
                                               textureLod);
    samples.clearcoatRoughness = sampleMaterialTexture(parameters, material.textureIndices1.w, uvSets,
                                                       samples.clearcoatRoughness, textureLod);
    samples.clearcoatNormal = sampleMaterialTexture(parameters, material.textureIndices2.x, uvSets,
                                                    samples.clearcoatNormal, textureLod);
    samples.anisotropy = sampleMaterialTexture(parameters, material.textureIndices2.w, uvSets, samples.anisotropy,
                                                textureLod);
    samples.sheenColor = sampleMaterialTexture(parameters, material.textureIndices2.y, uvSets, samples.sheenColor,
                                                textureLod);
    samples.sheenRoughness = sampleMaterialTexture(parameters, material.textureIndices2.z, uvSets,
                                                   samples.sheenRoughness, textureLod);
    return applyBaseMaterialTextures(material, surface, samples);
}

struct ResolvedHit
{
    GpuMaterial material;
    SurfaceData surface;
    float3 position;
    float3 geometricNormal;
    uint materialIndex;
    uint instanceIndex;
};

ResolvedHit resolveHit(LaunchParameters parameters, RadiancePayload payload, RayDesc ray, float textureLod)
{
    ResolvedHit result;
    GpuOptixInstance hitInstance = parameters.instances[payload.instanceIndex];
    uint3 localTriangle = parameters.indices[hitInstance.geometry.y + payload.primitiveIndex];
    uint3 triangle = localTriangle + hitInstance.geometry.x;
    float barycentric0 = 1.0 - payload.barycentrics.x - payload.barycentrics.y;

    float3 worldPosition0 = mul(hitInstance.transform, float4(parameters.positions[triangle.x], 1.0)).xyz;
    float3 worldPosition1 = mul(hitInstance.transform, float4(parameters.positions[triangle.y], 1.0)).xyz;
    float3 worldPosition2 = mul(hitInstance.transform, float4(parameters.positions[triangle.z], 1.0)).xyz;
    result.geometricNormal = normalize(cross(worldPosition1 - worldPosition0, worldPosition2 - worldPosition0));
    result.position = ray.Origin + ray.Direction * payload.hitDistance;

    float3 shadingNormal = normalize(parameters.normals[triangle.x] * barycentric0 +
                                     parameters.normals[triangle.y] * payload.barycentrics.x +
                                     parameters.normals[triangle.z] * payload.barycentrics.y);
    shadingNormal = normalize(mul((float3x3)hitInstance.normalTransform, shadingNormal));
    if (dot(shadingNormal, result.geometricNormal) < 0.0)
        shadingNormal = -shadingNormal;
    float4 vertexTangent = parameters.tangents[triangle.x] * barycentric0 +
                           parameters.tangents[triangle.y] * payload.barycentrics.x +
                           parameters.tangents[triangle.z] * payload.barycentrics.y;
    vertexTangent.xyz = normalize(mul((float3x3)hitInstance.transform, vertexTangent.xyz));
    vertexTangent.w *= hitInstance.handedness;
    float4 uv = parameters.uvs[triangle.x] * barycentric0 +
                parameters.uvs[triangle.y] * payload.barycentrics.x +
                parameters.uvs[triangle.z] * payload.barycentrics.y;

    result.materialIndex = parameters.materialOverrideId != 0xffffffffu
                               ? parameters.materialOverrideId
                               : hitInstance.materialIndex;
    result.material = parameters.materials[result.materialIndex];
    result.instanceIndex = payload.instanceIndex;
    result.surface = resolveMaterialSurface(parameters, result.material, result.position,
                                            result.geometricNormal, shadingNormal, vertexTangent, uv, textureLod);
    return result;
}

uint debugHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float3 debugIdColor(uint value)
{
    uint hash = debugHash(value + 1u);
    return 0.2 + float3(float(hash & 0xffu), float((hash >> 8) & 0xffu),
                        float((hash >> 16) & 0xffu)) / 318.75;
}

float3 surfaceDebugColor(uint debugView, SurfaceData surface, GpuMaterial material,
                         uint materialId, float3 geometricNormal, float3 viewDirection, uint mediumDepth)
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
    if (debugView == 16u) return debugIdColor(materialId);
    if (debugView == 17u)
        return (material.textureIndices0.x == INVALID_TEXTURE_ID && material.textureIndices0.y == INVALID_TEXTURE_ID &&
                material.textureIndices0.z == INVALID_TEXTURE_ID && material.materialFlags.z == INVALID_TEXTURE_ID &&
                material.textureIndices3.x == INVALID_TEXTURE_ID)
                   ? float3(0.05) :
               debugIdColor(material.textureIndices0.x ^ (material.textureIndices0.y * 31u) ^
                            (material.textureIndices0.z * 131u) ^ (material.materialFlags.z * 521u) ^
                            (material.textureIndices3.x * 8191u));
    if (debugView == 18u)
        return mediumDepth > 1u ? float3(0.12, 0.52, 0.9) :
               saturate(surface.volumeAbsorption * surface.volumeDensity +
                        surface.volumeScattering * surface.volumeDensity + surface.subsurfaceColor * surface.subsurface);
    return float3(0.0);
}

float random01(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return float(value & 0x00ffffff) / float(0x01000000);
}

bool skipCameraPathHit(ResolvedHit hit, float3 rayDirection, float coverageSample)
{
    bool frontFacing = dot(hit.geometricNormal, -rayDirection) > 0.0;
    bool needsBackFaceForTransport = hit.surface.transmission > 0.0 || hit.surface.volumeDensity > 0.0;
    if (!frontFacing && (hit.surface.flags & MATERIAL_FLAG_DOUBLE_SIDED) == 0u && !needsBackFaceForTransport)
        return true;
    uint alphaMode = hit.material.materialFlags.y;
    if (alphaMode == ALPHA_MODE_MASK)
        return hit.surface.alpha < hit.material.surfaceParameters.w;
    if (alphaMode == ALPHA_MODE_BLEND)
        return coverageSample >= hit.surface.alpha;
    return false;
}

float3 traceVisibility(LaunchParameters parameters, RayDesc sourceRay)
{
    float3 visibility = float3(1.0);
    RayDesc shadowRay = sourceRay;
    for (uint layer = 0; layer < 32u; ++layer)
    {
        RadiancePayload shadowPayload;
        shadowPayload.primitiveIndex = 0;
        shadowPayload.instanceIndex = 0;
        shadowPayload.barycentrics = float2(0.0);
        shadowPayload.hitDistance = 0.0;
        shadowPayload.hit = 0;
        TraceRay(parameters.scene,
                 RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                 0xff, 0, 1, 0, shadowRay, shadowPayload);
        if (shadowPayload.hit == 0)
            return visibility;

        float shadowTextureLod = log2(max(shadowPayload.hitDistance * 0.05, 1.0));
        ResolvedHit hit = resolveHit(parameters, shadowPayload, shadowRay, shadowTextureLod);
        uint alphaMode = hit.material.materialFlags.y;
        // Sidedness controls surface shading, not whether an opaque triangle casts a shadow.
        // Treating single-sided back faces as transparent creates triangle-shaped light leaks on
        // open FBX meshes such as cloth, hoods and layered character parts.
        bool passThrough = alphaMode == ALPHA_MODE_MASK &&
                           hit.surface.alpha < hit.material.surfaceParameters.w;
        if (alphaMode == ALPHA_MODE_BLEND)
        {
            visibility *= 1.0 - hit.surface.alpha;
            passThrough = true;
        }
        if (hit.surface.transmission > 0.0)
        {
            float cosine = saturate(abs(dot(hit.geometricNormal, shadowRay.Direction)));
            float fresnel = fresnelDielectric(cosine, 1.0, hit.surface.ior);
            visibility *= lerp(float3(1.0), hit.surface.baseColor,
                               hit.surface.transmission * (1.0 - fresnel));
            passThrough = true;
        }
        if (!passThrough)
            return float3(0.0);
        if (max(visibility.x, max(visibility.y, visibility.z)) <= 1e-4)
            return float3(0.0);

        float coordinateScale = max(max(abs(hit.position.x), abs(hit.position.y)), abs(hit.position.z));
        float bias = max(0.0001, coordinateScale * 0.00001);
        float consumed = shadowPayload.hitDistance + bias;
        shadowRay.Origin = hit.position + shadowRay.Direction * bias;
        shadowRay.TMin = bias * 0.5;
        shadowRay.TMax = max(shadowRay.TMax - consumed, shadowRay.TMin);
        if (shadowRay.TMax <= shadowRay.TMin)
            return visibility;
    }
    return visibility;
}

float emissiveMeshHitPdf(LaunchParameters parameters, ResolvedHit hit, float3 rayDirection,
                         float hitDistance)
{
    if (parameters.emissiveLightPower <= 0.0 || parameters.materialOverrideId != 0xffffffffu)
        return 0.0;
    float emissionPower = luminance(max(hit.material.emissiveAndMetallic.xyz, 0.0));
    float signedCosineAtLight = dot(hit.geometricNormal, -rayDirection);
    float cosineAtLight = (hit.surface.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u
                              ? abs(signedCosineAtLight) : max(signedCosineAtLight, 0.0);
    if (emissionPower <= 0.0 || cosineAtLight <= 1e-6)
        return 0.0;
    float solidAnglePdf = emissionPower * hitDistance * hitDistance /
                          max(parameters.emissiveLightPower * cosineAtLight, 1e-8);
    return solidAnglePdf * emissiveLightSelectionProbability(parameters);
}

float3 rotateEnvironmentDirection(float3 direction, float angle, float3 lightDirection)
{
    float sine = sin(angle);
    float cosine = cos(angle);
    float3 yawRotated = float3(cosine * direction.x - sine * direction.z,
                               direction.y,
                               sine * direction.x + cosine * direction.z);
    float pitch = asin(clamp(lightDirection.y, -1.0, 1.0));
    float pitchSine = sin(pitch);
    float pitchCosine = cos(pitch);
    return float3(yawRotated.x,
                  pitchCosine * yawRotated.y + pitchSine * yawRotated.z,
                  -pitchSine * yawRotated.y + pitchCosine * yawRotated.z);
}

float3 inverseRotateEnvironmentDirection(float3 direction, float angle, float3 lightDirection)
{
    float pitch = asin(clamp(lightDirection.y, -1.0, 1.0));
    float pitchSine = sin(pitch);
    float pitchCosine = cos(pitch);
    float3 inversePitch = float3(direction.x,
                                 pitchCosine * direction.y - pitchSine * direction.z,
                                 pitchSine * direction.y + pitchCosine * direction.z);
    float sine = sin(angle);
    float cosine = cos(angle);
    return normalize(float3(cosine * inversePitch.x + sine * inversePitch.z,
                            inversePitch.y,
                            -sine * inversePitch.x + cosine * inversePitch.z));
}

float3 sampleHdrEnvironment(LaunchParameters parameters, float3 direction)
{
    if (parameters.environmentWidth == 0 || parameters.environmentHeight == 0)
        return float3(0.0);
    float3 rotated = rotateEnvironmentDirection(normalize(direction), parameters.environmentRotation,
                                                 parameters.lightDirection.xyz);
    float2 uv = float2(atan2(rotated.z, rotated.x) / (2.0 * PI) + 0.5,
                            acos(clamp(rotated.y, -1.0, 1.0)) / PI);
    float2 texel = uv * float2(parameters.environmentWidth, parameters.environmentHeight) - 0.5;
    int2 base = int2(floor(texel));
    float2 fraction = frac(texel);
    uint x0 = uint((base.x % int(parameters.environmentWidth) + int(parameters.environmentWidth)) %
                   int(parameters.environmentWidth));
    uint x1 = (x0 + 1) % parameters.environmentWidth;
    uint y0 = uint(clamp(base.y, 0, int(parameters.environmentHeight) - 1));
    uint y1 = min(y0 + 1, parameters.environmentHeight - 1);
    float3 a = lerp(parameters.environmentPixels[y0 * parameters.environmentWidth + x0].xyz,
                    parameters.environmentPixels[y0 * parameters.environmentWidth + x1].xyz, fraction.x);
    float3 b = lerp(parameters.environmentPixels[y1 * parameters.environmentWidth + x0].xyz,
                    parameters.environmentPixels[y1 * parameters.environmentWidth + x1].xyz, fraction.x);
    return lerp(a, b, fraction.y);
}

float3 sampleGlobalEnvironment(LaunchParameters parameters, float3 direction)
{
    if (parameters.globalLightMode == 1)
        return sampleHdrEnvironment(parameters, direction) * parameters.environmentIntensity;
    if (parameters.globalLightMode == 2)
    {
        float3 normalizedDirection = normalize(direction);
        float upperBlend = pow(saturate(normalizedDirection.y), 0.45);
        float lowerBlend = pow(saturate(-normalizedDirection.y), 0.35);
        float3 sky = lerp(parameters.skyHorizon.xyz, parameters.skyZenith.xyz, upperBlend);
        sky = lerp(sky, parameters.skyGround.xyz, lowerBlend);
        float sun = pow(saturate(dot(normalizedDirection, normalize(-parameters.lightDirection.xyz))), 512.0);
        return (sky + parameters.lightColorAndIntensity.xyz * parameters.lightColorAndIntensity.w * sun) *
               parameters.environmentIntensity;
    }
    return float3(0.0);
}

uint sampleCdf(StructuredBuffer<float> cdf, uint offset, uint intervalCount, float randomValue)
{
    uint low = 0;
    uint high = intervalCount;
    while (low + 1 < high)
    {
        uint middle = (low + high) / 2;
        if (cdf[offset + middle] <= randomValue)
            low = middle;
        else
            high = middle;
    }
    return min(low, intervalCount - 1);
}

float remapCdfSample(StructuredBuffer<float> cdf, uint offset, uint index, float randomValue)
{
    float lower = cdf[offset + index];
    float upper = cdf[offset + index + 1];
    return saturate((randomValue - lower) / max(upper - lower, 1e-8));
}

float hdrEnvironmentPdf(LaunchParameters parameters, float3 direction)
{
    if (parameters.environmentWidth == 0 || parameters.environmentHeight == 0 ||
        parameters.environmentImportanceTotal <= 0.0)
        return 0.0;
    float3 rotated = rotateEnvironmentDirection(normalize(direction), parameters.environmentRotation,
                                                 parameters.lightDirection.xyz);
    float2 uv = float2(atan2(rotated.z, rotated.x) / (2.0 * PI) + 0.5,
                            acos(clamp(rotated.y, -1.0, 1.0)) / PI);
    uint x = min(uint(frac(uv.x) * parameters.environmentWidth), parameters.environmentWidth - 1);
    uint y = min(uint(saturate(uv.y) * parameters.environmentHeight), parameters.environmentHeight - 1);
    float3 radiance = parameters.environmentPixels[y * parameters.environmentWidth + x].xyz;
    return max(luminance(radiance), 1e-8) *
           float(parameters.environmentWidth * parameters.environmentHeight) /
           max(parameters.environmentImportanceTotal * 2.0 * PI * PI, 1e-8);
}

struct EnvironmentLightSample
{
    float3 direction;
    float3 radiance;
    float pdf;
};

EnvironmentLightSample sampleHdrImportance(LaunchParameters parameters, float2 randomSample)
{
    EnvironmentLightSample sample;
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0);
    sample.pdf = 0.0;
    if (parameters.environmentWidth == 0 || parameters.environmentHeight == 0 ||
        parameters.environmentImportanceTotal <= 0.0)
        return sample;

    uint y = sampleCdf(parameters.environmentMarginalCdf, 0, parameters.environmentHeight, randomSample.y);
    float yFraction = remapCdfSample(parameters.environmentMarginalCdf, 0, y, randomSample.y);
    uint rowOffset = y * (parameters.environmentWidth + 1);
    uint x = sampleCdf(parameters.environmentConditionalCdf, rowOffset, parameters.environmentWidth, randomSample.x);
    float xFraction = remapCdfSample(parameters.environmentConditionalCdf, rowOffset, x, randomSample.x);
    float2 uv = float2((float(x) + xFraction) / float(parameters.environmentWidth),
                       (float(y) + yFraction) / float(parameters.environmentHeight));
    float phi = (uv.x - 0.5) * 2.0 * PI;
    float theta = uv.y * PI;
    float sineTheta = sin(theta);
    float3 localDirection = float3(cos(phi) * sineTheta, cos(theta), sin(phi) * sineTheta);
    sample.direction = inverseRotateEnvironmentDirection(localDirection, parameters.environmentRotation,
                                                         parameters.lightDirection.xyz);
    sample.radiance = sampleHdrEnvironment(parameters, sample.direction) * parameters.environmentIntensity;
    sample.pdf = hdrEnvironmentPdf(parameters, sample.direction);
    return sample;
}

float environmentLightPower(LaunchParameters parameters)
{
    if (parameters.globalLightMode == 1)
        return max(parameters.environmentImportanceTotal * parameters.environmentIntensity, 0.0);
    return 0.0;
}

float analyticLightPower(LaunchParameters parameters)
{
    if (parameters.globalLightMode == 1)
        return 0.0;
    uint count;
    uint stride;
    parameters.lights.GetDimensions(count, stride);
    float power = 0.0;
    for (uint index = 0u; index < count; ++index)
    {
        GpuLight light = parameters.lights[index];
        float contribution = max(luminance(light.colorAndIntensity.xyz) * light.colorAndIntensity.w, 0.0);
        if (uint(light.positionAndType.w + 0.5) == 3u)
            contribution *= max(light.coneAndArea.z * light.coneAndArea.w, 1e-6);
        power += contribution;
    }
    return power;
}

float allLightPower(LaunchParameters parameters)
{
    return environmentLightPower(parameters) + max(parameters.emissiveLightPower, 0.0) +
           analyticLightPower(parameters);
}

float environmentLightSelectionProbability(LaunchParameters parameters)
{
    return environmentLightPower(parameters) / max(allLightPower(parameters), 1e-8);
}

float emissiveLightSelectionProbability(LaunchParameters parameters)
{
    return max(parameters.emissiveLightPower, 0.0) / max(allLightPower(parameters), 1e-8);
}

uint sampleEmissiveTriangle(LaunchParameters parameters, float randomValue)
{
    uint count;
    uint stride;
    parameters.emissiveTriangles.GetDimensions(count, stride);
    uint low = 0;
    uint high = count;
    while (low < high)
    {
        uint middle = (low + high) / 2;
        if (randomValue < parameters.emissiveTriangles[middle].cdfAndPower.y)
            high = middle;
        else
            low = middle + 1;
    }
    return min(low, count - 1);
}

struct MeshLightSample
{
    float3 direction;
    float3 radiance;
    float distance;
    float pdf;
};

MeshLightSample sampleEmissiveMesh(LaunchParameters parameters, float3 position, float3 normal,
                                   float3 randomSample)
{
    MeshLightSample sample;
    sample.direction = normal;
    sample.radiance = float3(0.0);
    sample.distance = 0.0;
    sample.pdf = 0.0;
    uint count;
    uint stride;
    parameters.emissiveTriangles.GetDimensions(count, stride);
    if (count == 0 || parameters.emissiveLightPower <= 0.0)
        return sample;
    uint triangleIndex = sampleEmissiveTriangle(parameters, randomSample.x);
    GpuEmissiveTriangle triangle = parameters.emissiveTriangles[triangleIndex];
    float squareRoot = sqrt(randomSample.y);
    float barycentric0 = 1.0 - squareRoot;
    float barycentric1 = squareRoot * (1.0 - randomSample.z);
    float barycentric2 = squareRoot * randomSample.z;
    float3 lightPosition = triangle.position0.xyz * barycentric0 +
                           triangle.position1.xyz * barycentric1 +
                           triangle.position2.xyz * barycentric2;
    float3 toLight = lightPosition - position;
    float distanceSquared = dot(toLight, toLight);
    sample.distance = sqrt(max(distanceSquared, 1e-12));
    sample.direction = toLight / sample.distance;
    float3 lightNormal = normalize(cross(triangle.position1.xyz - triangle.position0.xyz,
                                         triangle.position2.xyz - triangle.position0.xyz));
    float signedCosineAtLight = dot(lightNormal, -sample.direction);
    bool doubleSided = triangle.materialData.y != 0u;
    float cosineAtLight = doubleSided ? abs(signedCosineAtLight) : max(signedCosineAtLight, 0.0);
    if (cosineAtLight <= 1e-6)
        return sample;
    float selectionPdf = triangle.cdfAndPower.z / max(parameters.emissiveLightPower, 1e-8);
    float areaPdf = selectionPdf / max(triangle.emissionAndArea.w, 1e-8);
    sample.pdf = areaPdf * distanceSquared / max(cosineAtLight, 1e-6);
    float2 uv = triangle.uv0Uv1.xy * barycentric0 + triangle.uv0Uv1.zw * barycentric1 +
                triangle.uv2AndPadding.xy * barycentric2;
    float2 uv1 = triangle.uvSet1_0_1.xy * barycentric0 + triangle.uvSet1_0_1.zw * barycentric1 +
                 triangle.uvSet1_2AndPadding.xy * barycentric2;
    float textureLod = log2(max(sample.distance * 0.05, 1.0));
    float3 emissiveTexture = sampleMaterialTexture(parameters, triangle.materialData.x, float4(uv, uv1),
                                                    float4(1.0), textureLod).xyz;
    sample.radiance = triangle.emissionAndArea.xyz * max(emissiveTexture, 0.0);
    return sample;
}

struct AnalyticLightSample
{
    float3 direction;
    float3 radiance;
    float distance;
    float selectionPdf;
    float solidAnglePdf;
    bool delta;
};

AnalyticLightSample sampleAnalyticLight(LaunchParameters parameters, float3 position, float3 randomSample)
{
    AnalyticLightSample sample;
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0);
    sample.distance = 0.0;
    sample.selectionPdf = 0.0;
    sample.solidAnglePdf = 0.0;
    sample.delta = true;
    float totalPower = analyticLightPower(parameters);
    if (totalPower <= 0.0)
        return sample;
    uint count;
    uint stride;
    parameters.lights.GetDimensions(count, stride);
    float target = randomSample.x * totalPower;
    float accumulated = 0.0;
    GpuLight light = parameters.lights[0];
    float selectedPower = 0.0;
    for (uint index = 0u; index < count; ++index)
    {
        GpuLight candidate = parameters.lights[index];
        float power = max(luminance(candidate.colorAndIntensity.xyz) * candidate.colorAndIntensity.w, 0.0);
        if (uint(candidate.positionAndType.w + 0.5) == 3u)
            power *= max(candidate.coneAndArea.z * candidate.coneAndArea.w, 1e-6);
        accumulated += power;
        if (target <= accumulated || index + 1u == count)
        {
            light = candidate;
            selectedPower = power;
            break;
        }
    }
    sample.selectionPdf = selectedPower / max(totalPower, 1e-8);
    uint type = uint(light.positionAndType.w + 0.5);
    sample.radiance = max(light.colorAndIntensity.xyz, 0.0) * max(light.colorAndIntensity.w, 0.0);
    if (type == 0u)
    {
        sample.direction = normalize(-light.directionAndRange.xyz);
        sample.distance = 10000.0;
        sample.solidAnglePdf = 1.0;
        return sample;
    }
    float3 lightPosition = light.positionAndType.xyz;
    if (type == 3u)
    {
        float3 lightNormal = normalize(light.directionAndRange.xyz);
        float3 helper = abs(lightNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        float3 tangent = normalize(cross(helper, lightNormal));
        float3 bitangent = cross(lightNormal, tangent);
        lightPosition += tangent * (randomSample.y - 0.5) * light.coneAndArea.z +
                         bitangent * (randomSample.z - 0.5) * light.coneAndArea.w;
        sample.delta = false;
    }
    float3 toLight = lightPosition - position;
    float distanceSquared = max(dot(toLight, toLight), 1e-8);
    sample.distance = sqrt(distanceSquared);
    sample.direction = toLight / sample.distance;
    float range = max(light.directionAndRange.w, 1e-4);
    float rangeAttenuation = saturate(1.0 - pow(sample.distance / range, 4.0));
    rangeAttenuation *= rangeAttenuation;
    if (type == 3u)
    {
        float cosineAtLight = max(dot(normalize(light.directionAndRange.xyz), -sample.direction), 0.0);
        float area = max(light.coneAndArea.z * light.coneAndArea.w, 1e-8);
        sample.solidAnglePdf = distanceSquared / max(area * cosineAtLight, 1e-8);
        sample.radiance *= rangeAttenuation;
    }
    else
    {
        sample.solidAnglePdf = 1.0;
        sample.radiance *= rangeAttenuation / distanceSquared;
        if (type == 2u)
        {
            float coneCosine = dot(normalize(light.directionAndRange.xyz), -sample.direction);
            sample.radiance *= smoothstep(light.coneAndArea.y, light.coneAndArea.x, coneCosine);
        }
    }
    return sample;
}

uint toneMapAndPack(float3 color, float exposure, uint displayBgra)
{
    float3 mapped = max(color * exp2(exposure), 0.0);
    mapped = mapped / (1.0f + mapped);
    mapped = float3(mapped.x <= 0.0031308f ? 12.92f * mapped.x : 1.055f * pow(mapped.x, 0.41666667f) - 0.055f,
                    mapped.y <= 0.0031308f ? 12.92f * mapped.y : 1.055f * pow(mapped.y, 0.41666667f) - 0.055f,
                    mapped.z <= 0.0031308f ? 12.92f * mapped.z : 1.055f * pow(mapped.z, 0.41666667f) - 0.055f);
    uint3 rgb = uint3(clamp(mapped * 255.0f + 0.5f, 0.0f, 255.0f));
    return displayBgra != 0 ? rgb.z | (rgb.y << 8) | (rgb.x << 16) | 0xff000000
                            : rgb.x | (rgb.y << 8) | (rgb.z << 16) | 0xff000000;
}

[shader("raygeneration")]
void RayGen(LaunchParameters parameters)
{
    uint3 launchIndex = DispatchRaysIndex();
    if (any(launchIndex.xy >= parameters.outputSize))
        return;

    float3 forward = normalize(parameters.cameraTargetAndAspect.xyz - parameters.cameraPositionAndFov.xyz);
    float3 right = normalize(cross(forward, parameters.cameraUp.xyz));
    float3 up = normalize(cross(right, forward));
    float tanHalfFov = tan(parameters.cameraPositionAndFov.w * 0.5);

    uint linearIndex = launchIndex.y * parameters.outputSize.x + launchIndex.x;
    uint sampleCount = max(parameters.samplesPerFrame, 1u);
    float3 sampleSum = float3(0.0);
    float3 guideAlbedo = float3(0.0);
    float3 guideNormal = float3(0.0);
    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        uint seed = linearIndex * 9781u + (parameters.frameIndex * sampleCount + sampleIndex) * 6271u;
        float2 jitter = float2(random01(seed), random01(seed ^ 0x68bc21ebu));
        float2 pixel = (float2(launchIndex.xy) + jitter) / float2(parameters.outputSize);
        float2 screen = pixel * 2.0 - 1.0;

        RayDesc ray;
        ray.Origin = parameters.cameraPositionAndFov.xyz;
        ray.Direction = normalize(forward + right * screen.x * parameters.cameraTargetAndAspect.w * tanHalfFov -
                                  up * screen.y * tanHalfFov);
        ray.TMin = 0.001;
        ray.TMax = 10000.0;

        float3 sampleRadiance = float3(0.0);
        float3 throughput = float3(1.0);
        float previousBsdfPdf = 0.0;
        bool previousBsdfDelta = true;
        float mediumIors[8];
        float3 mediumAbsorption[8];
        float3 mediumVolumeAbsorption[8];
        float3 mediumScattering[8];
        float mediumDensity[8];
        float mediumAnisotropy[8];
        uint mediumDepth = 1;
        uint diagnosticPathDepth = 0;
        bool subsurfaceActive = false;
        float3 subsurfaceWalkColor = float3(1.0);
        float subsurfaceWalkRadius = 1.0;
        uint subsurfaceInstanceIndex = 0xffffffffu;
        uint subsurfaceWalkSteps = 0u;
        float rayConeWidth = 0.0;
        float rayConeSpread = 2.0 * tanHalfFov / float(max(parameters.outputSize.y, 1u));
        mediumIors[0] = 1.0;
        mediumAbsorption[0] = float3(0.0);
        mediumVolumeAbsorption[0] = float3(0.0);
        mediumScattering[0] = float3(0.0);
        mediumDensity[0] = 0.0;
        mediumAnisotropy[0] = 0.0;
        uint bounceLimit = (parameters.rayTracedReflections != 0 || parameters.indirectLighting != 0)
                               ? max(parameters.maxBounces, 1u) : 1u;
        for (uint bounce = 0; bounce < bounceLimit; ++bounce)
        {
            diagnosticPathDepth = bounce + 1u;
            RadiancePayload payload;
            ResolvedHit resolvedHit;
            RayDesc traversalRay = ray;
            float accumulatedDistance = 0.0;
            bool resolvedSurface = false;
            for (uint transparencyLayer = 0; transparencyLayer < 32u; ++transparencyLayer)
            {
                payload.primitiveIndex = 0;
                payload.instanceIndex = 0;
                payload.barycentrics = float2(0.0);
                payload.hitDistance = 0.0;
                payload.hit = 0;
                TraceRay(parameters.scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, traversalRay, payload);
                if (payload.hit == 0)
                    break;
                accumulatedDistance += payload.hitDistance;
                float textureFootprint = rayConeWidth + accumulatedDistance * rayConeSpread;
                float textureLod = log2(max(textureFootprint * 1024.0, 1.0));
                resolvedHit = resolveHit(parameters, payload, traversalRay, textureLod);
                float coverageSample = random01(seed ^ ((bounce + 1u) * 0x8da6b343u) ^
                                                ((transparencyLayer + 1u) * 0xd8163841u));
                if (!skipCameraPathHit(resolvedHit, traversalRay.Direction, coverageSample))
                {
                    payload.hitDistance = accumulatedDistance;
                    resolvedSurface = true;
                    break;
                }
                float alphaCoordinateScale = max(max(abs(resolvedHit.position.x), abs(resolvedHit.position.y)),
                                                 abs(resolvedHit.position.z));
                float alphaRayBias = max(0.0001, alphaCoordinateScale * 0.00001);
                traversalRay.Origin = resolvedHit.position + traversalRay.Direction * alphaRayBias;
                traversalRay.TMin = alphaRayBias * 0.5;
                traversalRay.TMax = max(traversalRay.TMax - payload.hitDistance - alphaRayBias,
                                        traversalRay.TMin);
            }
            if (!resolvedSurface)
                payload.hit = 0;
            float transportDistance = payload.hit != 0 ? payload.hitDistance : ray.TMax;
            if (subsurfaceActive)
            {
                if (payload.hit != 0 && resolvedHit.instanceIndex != subsurfaceInstanceIndex)
                    break;
                float extinction = 1.0 / max(subsurfaceWalkRadius, 1e-4);
                float freeFlightRandom = random01(seed ^ ((bounce + 1u) * 0xb5297a4du) ^
                                                  ((subsurfaceWalkSteps + 1u) * 0x68e31da4u));
                float freeFlightDistance = -log(max(1.0 - freeFlightRandom, 1e-7)) / extinction;
                if (freeFlightDistance < transportDistance)
                {
                    float3 scatterPosition = ray.Origin + ray.Direction * freeFlightDistance;
                    throughput *= saturate(subsurfaceWalkColor);
                    ++subsurfaceWalkSteps;
                    if (subsurfaceWalkSteps >= 3u)
                    {
                        float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.2, 0.9);
                        if (random01(seed ^ ((bounce + 1u) * 0x1b56c4e9u)) > survival)
                            break;
                        throughput /= survival;
                    }
                    float3 scatterDirection = sampleHenyeyGreenstein(
                        ray.Direction, 0.0,
                        float2(random01(seed ^ ((bounce + 1u) * 0x7f4a7c15u)),
                               random01(seed ^ ((bounce + 1u) * 0x94d049bbu))));
                    rayConeWidth += freeFlightDistance * rayConeSpread;
                    rayConeSpread = max(rayConeSpread, 0.25);
                    previousBsdfPdf = 1.0 / (4.0 * PI);
                    previousBsdfDelta = true;
                    ray.Origin = scatterPosition + scatterDirection * 0.0001;
                    ray.Direction = scatterDirection;
                    ray.TMin = 0.00005;
                    ray.TMax = 10000.0;
                    continue;
                }
                subsurfaceActive = false;
                previousBsdfDelta = true;
                if (payload.hit != 0)
                {
                    float exitScale = max(max(abs(resolvedHit.position.x), abs(resolvedHit.position.y)),
                                          abs(resolvedHit.position.z));
                    float exitBias = max(0.0001, exitScale * 0.00001);
                    rayConeWidth += transportDistance * rayConeSpread;
                    ray.Origin = resolvedHit.position + ray.Direction * exitBias;
                    ray.TMin = exitBias * 0.5;
                    ray.TMax = 10000.0;
                    continue;
                }
            }
            if (mediumDepth > 1)
            {
                float3 sigmaA = mediumAbsorption[mediumDepth - 1] +
                                mediumVolumeAbsorption[mediumDepth - 1] * mediumDensity[mediumDepth - 1];
                float3 sigmaS = mediumScattering[mediumDepth - 1] * mediumDensity[mediumDepth - 1];
                float3 sigmaT = sigmaA + sigmaS;
                float majorant = max(sigmaT.x, max(sigmaT.y, sigmaT.z));
                if (majorant > 1e-6)
                {
                    float freeFlightRandom = random01(seed ^ ((bounce + 1) * 0xa24baed5u));
                    float freeFlightDistance = -log(max(1.0 - freeFlightRandom, 1e-7)) / majorant;
                    bool scatteringMedium = max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > 0.0;
                    if (freeFlightDistance < transportDistance && scatteringMedium)
                    {
                        float3 volumePosition = ray.Origin + ray.Direction * freeFlightDistance;
                        float proposalPdf = majorant * exp(-majorant * freeFlightDistance);
                        throughput *= exp(-sigmaT * freeFlightDistance) * sigmaS /
                                      max(proposalPdf, 1e-20);
                        float environmentProbability = environmentLightSelectionProbability(parameters);
                        float emissiveProbability = emissiveLightSelectionProbability(parameters);
                        float analyticProbability = analyticLightPower(parameters) /
                                                    max(allLightPower(parameters), 1e-8);
                        float volumeLightSelector = random01(seed ^ ((bounce + 1) * 0x6c8e9cf5u));
                        float3 volumeLightDirection = float3(0.0, 1.0, 0.0);
                        float3 volumeLightRadiance = float3(0.0);
                        float volumeLightPdf = 0.0;
                        float volumeLightDistance = 10000.0;
                        bool volumeLightDelta = false;
                        if (volumeLightSelector < environmentProbability)
                        {
                            EnvironmentLightSample environmentLight = sampleHdrImportance(
                                parameters,
                                float2(random01(seed ^ ((bounce + 1) * 0x9e3779b9u)),
                                       random01(seed ^ ((bounce + 1) * 0x7f4a7c15u))));
                            volumeLightDirection = environmentLight.direction;
                            volumeLightRadiance = environmentLight.radiance;
                            volumeLightPdf = environmentLight.pdf * environmentProbability;
                        }
                        else if (volumeLightSelector < environmentProbability + emissiveProbability)
                        {
                            MeshLightSample meshLight = sampleEmissiveMesh(
                                parameters, volumePosition, -ray.Direction,
                                float3(random01(seed ^ ((bounce + 1) * 0x85ebca6bu)),
                                       random01(seed ^ ((bounce + 1) * 0xc2b2ae35u)),
                                       random01(seed ^ ((bounce + 1) * 0x27d4eb2fu))));
                            volumeLightDirection = meshLight.direction;
                            volumeLightRadiance = meshLight.radiance;
                            volumeLightDistance = meshLight.distance;
                            volumeLightPdf = meshLight.pdf * emissiveProbability;
                        }
                        else if (analyticProbability > 0.0)
                        {
                            AnalyticLightSample analyticLight = sampleAnalyticLight(
                                parameters,
                                volumePosition,
                                float3(random01(seed ^ ((bounce + 1) * 0x9e3779b9u)),
                                       random01(seed ^ ((bounce + 1) * 0x7f4a7c15u)),
                                       random01(seed ^ ((bounce + 1) * 0x27d4eb2fu))));
                            volumeLightDirection = analyticLight.direction;
                            volumeLightRadiance = analyticLight.radiance;
                            volumeLightDistance = analyticLight.distance;
                            volumeLightPdf = analyticProbability * analyticLight.selectionPdf *
                                             analyticLight.solidAnglePdf;
                            volumeLightDelta = analyticLight.delta;
                        }
                        if (volumeLightPdf > 0.0)
                        {
                            RayDesc volumeShadowRay;
                            volumeShadowRay.Origin = volumePosition + volumeLightDirection * 0.0001;
                            volumeShadowRay.Direction = volumeLightDirection;
                            volumeShadowRay.TMin = 0.00005;
                            volumeShadowRay.TMax = max(volumeLightDistance - 0.0001, 0.00005);
                            float3 volumeVisibility = parameters.rayTracedShadows != 0
                                                          ? traceVisibility(parameters, volumeShadowRay)
                                                          : float3(1.0);
                            if (max(volumeVisibility.x, max(volumeVisibility.y, volumeVisibility.z)) > 0.0)
                            {
                                float phase = henyeyGreenstein(dot(ray.Direction, volumeLightDirection),
                                                               mediumAnisotropy[mediumDepth - 1]);
                                float misWeight = volumeLightDelta ? 1.0 :
                                    misPowerHeuristic(volumeLightPdf, phase);
                                sampleRadiance += throughput * volumeLightRadiance * volumeVisibility * phase *
                                                  misWeight / volumeLightPdf;
                            }
                        }
                        float3 phaseDirection = sampleHenyeyGreenstein(
                            ray.Direction, mediumAnisotropy[mediumDepth - 1],
                            float2(random01(seed ^ ((bounce + 1) * 0x94d049bbu)),
                                   random01(seed ^ ((bounce + 1) * 0x369dea0fu))));
                        previousBsdfPdf = henyeyGreenstein(dot(ray.Direction, phaseDirection),
                                                           mediumAnisotropy[mediumDepth - 1]);
                        previousBsdfDelta = false;
                        rayConeWidth += freeFlightDistance * rayConeSpread;
                        rayConeSpread = max(rayConeSpread, 0.25);
                        ray.Origin = volumePosition + phaseDirection * 0.0001;
                        ray.Direction = phaseDirection;
                        ray.TMin = 0.00005;
                        ray.TMax = 10000.0;
                        continue;
                    }
                    float3 transmittance = exp(-sigmaT * transportDistance);
                    if (scatteringMedium)
                        transmittance /= max(exp(-majorant * transportDistance), 1e-20);
                    throughput *= transmittance;
                }
            }
            if (payload.hit == 0)
            {
                if (bounce != 0 || parameters.environmentVisible != 0)
                {
                    float3 environmentRadiance = sampleGlobalEnvironment(parameters, ray.Direction);
                    float misWeight = 1.0;
                    if (bounce > 0 && !previousBsdfDelta && parameters.globalLightMode == 1)
                        misWeight = misPowerHeuristic(previousBsdfPdf,
                                                     hdrEnvironmentPdf(parameters, ray.Direction) *
                                                         environmentLightSelectionProbability(parameters));
                    sampleRadiance += throughput * environmentRadiance * misWeight;
                }
                break;
            }

            float3 viewDirection = normalize(-ray.Direction);
            uint materialIndex = resolvedHit.materialIndex;
            GpuMaterial material = resolvedHit.material;
            float3 worldPosition = resolvedHit.position;
            SurfaceData surface = resolvedHit.surface;
            bool enteringSurface = dot(surface.geometricNormal, viewDirection) >= 0.0;
            float etaI = mediumIors[mediumDepth - 1];
            float etaT = enteringSurface ? surface.ior
                                         : (mediumDepth > 1 ? mediumIors[mediumDepth - 2] : 1.0);
            float3 normal = surface.shadingNormal;
            if (dot(normal, viewDirection) < 0.0)
            {
                normal = -normal;
                surface.shadingNormal = normal;
                surface.clearcoatNormal = -surface.clearcoatNormal;
                surface.bitangent = -surface.bitangent;
            }
            if (parameters.debugView != 0u && parameters.debugView != 19u)
            {
                sampleRadiance = surfaceDebugColor(parameters.debugView, surface, material, materialIndex,
                                                   surface.geometricNormal, viewDirection, mediumDepth);
                break;
            }
            if (bounce == 0 && sampleIndex == 0 && parameters.writeDisplay == 0)
            {
                guideAlbedo = surface.baseColor;
                guideNormal = normal;
            }
            float coordinateScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
            float rayBias = max(0.0001, coordinateScale * 0.00001);
            float emissiveMisWeight = 1.0;
            if (bounce > 0 && !previousBsdfDelta)
            {
                float lightPdf = emissiveMeshHitPdf(parameters, resolvedHit, ray.Direction,
                                                    payload.hitDistance);
                if (lightPdf > 0.0)
                    emissiveMisWeight = misPowerHeuristic(previousBsdfPdf, lightPdf);
            }
            sampleRadiance += throughput * surface.emissive * emissiveMisWeight;

            float3 sampledLightDirection = float3(0.0, 1.0, 0.0);
            float3 sampledLightRadiance = float3(0.0);
            float sampledLightPdf = 0.0;
            float sampledLightDistance = 10000.0;
            bool sampledLightDelta = true;
            float environmentSelectionProbability = environmentLightSelectionProbability(parameters);
            float emissiveSelectionProbability = emissiveLightSelectionProbability(parameters);
            float analyticSelectionProbability = analyticLightPower(parameters) /
                                                 max(allLightPower(parameters), 1e-8);
            float lightSelector = random01(seed ^ ((bounce + 1) * 0x6c8e9cf5u));
            if (lightSelector < environmentSelectionProbability)
            {
                EnvironmentLightSample environmentSample = sampleHdrImportance(
                    parameters,
                    float2(random01(seed ^ ((bounce + 1) * 0x2c1b3c6du)),
                           random01(seed ^ ((bounce + 1) * 0x297a2d39u))));
                sampledLightDirection = environmentSample.direction;
                sampledLightRadiance = environmentSample.radiance;
                sampledLightPdf = environmentSample.pdf * environmentSelectionProbability;
                sampledLightDelta = false;
            }
            else if (lightSelector < environmentSelectionProbability + emissiveSelectionProbability)
            {
                MeshLightSample meshLightSample = sampleEmissiveMesh(
                    parameters, worldPosition, normal,
                    float3(random01(seed ^ ((bounce + 1) * 0x85ebca6bu)),
                           random01(seed ^ ((bounce + 1) * 0xc2b2ae35u)),
                           random01(seed ^ ((bounce + 1) * 0x27d4eb2fu))));
                sampledLightDirection = meshLightSample.direction;
                sampledLightRadiance = meshLightSample.radiance;
                sampledLightDistance = meshLightSample.distance;
                sampledLightPdf = meshLightSample.pdf * emissiveSelectionProbability;
                sampledLightDelta = false;
            }
            else if (analyticSelectionProbability > 0.0)
            {
                AnalyticLightSample analyticSample = sampleAnalyticLight(
                    parameters,
                    worldPosition,
                    float3(random01(seed ^ ((bounce + 1) * 0x2c1b3c6du)),
                           random01(seed ^ ((bounce + 1) * 0x297a2d39u)),
                           random01(seed ^ ((bounce + 1) * 0x94d049bbu))));
                sampledLightDirection = analyticSample.direction;
                sampledLightRadiance = analyticSample.radiance;
                sampledLightDistance = analyticSample.distance;
                sampledLightPdf = analyticSelectionProbability * analyticSample.selectionPdf *
                                  analyticSample.solidAnglePdf;
                sampledLightDelta = analyticSample.delta;
            }

            float3 visibility = float3(1.0);
            float signedNDotLight = dot(normal, sampledLightDirection);
            if (parameters.rayTracedShadows != 0 && sampledLightPdf > 0.0)
            {
                RayDesc shadowRay;
                shadowRay.Origin = worldPosition + (signedNDotLight >= 0.0 ? normal : -normal) * rayBias;
                shadowRay.Direction = sampledLightDirection;
                shadowRay.TMin = rayBias * 0.5;
                shadowRay.TMax = max(sampledLightDistance - rayBias, rayBias);

                visibility = traceVisibility(parameters, shadowRay);
            }
            float nDotLight = abs(signedNDotLight);
            if (max(visibility.x, max(visibility.y, visibility.z)) > 0.0 &&
                sampledLightPdf > 0.0 && nDotLight > 0.0)
            {
                float3 bsdfValue = evaluateBsdfWithSubsurfaceMix(surface, viewDirection,
                                                                  sampledLightDirection, etaI, etaT);
                float bsdfPdf = evaluateBsdfPdfWithIors(surface, viewDirection, sampledLightDirection,
                                                        etaI, etaT);
                float misWeight = sampledLightDelta ? 1.0 : misPowerHeuristic(sampledLightPdf, bsdfPdf);
                sampleRadiance += throughput * bsdfValue * sampledLightRadiance *
                                  (nDotLight * visibility * misWeight / sampledLightPdf);
            }
            if (bounce + 1 >= bounceLimit)
                break;

            float subsurfaceSelector = random01(seed ^ ((bounce + 1) * 0x165667b1u));
            if (enteringSurface && surface.subsurface > 0.0 &&
                subsurfaceSelector < surface.subsurface && parameters.indirectLighting != 0)
            {
                float3 inwardDirection = sampleSubsurfaceInwardDirection(
                    surface,
                    float2(random01(seed ^ ((bounce + 1) * 0xd3a2646cu)),
                           random01(seed ^ ((bounce + 1) * 0xfd7046c5u))));
                throughput *= surface.baseColor;
                subsurfaceWalkColor = surface.subsurfaceColor;
                subsurfaceWalkRadius = surface.subsurfaceRadius;
                subsurfaceActive = true;
                subsurfaceInstanceIndex = resolvedHit.instanceIndex;
                subsurfaceWalkSteps = 0u;
                previousBsdfPdf = abs(dot(normal, inwardDirection)) / PI;
                previousBsdfDelta = true;
                rayConeWidth += payload.hitDistance * rayConeSpread;
                rayConeSpread = max(rayConeSpread, 0.25);
                ray.Origin = worldPosition - normal * rayBias;
                ray.Direction = inwardDirection;
                ray.TMin = rayBias * 0.5;
                ray.TMax = 10000.0;
                continue;
            }

            float lobeRandom = random01(seed ^ ((bounce + 1) * 0x51ed270bu));
            bool enableDiffuseTransport = parameters.indirectLighting != 0;
            bool enableReflectionTransport = parameters.rayTracedReflections != 0;
            bool enableTransmissionTransport = parameters.indirectLighting != 0;
            BsdfSample bsdfSample = sampleBsdfWithIorsControlled(
                surface, viewDirection,
                float2(random01(seed ^ ((bounce + 1) * 0x68bc21ebu)),
                       random01(seed ^ ((bounce + 1) * 0x02e5be93u))),
                lobeRandom, etaI, etaT, enableDiffuseTransport, enableReflectionTransport,
                enableTransmissionTransport);
            if (bsdfSample.pdf <= 0.0 || max(bsdfSample.weight.x, max(bsdfSample.weight.y,
                                                                      bsdfSample.weight.z)) <= 0.0)
                break;
            throughput *= bsdfSample.weight;
            previousBsdfPdf = bsdfSample.pdf;
            previousBsdfDelta = bsdfSample.delta;
            rayConeWidth += payload.hitDistance * rayConeSpread;
            rayConeSpread = max(rayConeSpread, bsdfSample.transmission ? surface.roughness * 0.15
                                                                       : surface.roughness * 0.25);
            if (bsdfSample.transmission)
            {
                if (enteringSurface && mediumDepth < 8)
                {
                    mediumIors[mediumDepth] = surface.ior;
                    mediumAbsorption[mediumDepth] = absorptionCoefficient(surface);
                    mediumVolumeAbsorption[mediumDepth] = surface.volumeAbsorption;
                    mediumScattering[mediumDepth] = surface.volumeScattering;
                    mediumDensity[mediumDepth] = surface.volumeDensity;
                    mediumAnisotropy[mediumDepth] = surface.volumeAnisotropy;
                    ++mediumDepth;
                }
                else if (!enteringSurface && mediumDepth > 1)
                    --mediumDepth;
            }
            float maximumThroughput = max(throughput.x, max(throughput.y, throughput.z));
            if (maximumThroughput < 0.002)
                break;
            if (bounce >= 2)
            {
                float survival = clamp(maximumThroughput, 0.1, 0.95);
                if (random01(seed ^ ((bounce + 1) * 0x9e3779b9u)) > survival)
                    break;
                throughput /= survival;
            }

            float3 offsetNormal = dot(bsdfSample.direction, normal) >= 0.0 ? normal : -normal;
            ray.Origin = worldPosition + offsetNormal * rayBias;
            ray.Direction = bsdfSample.direction;
            ray.TMin = rayBias * 0.5;
            ray.TMax = 10000.0;
        }
        if (parameters.debugView == 19u)
        {
            float depth = float(diagnosticPathDepth) / float(max(bounceLimit, 1u));
            sampleRadiance = float3(depth, depth * depth, 1.0 - depth);
        }
        sampleSum += sampleRadiance;
    }

    float3 current = sampleSum / float(sampleCount);
    if (parameters.accumulatedSamples > 0)
    {
        float3 previous = parameters.output[linearIndex].xyz;
        float weight = float(sampleCount) / float(parameters.accumulatedSamples + sampleCount);
        current = lerp(previous, current, weight);
    }
    parameters.output[linearIndex] = float4(current, 1.0);
    if (parameters.writeDisplay == 0)
    {
        parameters.albedoGuide[linearIndex] = float4(guideAlbedo, 1.0);
        parameters.normalGuide[linearIndex] = float4(guideNormal, 0.0);
    }
    else
        parameters.displayOutput[linearIndex] = toneMapAndPack(current, parameters.exposure, parameters.displayBgra);
}

[shader("raygeneration")]
void ToneMap(LaunchParameters parameters)
{
    uint3 launchIndex = DispatchRaysIndex();
    if (any(launchIndex.xy >= parameters.outputSize))
        return;
    uint linearIndex = launchIndex.y * parameters.outputSize.x + launchIndex.x;
    parameters.displayOutput[linearIndex] = toneMapAndPack(float3(parameters.denoisedOutput[linearIndex].xyz),
                                                           parameters.exposure,
                                                           parameters.displayBgra);
}

[shader("miss")]
void Miss(inout RadiancePayload payload)
{
    payload.hit = 0;
}

[shader("closesthit")]
void ClosestHit(inout RadiancePayload payload,
                BuiltInTriangleIntersectionAttributes attributes)
{
    payload.primitiveIndex = PrimitiveIndex();
    payload.instanceIndex = InstanceID();
    payload.barycentrics = attributes.barycentrics;
    payload.hitDistance = RayTCurrent();
    payload.hit = 1;
}
