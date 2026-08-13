import MaterialTypes;
import MaterialSampling;
import Pbr;
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
    StructuredBuffer<float3> normals;
    StructuredBuffer<float4> tangents;
    StructuredBuffer<float2> uvs;
    StructuredBuffer<uint3> indices;
    StructuredBuffer<GpuOptixInstance> instances;
    StructuredBuffer<GpuMaterial> materials;
    StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;
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
    uint environmentMipCount;
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

float4 sampleMaterialTexture(LaunchParameters parameters, uint textureId, float2 uv, float4 fallback)
{
    return textureId == INVALID_TEXTURE_ID
               ? fallback
               : parameters.materialTextureTable.textures[textureId].SampleLevel(
                     parameters.materialTextureTable.sampler, uv, 0.0);
}

SurfaceData resolveMaterialSurface(LaunchParameters parameters, GpuMaterial material, float3 position,
                                   float3 geometricNormal, float4 tangentAndSign, float2 uv)
{
    SurfaceData surface = makeSurfaceData(material, position, geometricNormal, geometricNormal, tangentAndSign, uv);
    MaterialTextureSamples samples = defaultMaterialTextureSamples();
    samples.baseColor = sampleMaterialTexture(parameters, material.textureIndices0.x, uv, samples.baseColor);
    samples.metallicRoughness = sampleMaterialTexture(parameters, material.textureIndices0.y, uv,
                                                      samples.metallicRoughness);
    samples.normal = sampleMaterialTexture(parameters, material.textureIndices0.z, uv, samples.normal);
    samples.emissive = sampleMaterialTexture(parameters, material.textureIndices0.w, uv, samples.emissive);
    samples.occlusion = sampleMaterialTexture(parameters, material.textureIndices1.x, uv, samples.occlusion);
    samples.transmission = sampleMaterialTexture(parameters, material.textureIndices1.y, uv, samples.transmission);
    samples.clearcoat = sampleMaterialTexture(parameters, material.textureIndices1.z, uv, samples.clearcoat);
    samples.clearcoatRoughness = sampleMaterialTexture(parameters, material.textureIndices1.w, uv,
                                                       samples.clearcoatRoughness);
    samples.clearcoatNormal = sampleMaterialTexture(parameters, material.textureIndices2.x, uv,
                                                    samples.clearcoatNormal);
    samples.anisotropy = sampleMaterialTexture(parameters, material.textureIndices2.w, uv, samples.anisotropy);
    samples.sheenColor = sampleMaterialTexture(parameters, material.textureIndices2.y, uv, samples.sheenColor);
    samples.sheenRoughness = sampleMaterialTexture(parameters, material.textureIndices2.z, uv,
                                                   samples.sheenRoughness);
    return applyBaseMaterialTextures(material, surface, samples);
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
                material.textureIndices0.z == INVALID_TEXTURE_ID) ? float3(0.05) :
               debugIdColor(material.textureIndices0.x ^ (material.textureIndices0.y * 31u) ^
                            (material.textureIndices0.z * 131u));
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
    return max(luminance(parameters.lightColorAndIntensity.xyz) * parameters.lightColorAndIntensity.w, 0.0);
}

float environmentLightSelectionProbability(LaunchParameters parameters)
{
    float environmentPower = environmentLightPower(parameters);
    return environmentPower / max(environmentPower + parameters.emissiveLightPower, 1e-8);
}

uint sampleEmissiveTriangle(LaunchParameters parameters, float randomValue)
{
    uint count;
    uint stride;
    parameters.emissiveTriangles.GetDimensions(count, stride);
    uint low = 0;
    uint high = count;
    while (low + 1 < high)
    {
        uint middle = (low + high) / 2;
        if (parameters.emissiveTriangles[middle].cdfAndPower.y <= randomValue)
            low = middle + 1;
        else
            high = middle;
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
    float cosineAtLight = abs(dot(lightNormal, -sample.direction));
    float selectionPdf = triangle.cdfAndPower.z / max(parameters.emissiveLightPower, 1e-8);
    float areaPdf = selectionPdf / max(triangle.emissionAndArea.w, 1e-8);
    sample.pdf = areaPdf * distanceSquared / max(cosineAtLight, 1e-6);
    sample.radiance = triangle.emissionAndArea.xyz;
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
            payload.primitiveIndex = 0;
            payload.instanceIndex = 0;
            payload.barycentrics = float2(0.0);
            payload.hitDistance = 0.0;
            payload.hit = 0;
            TraceRay(parameters.scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
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
                    if (freeFlightDistance < payload.hitDistance && max(sigmaS.x, max(sigmaS.y, sigmaS.z)) > 0.0)
                    {
                        float3 volumePosition = ray.Origin + ray.Direction * freeFlightDistance;
                        throughput *= exp(-sigmaT * freeFlightDistance) * sigmaS / majorant;
                        float environmentProbability = environmentLightSelectionProbability(parameters);
                        float volumeLightSelector = random01(seed ^ ((bounce + 1) * 0x6c8e9cf5u));
                        bool chooseEnvironment = volumeLightSelector < environmentProbability ||
                                                 parameters.emissiveLightPower <= 0.0;
                        float3 volumeLightDirection;
                        float3 volumeLightRadiance;
                        float volumeLightPdf;
                        float volumeLightDistance = 10000.0;
                        bool volumeLightDelta = false;
                        if (!chooseEnvironment)
                        {
                            MeshLightSample meshLight = sampleEmissiveMesh(
                                parameters, volumePosition, -ray.Direction,
                                float3(random01(seed ^ ((bounce + 1) * 0x85ebca6bu)),
                                       random01(seed ^ ((bounce + 1) * 0xc2b2ae35u)),
                                       random01(seed ^ ((bounce + 1) * 0x27d4eb2fu))));
                            volumeLightDirection = meshLight.direction;
                            volumeLightRadiance = meshLight.radiance;
                            volumeLightDistance = meshLight.distance;
                            volumeLightPdf = meshLight.pdf * (1.0 - environmentProbability);
                        }
                        else if (parameters.globalLightMode == 1)
                        {
                            EnvironmentLightSample environmentLight = sampleHdrImportance(
                                parameters,
                                float2(random01(seed ^ ((bounce + 1) * 0x9e3779b9u)),
                                       random01(seed ^ ((bounce + 1) * 0x7f4a7c15u))));
                            volumeLightDirection = environmentLight.direction;
                            volumeLightRadiance = environmentLight.radiance;
                            volumeLightPdf = environmentLight.pdf * environmentProbability;
                        }
                        else
                        {
                            volumeLightDirection = normalize(-parameters.lightDirection.xyz);
                            volumeLightRadiance = parameters.lightColorAndIntensity.xyz *
                                                  parameters.lightColorAndIntensity.w;
                            volumeLightPdf = max(environmentProbability, 1e-8);
                            volumeLightDelta = true;
                        }
                        if (volumeLightPdf > 0.0)
                        {
                            RayDesc volumeShadowRay;
                            volumeShadowRay.Origin = volumePosition + volumeLightDirection * 0.0001;
                            volumeShadowRay.Direction = volumeLightDirection;
                            volumeShadowRay.TMin = 0.00005;
                            volumeShadowRay.TMax = max(volumeLightDistance - 0.0001, 0.00005);
                            RadiancePayload volumeShadowPayload;
                            volumeShadowPayload.primitiveIndex = 0;
                            volumeShadowPayload.instanceIndex = 0;
                            volumeShadowPayload.barycentrics = float2(0.0);
                            volumeShadowPayload.hitDistance = 0.0;
                            volumeShadowPayload.hit = 1;
                            TraceRay(parameters.scene,
                                     RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                                         RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                                     0xff, 0, 1, 0, volumeShadowRay, volumeShadowPayload);
                            bool volumeVisible = volumeShadowPayload.hit == 0;
                            if (!volumeVisible)
                            {
                                GpuOptixInstance blocker = parameters.instances[volumeShadowPayload.instanceIndex];
                                volumeVisible = parameters.materials[blocker.materialIndex].transmissionClearcoat.x > 0.0;
                            }
                            if (volumeVisible)
                            {
                                float phase = henyeyGreenstein(dot(ray.Direction, volumeLightDirection),
                                                               mediumAnisotropy[mediumDepth - 1]);
                                float misWeight = volumeLightDelta ? 1.0 :
                                    misPowerHeuristic(volumeLightPdf, phase);
                                sampleRadiance += throughput * volumeLightRadiance * phase *
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
                        ray.Origin = volumePosition + phaseDirection * 0.0001;
                        ray.Direction = phaseDirection;
                        ray.TMin = 0.00005;
                        ray.TMax = 10000.0;
                        continue;
                    }
                    throughput *= exp(-sigmaT * payload.hitDistance);
                }
            }
            if (subsurfaceActive)
            {
                throughput *= subsurfaceRandomWalkTransmittance(subsurfaceWalkColor,
                                                                 subsurfaceWalkRadius,
                                                                 payload.hitDistance);
                subsurfaceActive = false;
            }

            GpuOptixInstance hitInstance = parameters.instances[payload.instanceIndex];
            uint3 triangle = parameters.indices[hitInstance.geometry.y + payload.primitiveIndex] +
                             hitInstance.geometry.x;
            float barycentric0 = 1.0 - payload.barycentrics.x - payload.barycentrics.y;
            float3 normal = normalize(parameters.normals[triangle.x] * barycentric0 +
                                      parameters.normals[triangle.y] * payload.barycentrics.x +
                                      parameters.normals[triangle.z] * payload.barycentrics.y);
            normal = normalize(mul((float3x3)hitInstance.normalTransform, normal));
            float4 vertexTangent = parameters.tangents[triangle.x] * barycentric0 +
                                   parameters.tangents[triangle.y] * payload.barycentrics.x +
                                   parameters.tangents[triangle.z] * payload.barycentrics.y;
            vertexTangent.xyz = normalize(mul((float3x3)hitInstance.transform, vertexTangent.xyz));
            vertexTangent.w *= hitInstance.handedness;
            float2 uv = parameters.uvs[triangle.x] * barycentric0 +
                        parameters.uvs[triangle.y] * payload.barycentrics.x +
                        parameters.uvs[triangle.z] * payload.barycentrics.y;
            float3 viewDirection = normalize(-ray.Direction);

            uint materialIndex = parameters.materialOverrideId != 0xffffffffu
                                     ? parameters.materialOverrideId
                                     : hitInstance.materialIndex;
            GpuMaterial material = parameters.materials[materialIndex];
            float3 worldPosition = ray.Origin + ray.Direction * payload.hitDistance;
            SurfaceData surface = resolveMaterialSurface(parameters, material, worldPosition, normal, vertexTangent, uv);
            if ((surface.flags & MATERIAL_FLAG_ALPHA_MASK) != 0u && surface.alpha < material.surfaceParameters.w)
            {
                float alphaCoordinateScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
                float alphaRayBias = max(0.0001, alphaCoordinateScale * 0.00001);
                ray.Origin = worldPosition + ray.Direction * alphaRayBias;
                ray.TMin = alphaRayBias * 0.5;
                ray.TMax = 10000.0;
                continue;
            }
            bool enteringSurface = dot(surface.geometricNormal, viewDirection) >= 0.0;
            float etaI = mediumIors[mediumDepth - 1];
            float etaT = enteringSurface ? surface.ior
                                         : (mediumDepth > 1 ? mediumIors[mediumDepth - 2] : 1.0);
            normal = surface.shadingNormal;
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
            sampleRadiance += throughput * surface.emissive;

            float3 sampledLightDirection = normalize(-parameters.lightDirection.xyz);
            float3 sampledLightRadiance = parameters.lightColorAndIntensity.xyz *
                                          parameters.lightColorAndIntensity.w;
            float sampledLightPdf = 1.0;
            float sampledLightDistance = 10000.0;
            bool sampledLightDelta = true;
            float environmentSelectionProbability = environmentLightSelectionProbability(parameters);
            float lightSelector = random01(seed ^ ((bounce + 1) * 0x6c8e9cf5u));
            bool sampleEnvironmentLight = lightSelector < environmentSelectionProbability ||
                                          parameters.emissiveLightPower <= 0.0;
            if (!sampleEnvironmentLight)
            {
                MeshLightSample meshLightSample = sampleEmissiveMesh(
                    parameters, worldPosition, normal,
                    float3(random01(seed ^ ((bounce + 1) * 0x85ebca6bu)),
                           random01(seed ^ ((bounce + 1) * 0xc2b2ae35u)),
                           random01(seed ^ ((bounce + 1) * 0x27d4eb2fu))));
                sampledLightDirection = meshLightSample.direction;
                sampledLightRadiance = meshLightSample.radiance;
                sampledLightDistance = meshLightSample.distance;
                sampledLightPdf = meshLightSample.pdf * (1.0 - environmentSelectionProbability);
                sampledLightDelta = false;
            }
            else if (parameters.globalLightMode == 1)
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
            else
                sampledLightPdf = max(environmentSelectionProbability, 1e-8);

            float visibility = 1.0;
            if (parameters.rayTracedShadows != 0 && sampledLightPdf > 0.0)
            {
                RayDesc shadowRay;
                shadowRay.Origin = worldPosition + normal * rayBias;
                shadowRay.Direction = sampledLightDirection;
                shadowRay.TMin = rayBias * 0.5;
                shadowRay.TMax = max(sampledLightDistance - rayBias, rayBias);

                RadiancePayload shadowPayload;
                shadowPayload.primitiveIndex = 0;
                shadowPayload.instanceIndex = 0;
                shadowPayload.barycentrics = float2(0.0);
                shadowPayload.hitDistance = 0.0;
                shadowPayload.hit = 1;
                TraceRay(parameters.scene,
                         RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
                             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                         0xff, 0, 1, 0, shadowRay, shadowPayload);
                visibility = shadowPayload.hit == 0 ? 1.0 : 0.0;
            }
            float nDotLight = saturate(dot(normal, sampledLightDirection));
            if (visibility > 0.0 && sampledLightPdf > 0.0 && nDotLight > 0.0)
            {
                float3 bsdfValue = evaluateBsdfWithIors(surface, viewDirection, sampledLightDirection,
                                                        etaI, etaT);
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
                previousBsdfPdf = abs(dot(normal, inwardDirection)) / PI;
                previousBsdfDelta = false;
                ray.Origin = worldPosition - normal * rayBias;
                ray.Direction = inwardDirection;
                ray.TMin = rayBias * 0.5;
                ray.TMax = 10000.0;
                continue;
            }

            float lobeRandom = parameters.rayTracedReflections != 0
                                   ? random01(seed ^ ((bounce + 1) * 0x51ed270bu))
                                   : 0.0;
            BsdfSample bsdfSample = sampleBsdfWithIors(
                surface, viewDirection,
                float2(random01(seed ^ ((bounce + 1) * 0x68bc21ebu)),
                       random01(seed ^ ((bounce + 1) * 0x02e5be93u))),
                lobeRandom, etaI, etaT);
            if (bsdfSample.pdf <= 0.0 || max(bsdfSample.weight.x, max(bsdfSample.weight.y,
                                                                      bsdfSample.weight.z)) <= 0.0)
                break;
            throughput *= bsdfSample.weight;
            previousBsdfPdf = bsdfSample.pdf;
            previousBsdfDelta = bsdfSample.delta;
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
