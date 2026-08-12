import Pbr;

struct RadiancePayload
{
    uint primitiveIndex;
    float2 barycentrics;
    float hitDistance;
    uint hit;
};

struct GpuMaterial
{
    float4 baseColorAndMetallic;
    float4 emissiveAndRoughness;
};

struct LaunchParameters
{
    RWStructuredBuffer<float4> output;
    RWStructuredBuffer<half4> denoisedOutput;
    RWStructuredBuffer<float4> albedoGuide;
    RWStructuredBuffer<float4> normalGuide;
    RWStructuredBuffer<uint> displayOutput;
    StructuredBuffer<float3> normals;
    StructuredBuffer<uint3> indices;
    StructuredBuffer<uint> triangleMaterialIndices;
    StructuredBuffer<GpuMaterial> materials;
    StructuredBuffer<float4> environmentPixels;
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
    uint padding0;
    uint padding1;
    uint padding2;
    uint indirectLighting;
    uint globalLightMode;
    uint environmentWidth;
    uint environmentHeight;
    float environmentIntensity;
    float environmentRotation;
    uint environmentVisible;
    uint environmentMipCount;
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
        uint bounceLimit = (parameters.rayTracedReflections != 0 || parameters.indirectLighting != 0)
                               ? max(parameters.maxBounces, 1u) : 1u;
        for (uint bounce = 0; bounce < bounceLimit; ++bounce)
        {
            RadiancePayload payload;
            payload.primitiveIndex = 0;
            payload.barycentrics = float2(0.0);
            payload.hitDistance = 0.0;
            payload.hit = 0;
            TraceRay(parameters.scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
            if (payload.hit == 0)
            {
                if (bounce != 0 || parameters.environmentVisible != 0)
                    sampleRadiance += throughput * sampleGlobalEnvironment(parameters, ray.Direction);
                break;
            }

            uint3 triangle = parameters.indices[payload.primitiveIndex];
            float barycentric0 = 1.0 - payload.barycentrics.x - payload.barycentrics.y;
            float3 normal = normalize(parameters.normals[triangle.x] * barycentric0 +
                                      parameters.normals[triangle.y] * payload.barycentrics.x +
                                      parameters.normals[triangle.z] * payload.barycentrics.y);
            float3 viewDirection = normalize(-ray.Direction);
            if (dot(normal, viewDirection) < 0.0)
                normal = -normal;

            uint materialIndex = parameters.triangleMaterialIndices[payload.primitiveIndex];
            GpuMaterial material = parameters.materials[materialIndex];
            if (bounce == 0 && sampleIndex == 0 && parameters.writeDisplay == 0)
            {
                guideAlbedo = material.baseColorAndMetallic.xyz;
                guideNormal = normal;
            }
            float3 worldPosition = ray.Origin + ray.Direction * payload.hitDistance;
            float lightDistance = 10000.0;
            float3 lightDirection = normalize(-parameters.lightDirection.xyz);
            float coordinateScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
            float rayBias = max(0.0001, coordinateScale * 0.00001);
            float visibility = 1.0;
            if (parameters.rayTracedShadows != 0 && parameters.globalLightMode != 1)
            {
                RayDesc shadowRay;
                shadowRay.Origin = worldPosition + normal * rayBias;
                shadowRay.Direction = lightDirection;
                shadowRay.TMin = rayBias * 0.5;
                shadowRay.TMax = max(lightDistance - rayBias, rayBias);

                RadiancePayload shadowPayload;
                shadowPayload.primitiveIndex = 0;
                shadowPayload.barycentrics = float2(0.0);
                shadowPayload.hitDistance = 0.0;
                shadowPayload.hit = 1;
                TraceRay(parameters.scene,
                         RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
                             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                         0xff, 0, 1, 0, shadowRay, shadowPayload);
                visibility = shadowPayload.hit == 0 ? 1.0 : 0.0;
            }
            float3 lightRadiance = parameters.globalLightMode == 1
                                       ? float3(0.0)
                                       : parameters.lightColorAndIntensity.xyz *
                                             parameters.lightColorAndIntensity.w * visibility;
            float roughness = clamp(material.emissiveAndRoughness.w, 0.02, 1.0);
            float3 direct = evaluateMetallicRoughness(material.baseColorAndMetallic.xyz,
                                                      material.baseColorAndMetallic.w,
                                                      roughness,
                                                      normal,
                                                      viewDirection,
                                                      lightDirection,
                                                      lightRadiance);
            sampleRadiance += throughput *
                              (direct + material.emissiveAndRoughness.xyz);
            if (bounce + 1 >= bounceLimit)
                break;

            float3 f0 = lerp(float3(0.04), material.baseColorAndMetallic.xyz,
                             material.baseColorAndMetallic.w);
            bool sampleSpecular = parameters.rayTracedReflections != 0;
            if (parameters.indirectLighting != 0 && parameters.rayTracedReflections != 0)
            {
                float specularProbability = clamp(0.2 + material.baseColorAndMetallic.w * 0.65,
                                                  0.15, 0.9);
                sampleSpecular = random01(seed ^ ((bounce + 1) * 0x51ed270bu)) < specularProbability;
                throughput *= sampleSpecular ? f0 / specularProbability
                                             : material.baseColorAndMetallic.xyz *
                                               (1.0 - material.baseColorAndMetallic.w) /
                                               (1.0 - specularProbability);
            }
            else if (sampleSpecular)
                throughput *= f0;
            else
                throughput *= material.baseColorAndMetallic.xyz * (1.0 - material.baseColorAndMetallic.w);
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

            float randomPhi = random01(seed ^ ((bounce + 1) * 0x68bc21ebu));
            float randomTheta = random01(seed ^ ((bounce + 1) * 0x02e5be93u));
            float alpha = max(roughness * roughness, 0.002);
            float phi = 2.0 * PI * randomPhi;
            float cosTheta = sampleSpecular
                                 ? sqrt((1.0 - randomTheta) /
                                        max(1.0 + (alpha * alpha - 1.0) * randomTheta, 0.0001))
                                 : sqrt(1.0 - randomTheta);
            float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
            float3 tangent = normalize(cross(abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0)
                                                                      : float3(0.0, 1.0, 0.0),
                                             normal));
            float3 bitangent = cross(normal, tangent);
            float3 halfway = normalize(tangent * (cos(phi) * sinTheta) +
                                       bitangent * (sin(phi) * sinTheta) + normal * cosTheta);
            float3 reflectedDirection = sampleSpecular ? normalize(reflect(ray.Direction, halfway)) : halfway;
            if (dot(reflectedDirection, normal) <= 0.001)
                reflectedDirection = normalize(reflect(ray.Direction, normal));
            ray.Origin = worldPosition + normal * rayBias;
            ray.Direction = reflectedDirection;
            ray.TMin = rayBias * 0.5;
            ray.TMax = 10000.0;
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
    payload.barycentrics = attributes.barycentrics;
    payload.hitDistance = RayTCurrent();
    payload.hit = 1;
}
