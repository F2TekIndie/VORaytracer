import Pbr;

struct RadiancePayload
{
    float3 color;
    float3 baseColor;
    float metallic;
    float roughness;
};

struct LaunchParameters
{
    RWStructuredBuffer<float4> output;
    RaytracingAccelerationStructure scene;
    uint2 outputSize;
    uint frameIndex;
    uint maxBounces;
    uint samplesPerFrame;
    uint accumulatedSamples;
    float4 cameraPositionAndFov;
    float4 cameraTargetAndAspect;
    float4 cameraUp;
    float4 baseColorAndMetallic;
    float roughness;
    float3 materialPadding;
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

        RadiancePayload payload;
        payload.color = float3(0.0);
        payload.baseColor = parameters.baseColorAndMetallic.xyz;
        payload.metallic = parameters.baseColorAndMetallic.w;
        payload.roughness = parameters.roughness;
        TraceRay(parameters.scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
        sampleSum += payload.color;
    }

    float3 current = sampleSum / float(sampleCount);
    if (parameters.accumulatedSamples > 0)
    {
        float3 previous = parameters.output[linearIndex].xyz;
        float weight = float(sampleCount) / float(parameters.accumulatedSamples + sampleCount);
        current = lerp(previous, current, weight);
    }
    parameters.output[linearIndex] = float4(current, 1.0);
}

[shader("miss")]
void Miss(inout RadiancePayload payload)
{
    float3 direction = normalize(WorldRayDirection());
    float skyFactor = saturate(direction.y * 0.5 + 0.5);
    payload.color = lerp(float3(0.012, 0.018, 0.035), float3(0.16, 0.38, 0.78), skyFactor);
}

[shader("closesthit")]
void ClosestHit(inout RadiancePayload payload,
                BuiltInTriangleIntersectionAttributes attributes)
{
    float3 viewDirection = normalize(-WorldRayDirection());
    float3 barycentric = float3(attributes.barycentrics,
                                1.0 - attributes.barycentrics.x - attributes.barycentrics.y);
    float3 normal = normalize(barycentric * 2.0 - 1.0);
    if (dot(normal, viewDirection) < 0.0)
        normal = -normal;

    float3 lightDirection = normalize(float3(-0.45, 0.85, 0.3));
    float3 radiance = float3(5.0, 4.8, 4.4);
    float3 direct = evaluateMetallicRoughness(payload.baseColor,
                                              payload.metallic,
                                              payload.roughness,
                                              normal,
                                              viewDirection,
                                              lightDirection,
                                              radiance);
    payload.color = direct + payload.baseColor * 0.025;
}
