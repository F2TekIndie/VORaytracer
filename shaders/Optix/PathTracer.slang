import Pbr;

struct RadiancePayload
{
    uint primitiveIndex;
    float2 barycentrics;
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
    StructuredBuffer<float3> normals;
    StructuredBuffer<uint3> indices;
    StructuredBuffer<uint> triangleMaterialIndices;
    StructuredBuffer<GpuMaterial> materials;
    RaytracingAccelerationStructure scene;
    uint2 outputSize;
    uint frameIndex;
    uint maxBounces;
    uint samplesPerFrame;
    uint accumulatedSamples;
    float4 cameraPositionAndFov;
    float4 cameraTargetAndAspect;
    float4 cameraUp;
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
        payload.primitiveIndex = 0;
        payload.barycentrics = float2(0.0);
        payload.hit = 0;
        TraceRay(parameters.scene, RAY_FLAG_NONE, 0xff, 0, 1, 0, ray, payload);
        if (payload.hit != 0)
        {
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
            float3 lightDirection = normalize(float3(-0.45, 0.85, 0.3));
            float3 radiance = float3(5.0, 4.8, 4.4);
            float3 direct = evaluateMetallicRoughness(material.baseColorAndMetallic.xyz,
                                                      material.baseColorAndMetallic.w,
                                                      material.emissiveAndRoughness.w,
                                                      normal,
                                                      viewDirection,
                                                      lightDirection,
                                                      radiance);
            sampleSum += direct + material.baseColorAndMetallic.xyz * 0.025 + material.emissiveAndRoughness.xyz;
        }
        else
        {
            float skyFactor = saturate(ray.Direction.y * 0.5 + 0.5);
            sampleSum += lerp(float3(0.012, 0.018, 0.035), float3(0.16, 0.38, 0.78), skyFactor);
        }
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
    payload.hit = 0;
}

[shader("closesthit")]
void ClosestHit(inout RadiancePayload payload,
                BuiltInTriangleIntersectionAttributes attributes)
{
    payload.primitiveIndex = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
    payload.hit = 1;
}
