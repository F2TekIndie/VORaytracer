import Pbr;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure sceneTlas;
[[vk::binding(1, 0)]] Texture2D<float4> worldPositionTexture;
[[vk::binding(2, 0)]] Texture2D<float4> normalRoughnessTexture;
[[vk::binding(3, 0)]] RWTexture2D<float4> lightingOutput;

struct RayQuerySettings
{
    uint2 outputSize;
    uint frameIndex;
    uint enableShadows;
    float3 lightDirection;
    float lightIntensity;
};

[[vk::push_constant]] ConstantBuffer<RayQuerySettings> settings;

[shader("compute")]
[numthreads(8, 8, 1)]
void ComputeMain(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= settings.outputSize))
        return;

    float4 positionSample = worldPositionTexture.Load(int3(dispatchId.xy, 0));
    float4 normalSample = normalRoughnessTexture.Load(int3(dispatchId.xy, 0));
    float3 normal = normalize(normalSample.xyz);
    float visibility = 1.0;

    if (settings.enableShadows != 0 && positionSample.w > 0.0)
    {
        RayDesc ray;
        ray.Origin = positionSample.xyz + normal * 0.001;
        ray.Direction = normalize(-settings.lightDirection);
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, ray);
        query.Proceed();
        visibility = query.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
    }

    lightingOutput[dispatchId.xy] = float4(visibility.xxx * settings.lightIntensity, 1.0);
}
