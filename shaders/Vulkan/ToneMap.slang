[[vk::binding(0, 0)]] Texture2D<float4> hdrInput;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;

struct ToneMapSettings
{
    uint2 outputSize;
    float exposure;
    float padding;
};

[[vk::push_constant]] ConstantBuffer<ToneMapSettings> settings;

float3 aces(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ComputeMain(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= settings.outputSize))
        return;
    float3 hdr = hdrInput.Load(int3(dispatchId.xy, 0)).rgb * exp2(settings.exposure);
    outputImage[dispatchId.xy] = float4(aces(hdr), 1.0);
}
