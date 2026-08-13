import Microfacet;

float3 evaluateLambert(float3 baseColor)
{
    return max(baseColor, 0.0) / PI;
}

float cosineHemispherePdf(float3 normal, float3 direction)
{
    return saturate(dot(normal, direction)) / PI;
}
