import MaterialTypes;
import Microfacet;

float sheenMaximumColor(SurfaceData surface)
{
    return max(surface.sheenColor.x, max(surface.sheenColor.y, surface.sheenColor.z));
}

float charlieDistribution(float nDotH, float roughness)
{
    float alpha = max(roughness * roughness, 0.001);
    float inverseAlpha = 1.0 / alpha;
    float sineTheta = sqrt(max(1.0 - nDotH * nDotH, 0.0));
    return (2.0 + inverseAlpha) * pow(sineTheta, inverseAlpha) / (2.0 * PI);
}

float sheenVisibility(float nDotV, float nDotL)
{
    return 1.0 / max(4.0 * (nDotV + nDotL - nDotV * nDotL), 1e-5);
}

float3 evaluateSheen(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    float nDotL = saturate(dot(surface.shadingNormal, lightDirection));
    if (sheenMaximumColor(surface) <= 0.0 || nDotV <= 0.0 || nDotL <= 0.0)
        return float3(0.0);
    float3 halfway = normalize(viewDirection + lightDirection);
    return surface.sheenColor * charlieDistribution(saturate(dot(surface.shadingNormal, halfway)),
                                                     surface.sheenRoughness) *
           sheenVisibility(nDotV, nDotL);
}

float sheenReflectionPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float3 halfwayVector = viewDirection + lightDirection;
    if (sheenMaximumColor(surface) <= 0.0 || dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    float vDotH = saturate(dot(viewDirection, halfway));
    float nDotH = saturate(dot(surface.shadingNormal, halfway));
    return charlieDistribution(nDotH, surface.sheenRoughness) * nDotH /
           max(4.0 * vDotH, 1e-6);
}

float3 sampleSheenHalfway(SurfaceData surface, float2 randomSample)
{
    float alpha = max(surface.sheenRoughness * surface.sheenRoughness, 0.001);
    float sineTheta = pow(randomSample.x, alpha / (2.0 * alpha + 1.0));
    float cosineTheta = sqrt(max(1.0 - sineTheta * sineTheta, 0.0));
    float phi = 2.0 * PI * randomSample.y;
    return normalize(surface.tangent * (cos(phi) * sineTheta) +
                     surface.bitangent * (sin(phi) * sineTheta) +
                     surface.shadingNormal * cosineTheta);
}

float sheenEnergyScale(SurfaceData surface)
{
    return clamp(1.0 - sheenMaximumColor(surface) * (0.35 + 0.35 * surface.sheenRoughness), 0.0, 1.0);
}
