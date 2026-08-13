import MaterialTypes;
import Microfacet;

float3 subsurfaceDiffusionProfile(SurfaceData surface, float distance)
{
    float radius = max(surface.subsurfaceRadius, 1e-4);
    float scaledDistance = max(distance / radius, 1e-4);
    float profile = (exp(-scaledDistance) + exp(-scaledDistance / 3.0)) /
                    max(8.0 * PI * radius * max(distance, 1e-4), 1e-6);
    return max(surface.subsurfaceColor, 0.0) * profile;
}

float3 evaluateSubsurfaceApproximation(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                       float thickness)
{
    if (surface.subsurface <= 0.0)
        return float3(0.0);
    float wrap = saturate((dot(surface.shadingNormal, lightDirection) + 0.5) / 1.5);
    float backLighting = saturate(dot(-surface.shadingNormal, lightDirection));
    float3 diffusion = subsurfaceDiffusionProfile(surface, max(thickness, surface.subsurfaceRadius * 0.1));
    return surface.subsurface * surface.baseColor * diffusion * (wrap + backLighting * 0.75);
}

float3 sampleSubsurfaceInwardDirection(SurfaceData surface, float2 randomSample)
{
    float radius = sqrt(randomSample.x);
    float phi = 2.0 * PI * randomSample.y;
    float x = radius * cos(phi);
    float y = radius * sin(phi);
    float z = sqrt(max(1.0 - randomSample.x, 0.0));
    return normalize(surface.tangent * x + surface.bitangent * y - surface.shadingNormal * z);
}
