import MaterialTypes;
import Fresnel;
import Microfacet;
import BsdfConductor;

float clearcoatFresnel(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float3 halfwayVector = viewDirection + lightDirection;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    return surface.clearcoat * fresnelSchlick(saturate(dot(normalize(halfwayVector), viewDirection)),
                                               float3(0.04)).x;
}

float3 evaluateClearcoat(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    if (surface.clearcoat <= 0.0 || dot(surface.clearcoatNormal, viewDirection) <= 0.0 ||
        dot(surface.clearcoatNormal, lightDirection) <= 0.0)
        return float3(0.0);
    return surface.clearcoat * evaluateConductorGgx(float3(0.04), surface.clearcoatRoughness,
                                                     surface.clearcoatNormal, viewDirection, lightDirection);
}

float evaluateClearcoatPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float3 halfwayVector = viewDirection + lightDirection;
    if (surface.clearcoat <= 0.0 || dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    float vDotH = saturate(dot(viewDirection, halfway));
    return ggxVisibleNormalPdf(surface.clearcoatNormal, viewDirection, halfway,
                               surface.clearcoatRoughness) /
           max(4.0 * vDotH, 1e-6);
}
