import MaterialTypes;
import Fresnel;
import Microfacet;

float3 absorptionCoefficient(SurfaceData surface)
{
    return -log(clamp(surface.absorptionColor, 1e-6, 1.0)) /
           max(surface.absorptionDistance, 1e-4);
}

float3 beerLambert(float3 sigmaA, float distance)
{
    return exp(-max(sigmaA, 0.0) * max(distance, 0.0));
}

bool refractDielectric(float3 incidentDirection, float3 normal, float etaRatio, out float3 direction)
{
    float cosineI = dot(-incidentDirection, normal);
    float sineSquaredT = etaRatio * etaRatio * max(1.0 - cosineI * cosineI, 0.0);
    if (sineSquaredT >= 1.0)
    {
        direction = reflect(incidentDirection, normal);
        return false;
    }
    float cosineT = sqrt(max(1.0 - sineSquaredT, 0.0));
    direction = normalize(etaRatio * incidentDirection + (etaRatio * cosineI - cosineT) * normal);
    return true;
}

float3 evaluateRoughTransmission(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                 float etaI, float etaT)
{
    float nDotV = dot(surface.shadingNormal, viewDirection);
    float nDotL = dot(surface.shadingNormal, lightDirection);
    if (surface.transmission <= 0.0 || nDotV <= 0.0 || nDotL >= 0.0)
        return float3(0.0);
    float eta = etaT / etaI;
    float3 halfwayVector = viewDirection + lightDirection * eta;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return float3(0.0);
    float3 halfway = normalize(halfwayVector);
    if (dot(halfway, surface.shadingNormal) < 0.0)
        halfway = -halfway;
    float vDotH = dot(viewDirection, halfway);
    float lDotH = dot(lightDirection, halfway);
    if (vDotH <= 0.0 || lDotH >= 0.0)
        return float3(0.0);
    float fresnel = fresnelDielectric(vDotH, etaI, etaT);
    float distribution = distributionGgx(surface.shadingNormal, halfway, surface.roughness);
    float geometry = ggxG2(abs(nDotV), abs(nDotL), ggxAlpha(surface.roughness));
    float denominator = vDotH + eta * lDotH;
    float factor = abs(vDotH * lDotH) * eta * eta /
                   max(abs(nDotV * nDotL) * denominator * denominator, 1e-8);
    return surface.baseColor * surface.transmission * (1.0 - fresnel) * distribution * geometry * factor;
}

float evaluateRoughTransmissionPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                   float etaI, float etaT)
{
    float eta = etaT / etaI;
    float3 halfwayVector = viewDirection + lightDirection * eta;
    if (surface.transmission <= 0.0 || dot(surface.shadingNormal, viewDirection) <= 0.0 ||
        dot(surface.shadingNormal, lightDirection) >= 0.0 || dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    if (dot(halfway, surface.shadingNormal) < 0.0)
        halfway = -halfway;
    float vDotH = dot(viewDirection, halfway);
    float lDotH = dot(lightDirection, halfway);
    float denominator = vDotH + eta * lDotH;
    float jacobian = abs(eta * eta * lDotH / max(denominator * denominator, 1e-8));
    return ggxVisibleNormalPdf(surface.shadingNormal, viewDirection, halfway, surface.roughness) * jacobian;
}

bool sampleRoughTransmissionDirection(SurfaceData surface, float3 viewDirection, float2 directionSample,
                                      float etaI, float etaT, out float3 direction)
{
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
    {
        direction = float3(0.0);
        return false;
    }
    float3 localView = float3(dot(viewDirection, surface.tangent),
                              dot(viewDirection, surface.bitangent), nDotV);
    float3 localHalfway = sampleGgxVisibleNormalLocal(localView, ggxAlpha(surface.roughness), directionSample);
    float3 halfway = normalize(surface.tangent * localHalfway.x + surface.bitangent * localHalfway.y +
                               surface.shadingNormal * localHalfway.z);
    return refractDielectric(-viewDirection, halfway, etaI / etaT, direction);
}

bool roughTransmissionHalfwayIsTir(float3 viewDirection, float3 halfway, float etaI, float etaT)
{
    float3 ignoredDirection;
    return !refractDielectric(-viewDirection, halfway, etaI / etaT, ignoredDirection);
}

float3 evaluateRoughTransmissionTirReflection(SurfaceData surface, float3 viewDirection,
                                               float3 lightDirection, float etaI, float etaT)
{
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    float nDotL = saturate(dot(surface.shadingNormal, lightDirection));
    float3 halfwayVector = viewDirection + lightDirection;
    if (surface.transmission <= 0.0 || nDotV <= 0.0 || nDotL <= 0.0 ||
        dot(halfwayVector, halfwayVector) <= 1e-10)
        return float3(0.0);
    float3 halfway = normalize(halfwayVector);
    if (!roughTransmissionHalfwayIsTir(viewDirection, halfway, etaI, etaT))
        return float3(0.0);
    float distribution = distributionGgx(surface.shadingNormal, halfway, surface.roughness);
    float geometry = ggxG2(nDotV, nDotL, ggxAlpha(surface.roughness));
    float interfaceRatio = (etaT - etaI) / max(etaT + etaI, 1e-6);
    float approximateFresnel = fresnelSchlick(saturate(dot(viewDirection, halfway)),
                                               float3(interfaceRatio * interfaceRatio)).x;
    float missingTirReflection = surface.transmission * (1.0 - surface.metallic) *
                                 (1.0 - approximateFresnel);
    return missingTirReflection * distribution * geometry /
           max(4.0 * nDotV * nDotL, 1e-6);
}

float roughTransmissionTirReflectionPdf(SurfaceData surface, float3 viewDirection,
                                         float3 lightDirection, float etaI, float etaT)
{
    float3 halfwayVector = viewDirection + lightDirection;
    if (surface.transmission <= 0.0 || dot(surface.shadingNormal, viewDirection) <= 0.0 ||
        dot(surface.shadingNormal, lightDirection) <= 0.0 || dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    if (!roughTransmissionHalfwayIsTir(viewDirection, halfway, etaI, etaT))
        return 0.0;
    return ggxVisibleNormalPdf(surface.shadingNormal, viewDirection, halfway, surface.roughness) /
           max(4.0 * abs(dot(viewDirection, halfway)), 1e-6);
}
