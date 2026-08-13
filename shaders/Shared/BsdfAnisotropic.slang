import MaterialTypes;
import Fresnel;
import Microfacet;

void anisotropicBasis(SurfaceData surface, out float3 tangent, out float3 bitangent)
{
    float cosine = cos(surface.anisotropyRotation);
    float sine = sin(surface.anisotropyRotation);
    tangent = normalize(surface.tangent * cosine + surface.bitangent * sine);
    bitangent = normalize(-surface.tangent * sine + surface.bitangent * cosine);
}

float2 anisotropicAlphas(SurfaceData surface)
{
    float alpha = ggxAlpha(surface.roughness);
    float aspect = sqrt(max(1.0 - 0.9 * abs(surface.anisotropy), 0.1));
    float alphaX = alpha / aspect;
    float alphaY = alpha * aspect;
    return surface.anisotropy >= 0.0 ? float2(alphaX, alphaY) : float2(alphaY, alphaX);
}

float anisotropicGgxD(float3 halfway, float3 normal, float3 tangent, float3 bitangent, float2 alphas)
{
    float3 localHalfway = float3(dot(halfway, tangent), dot(halfway, bitangent), dot(halfway, normal));
    if (localHalfway.z <= 0.0)
        return 0.0;
    float denominator = localHalfway.x * localHalfway.x / (alphas.x * alphas.x) +
                        localHalfway.y * localHalfway.y / (alphas.y * alphas.y) +
                        localHalfway.z * localHalfway.z;
    return 1.0 / max(PI * alphas.x * alphas.y * denominator * denominator, 1e-8);
}

float anisotropicLambda(float3 direction, float3 normal, float3 tangent, float3 bitangent, float2 alphas)
{
    float3 localDirection = float3(dot(direction, tangent), dot(direction, bitangent), dot(direction, normal));
    if (localDirection.z <= 0.0)
        return 1e20;
    float scaledTangentSquared = (localDirection.x * alphas.x) * (localDirection.x * alphas.x) +
                                 (localDirection.y * alphas.y) * (localDirection.y * alphas.y);
    return 0.5 * (sqrt(1.0 + scaledTangentSquared /
                       max(localDirection.z * localDirection.z, 1e-8)) - 1.0);
}

float anisotropicG1(float3 direction, float3 normal, float3 tangent, float3 bitangent, float2 alphas)
{
    return 1.0 / (1.0 + anisotropicLambda(direction, normal, tangent, bitangent, alphas));
}

float3 evaluateAnisotropicGgx(float3 f0, SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    float nDotL = saturate(dot(surface.shadingNormal, lightDirection));
    if (nDotV <= 0.0 || nDotL <= 0.0)
        return float3(0.0);
    float3 halfway = normalize(viewDirection + lightDirection);
    float3 tangent;
    float3 bitangent;
    anisotropicBasis(surface, tangent, bitangent);
    float2 alphas = anisotropicAlphas(surface);
    float distribution = anisotropicGgxD(halfway, surface.shadingNormal, tangent, bitangent, alphas);
    float geometry = 1.0 / (1.0 + anisotropicLambda(viewDirection, surface.shadingNormal, tangent, bitangent, alphas) +
                            anisotropicLambda(lightDirection, surface.shadingNormal, tangent, bitangent, alphas));
    float3 fresnel = fresnelSchlick(saturate(dot(viewDirection, halfway)), f0);
    return distribution * geometry * fresnel / max(4.0 * nDotV * nDotL, 1e-6);
}

float anisotropicReflectionPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    float3 halfwayVector = viewDirection + lightDirection;
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0 || dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    float3 tangent;
    float3 bitangent;
    anisotropicBasis(surface, tangent, bitangent);
    float2 alphas = anisotropicAlphas(surface);
    float halfwayPdf = anisotropicGgxD(halfway, surface.shadingNormal, tangent, bitangent, alphas) *
                       anisotropicG1(viewDirection, surface.shadingNormal, tangent, bitangent, alphas) *
                       saturate(dot(viewDirection, halfway)) / max(nDotV, 1e-6);
    return halfwayPdf / max(4.0 * abs(dot(viewDirection, halfway)), 1e-6);
}

float3 sampleAnisotropicVisibleNormal(SurfaceData surface, float3 viewDirection, float2 randomSample)
{
    float3 tangent;
    float3 bitangent;
    anisotropicBasis(surface, tangent, bitangent);
    float2 alphas = anisotropicAlphas(surface);
    float3 localView = float3(dot(viewDirection, tangent), dot(viewDirection, bitangent),
                              dot(viewDirection, surface.shadingNormal));
    float3 stretchedView = normalize(float3(alphas.x * localView.x, alphas.y * localView.y, localView.z));
    float lensq = stretchedView.x * stretchedView.x + stretchedView.y * stretchedView.y;
    float3 tangent1 = lensq > 1e-8 ? float3(-stretchedView.y, stretchedView.x, 0.0) / sqrt(lensq)
                                   : float3(1.0, 0.0, 0.0);
    float3 tangent2 = cross(stretchedView, tangent1);
    float radius = sqrt(randomSample.x);
    float phi = 2.0 * PI * randomSample.y;
    float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    t2 = lerp(sqrt(max(1.0 - t1 * t1, 0.0)), t2, 0.5 * (1.0 + stretchedView.z));
    float3 hemisphereNormal = tangent1 * t1 + tangent2 * t2 +
                              stretchedView * sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0));
    float3 localHalfway = normalize(float3(alphas.x * hemisphereNormal.x,
                                           alphas.y * hemisphereNormal.y,
                                           max(hemisphereNormal.z, 0.0)));
    return normalize(tangent * localHalfway.x + bitangent * localHalfway.y +
                     surface.shadingNormal * localHalfway.z);
}
