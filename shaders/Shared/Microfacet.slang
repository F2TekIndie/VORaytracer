static const float PI = 3.14159265358979323846;

float distributionGgx(float3 normal, float3 halfway, float roughness)
{
    float alpha = max(roughness * roughness, 0.001);
    float alpha2 = alpha * alpha;
    float nDotH = saturate(dot(normal, halfway));
    float denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(PI * denominator * denominator, 1e-6);
}

float geometrySchlickGgx(float nDotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

float geometrySmith(float3 normal, float3 viewDirection, float3 lightDirection, float roughness)
{
    return geometrySchlickGgx(saturate(dot(normal, viewDirection)), roughness) *
           geometrySchlickGgx(saturate(dot(normal, lightDirection)), roughness);
}

float ggxAlpha(float roughness)
{
    return max(roughness * roughness, 0.001);
}

float ggxLambda(float cosineTheta, float alpha)
{
    float cosineSquared = max(cosineTheta * cosineTheta, 1e-8);
    float tangentSquared = max(1.0 - cosineSquared, 0.0) / cosineSquared;
    return 0.5 * (sqrt(1.0 + alpha * alpha * tangentSquared) - 1.0);
}

float ggxG1(float cosineTheta, float alpha)
{
    return cosineTheta > 0.0 ? 1.0 / (1.0 + ggxLambda(cosineTheta, alpha)) : 0.0;
}

float ggxG2(float nDotV, float nDotL, float alpha)
{
    return (nDotV > 0.0 && nDotL > 0.0)
               ? 1.0 / (1.0 + ggxLambda(nDotV, alpha) + ggxLambda(nDotL, alpha))
               : 0.0;
}

float3 sampleGgxVisibleNormalLocal(float3 localView, float alpha, float2 randomSample)
{
    float3 stretchedView = normalize(float3(alpha * localView.x, alpha * localView.y, localView.z));
    float lensq = stretchedView.x * stretchedView.x + stretchedView.y * stretchedView.y;
    float3 tangent1 = lensq > 1e-8
                          ? float3(-stretchedView.y, stretchedView.x, 0.0) / sqrt(lensq)
                          : float3(1.0, 0.0, 0.0);
    float3 tangent2 = cross(stretchedView, tangent1);
    float radius = sqrt(randomSample.x);
    float phi = 2.0 * PI * randomSample.y;
    float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    float blend = 0.5 * (1.0 + stretchedView.z);
    t2 = lerp(sqrt(max(1.0 - t1 * t1, 0.0)), t2, blend);
    float3 hemisphereNormal = tangent1 * t1 + tangent2 * t2 +
                              stretchedView * sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0));
    return normalize(float3(alpha * hemisphereNormal.x, alpha * hemisphereNormal.y,
                            max(hemisphereNormal.z, 0.0)));
}

float ggxVisibleNormalPdf(float3 normal, float3 viewDirection, float3 halfway, float roughness)
{
    float nDotV = saturate(dot(normal, viewDirection));
    float nDotH = saturate(dot(normal, halfway));
    float vDotH = saturate(dot(viewDirection, halfway));
    if (nDotV <= 0.0 || nDotH <= 0.0 || vDotH <= 0.0)
        return 0.0;
    return distributionGgx(normal, halfway, roughness) * ggxG1(nDotV, ggxAlpha(roughness)) *
           vDotH / max(nDotV, 1e-6);
}
