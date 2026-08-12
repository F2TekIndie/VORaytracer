static const float PI = 3.14159265358979323846;

float3 fresnelSchlick(float cosTheta, float3 f0)
{
    float factor = pow(saturate(1.0 - cosTheta), 5.0);
    return f0 + (1.0 - f0) * factor;
}

float3 fresnelSchlickRoughness(float cosTheta, float3 f0, float roughness)
{
    float factor = pow(saturate(1.0 - cosTheta), 5.0);
    return f0 + (max(float3(1.0 - roughness), f0) - f0) * factor;
}

float distributionGgx(float3 normal, float3 halfway, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = saturate(dot(normal, halfway));
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1e-5);
}

float geometrySchlickGgx(float nDotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    return nDotV / max(nDotV * (1.0 - k) + k, 1e-5);
}

float geometrySmith(float3 normal, float3 viewDirection, float3 lightDirection, float roughness)
{
    return geometrySchlickGgx(saturate(dot(normal, viewDirection)), roughness) *
           geometrySchlickGgx(saturate(dot(normal, lightDirection)), roughness);
}

float3 evaluateMetallicRoughness(
    float3 baseColor,
    float metallic,
    float roughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    float3 radiance)
{
    float3 halfway = normalize(viewDirection + lightDirection);
    float3 f0 = lerp(float3(0.04), baseColor, metallic);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float distribution = distributionGgx(normal, halfway, max(roughness, 0.045));
    float geometry = geometrySmith(normal, viewDirection, lightDirection, roughness);
    float3 specular = distribution * geometry * fresnel /
                      max(4.0 * saturate(dot(normal, viewDirection)) * saturate(dot(normal, lightDirection)), 1e-4);
    float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
    return (diffuseWeight * baseColor / PI + specular) * radiance * saturate(dot(normal, lightDirection));
}
