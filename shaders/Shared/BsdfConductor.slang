import Fresnel;
import Microfacet;

float3 evaluateConductorGgx(
    float3 f0,
    float roughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection)
{
    float3 halfway = normalize(viewDirection + lightDirection);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float distribution = distributionGgx(normal, halfway, roughness);
    float geometry = ggxG2(saturate(dot(normal, viewDirection)), saturate(dot(normal, lightDirection)),
                            ggxAlpha(roughness));
    return distribution * geometry * fresnel /
           max(4.0 * saturate(dot(normal, viewDirection)) * saturate(dot(normal, lightDirection)), 1e-5);
}
