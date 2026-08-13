import BsdfConductor;
import BsdfDiffuse;
import Fresnel;

float3 evaluateDielectricBase(
    float3 baseColor,
    float roughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection)
{
    float3 halfway = normalize(viewDirection + lightDirection);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), float3(0.04));
    float3 specular = evaluateConductorGgx(float3(0.04), roughness, normal, viewDirection, lightDirection);
    return (1.0 - fresnel) * evaluateLambert(baseColor) + specular;
}
