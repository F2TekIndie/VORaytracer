import MaterialTypes;
import Fresnel;
import Microfacet;
import BsdfTypes;
import BsdfDiffuse;
import BsdfConductor;
import BsdfDielectric;
import BsdfClearcoat;
import BsdfTransmission;
import BsdfAnisotropic;
import BsdfSheen;

float luminance(float3 value)
{
    return dot(value, float3(0.2126, 0.7152, 0.0722));
}

float dielectricF0(float ior)
{
    float ratio = (ior - 1.0) / (ior + 1.0);
    return ratio * ratio;
}

struct LobeProbabilities
{
    float diffuse;
    float specular;
    float clearcoat;
    float transmission;
    float sheen;
};

LobeProbabilities lobeProbabilities(SurfaceData surface, float3 viewDirection, float etaI, float etaT)
{
    LobeProbabilities probabilities;
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
    float3 viewFresnel = fresnelSchlick(nDotV, f0);
    float specularWeight = max(luminance(viewFresnel), 1e-5);
    float diffuseWeight = max(luminance(surface.baseColor) * (1.0 - surface.metallic) *
                                  (1.0 - surface.transmission) * (1.0 - luminance(viewFresnel)),
                              0.0);
    float dielectricFresnel = fresnelDielectric(nDotV, etaI, etaT);
    float transmissionWeight = surface.transmission * (1.0 - surface.metallic) *
                               (1.0 - dielectricFresnel) * max(luminance(surface.baseColor), 0.05);
    float clearcoatWeight = surface.clearcoat * 0.15;
    float sheenWeight = sheenMaximumColor(surface) * (0.2 + 0.3 * surface.sheenRoughness);
    float total = specularWeight + diffuseWeight + transmissionWeight + clearcoatWeight + sheenWeight;
    if (total <= 1e-6)
    {
        probabilities.diffuse = 0.0;
        probabilities.specular = 1.0;
        probabilities.clearcoat = 0.0;
        probabilities.transmission = 0.0;
        probabilities.sheen = 0.0;
        return probabilities;
    }
    probabilities.diffuse = diffuseWeight / total;
    probabilities.specular = specularWeight / total;
    probabilities.clearcoat = clearcoatWeight / total;
    probabilities.transmission = transmissionWeight / total;
    probabilities.sheen = sheenWeight / total;
    return probabilities;
}

float3 evaluateBsdfWithIors(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                            float etaI, float etaT)
{
    float signedNDotL = dot(surface.shadingNormal, lightDirection);
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return float3(0.0);
    if (signedNDotL < 0.0)
        return evaluateRoughTransmission(surface, viewDirection, lightDirection, etaI, etaT);
    if (signedNDotL == 0.0)
        return float3(0.0);

    float3 halfwayVector = viewDirection + lightDirection;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return float3(0.0);
    float3 halfway = normalize(halfwayVector);
    float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float3 specular = abs(surface.anisotropy) > 1e-4
                          ? evaluateAnisotropicGgx(f0, surface, viewDirection, lightDirection)
                          : evaluateConductorGgx(f0, surface.roughness, surface.shadingNormal,
                                                 viewDirection, lightDirection);
    float3 diffuse = (1.0 - fresnel) * (1.0 - surface.metallic) *
                     (1.0 - surface.transmission) * evaluateLambert(surface.baseColor);
    float attenuation = 1.0 - clearcoatFresnel(surface, viewDirection, lightDirection);
    float3 layeredBase = (diffuse + specular) * attenuation +
                         evaluateClearcoat(surface, viewDirection, lightDirection);
    return layeredBase * sheenEnergyScale(surface) + evaluateSheen(surface, viewDirection, lightDirection);
}

float3 evaluateBsdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return evaluateBsdfWithIors(surface, viewDirection, lightDirection,
                                entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
}

float evaluateBsdfPdfWithIors(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                              float etaI, float etaT)
{
    float signedNDotL = dot(surface.shadingNormal, lightDirection);
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return 0.0;
    LobeProbabilities probabilities = lobeProbabilities(surface, viewDirection, etaI, etaT);
    if (signedNDotL < 0.0)
        return probabilities.transmission * evaluateRoughTransmissionPdf(surface, viewDirection, lightDirection,
                                                                           etaI, etaT);
    if (signedNDotL == 0.0)
        return 0.0;
    float3 halfwayVector = viewDirection + lightDirection;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return 0.0;
    float3 halfway = normalize(halfwayVector);
    float vDotH = saturate(dot(viewDirection, halfway));
    float specularPdf = abs(surface.anisotropy) > 1e-4
                            ? anisotropicReflectionPdf(surface, viewDirection, lightDirection)
                            : ggxVisibleNormalPdf(surface.shadingNormal, viewDirection, halfway,
                                                   surface.roughness) /
                                  max(4.0 * vDotH, 1e-6);
    float diffusePdf = cosineHemispherePdf(surface.shadingNormal, lightDirection);
    return probabilities.diffuse * diffusePdf + probabilities.specular * specularPdf +
           probabilities.clearcoat * evaluateClearcoatPdf(surface, viewDirection, lightDirection) +
           probabilities.sheen * sheenReflectionPdf(surface, viewDirection, lightDirection);
}

float evaluateBsdfPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return evaluateBsdfPdfWithIors(surface, viewDirection, lightDirection,
                                   entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
}

BsdfSample sampleBsdfWithIors(SurfaceData surface, float3 viewDirection, float2 directionSample,
                              float lobeSample, float etaI, float etaT)
{
    BsdfSample sample = invalidBsdfSample();
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return sample;

    LobeProbabilities probabilities = lobeProbabilities(surface, viewDirection, etaI, etaT);
    float diffuseEnd = probabilities.diffuse;
    float specularEnd = diffuseEnd + probabilities.specular;
    float clearcoatEnd = specularEnd + probabilities.clearcoat;
    float transmissionEnd = clearcoatEnd + probabilities.transmission;
    bool sampleSpecular = lobeSample >= diffuseEnd && lobeSample < specularEnd;
    bool sampleClearcoat = lobeSample >= specularEnd && lobeSample < clearcoatEnd;
    bool sampleTransmission = lobeSample >= clearcoatEnd && lobeSample < transmissionEnd;
    bool sampleSheen = lobeSample >= transmissionEnd;
    if (sampleSheen)
    {
        float3 halfway = sampleSheenHalfway(surface, directionSample);
        sample.direction = normalize(reflect(-viewDirection, halfway));
        sample.lobe = uint(BsdfLobe::SheenLobe);
    }
    else if (sampleTransmission)
    {
        bool refracted = sampleRoughTransmissionDirection(surface, viewDirection, directionSample,
                                                           etaI, etaT, sample.direction);
        sample.lobe = uint(refracted ? BsdfLobe::SpecularTransmissionLobe
                                     : BsdfLobe::SpecularReflectionLobe);
        sample.transmission = refracted;
        sample.delta = surface.roughness <= 0.021;
    }
    else if (sampleClearcoat)
    {
        float clearcoatNDotV = saturate(dot(surface.clearcoatNormal, viewDirection));
        if (clearcoatNDotV <= 0.0)
            return sample;
        float3 clearcoatTangent = normalize(surface.tangent - surface.clearcoatNormal *
                                             dot(surface.clearcoatNormal, surface.tangent));
        float3 clearcoatBitangent = normalize(cross(surface.clearcoatNormal, clearcoatTangent));
        float3 localView = float3(dot(viewDirection, clearcoatTangent),
                                  dot(viewDirection, clearcoatBitangent), clearcoatNDotV);
        float3 localHalfway = sampleGgxVisibleNormalLocal(localView, ggxAlpha(surface.clearcoatRoughness),
                                                          directionSample);
        float3 halfway = normalize(clearcoatTangent * localHalfway.x +
                                   clearcoatBitangent * localHalfway.y +
                                   surface.clearcoatNormal * localHalfway.z);
        sample.direction = normalize(reflect(-viewDirection, halfway));
        sample.lobe = uint(BsdfLobe::ClearcoatLobe);
        sample.delta = surface.clearcoatRoughness <= 0.021;
    }
    else if (sampleSpecular)
    {
        float3 halfway;
        if (abs(surface.anisotropy) > 1e-4)
            halfway = sampleAnisotropicVisibleNormal(surface, viewDirection, directionSample);
        else
        {
            float3 localView = float3(dot(viewDirection, surface.tangent),
                                      dot(viewDirection, surface.bitangent), nDotV);
            float3 localHalfway = sampleGgxVisibleNormalLocal(localView, ggxAlpha(surface.roughness),
                                                              directionSample);
            halfway = normalize(surface.tangent * localHalfway.x + surface.bitangent * localHalfway.y +
                                 surface.shadingNormal * localHalfway.z);
        }
        sample.direction = normalize(reflect(-viewDirection, halfway));
        sample.lobe = uint(BsdfLobe::SpecularReflectionLobe);
        sample.delta = surface.roughness <= 0.021;
    }
    else
    {
        float radius = sqrt(directionSample.x);
        float phi = 2.0 * PI * directionSample.y;
        float x = radius * cos(phi);
        float y = radius * sin(phi);
        float z = sqrt(max(1.0 - directionSample.x, 0.0));
        sample.direction = normalize(surface.tangent * x + surface.bitangent * y + surface.shadingNormal * z);
        sample.lobe = uint(BsdfLobe::DiffuseLobe);
    }

    float nDotL = abs(dot(surface.shadingNormal, sample.direction));
    if (nDotL <= 0.0)
        return invalidBsdfSample();
    sample.pdf = evaluateBsdfPdfWithIors(surface, viewDirection, sample.direction, etaI, etaT);
    if (sample.pdf <= 1e-8)
        return invalidBsdfSample();
    sample.weight = evaluateBsdfWithIors(surface, viewDirection, sample.direction, etaI, etaT) *
                    nDotL / sample.pdf;
    return sample;
}

BsdfSample sampleBsdf(SurfaceData surface, float3 viewDirection, float2 directionSample, float lobeSample)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return sampleBsdfWithIors(surface, viewDirection, directionSample, lobeSample,
                              entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
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
    GpuMaterial material;
    material.baseColorFactor = float4(baseColor, 1.0);
    material.emissiveAndMetallic = float4(0.0, 0.0, 0.0, metallic);
    material.surfaceParameters = float4(roughness, 1.0, 1.0, 0.5);
    material.transmissionClearcoat = float4(0.0, 1.5, 0.0, 0.0);
    material.anisotropySheen = float4(0.0);
    material.sheenColorAbsorptionDistance = float4(0.0, 0.0, 0.0, 1e30);
    material.absorptionColorSubsurface = float4(1.0, 1.0, 1.0, 0.0);
    material.subsurfaceColorRadius = float4(1.0);
    material.volumeAbsorptionDensity = float4(0.0);
    material.volumeScatteringAnisotropy = float4(0.0);
    material.materialFlags = uint4(0u);
    material.textureIndices0 = uint4(INVALID_TEXTURE_ID);
    material.textureIndices1 = uint4(INVALID_TEXTURE_ID);
    material.textureIndices2 = uint4(INVALID_TEXTURE_ID);
    SurfaceData surface = makeSurfaceData(material, float3(0.0), normal, normal,
                                          float4(1.0, 0.0, 0.0, 1.0), float2(0.0));
    return evaluateBsdf(surface, viewDirection, lightDirection) *
           saturate(dot(normal, lightDirection)) * radiance;
}
