import MaterialTypes;
import Fresnel;
import Microfacet;
import BsdfTypes;
import BsdfDiffuse;
import BsdfConductor;
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

LobeProbabilities controlledLobeProbabilities(SurfaceData surface, float3 viewDirection, float etaI, float etaT,
                                               bool enableDiffuse, bool enableReflection,
                                               bool enableTransmission)
{
    LobeProbabilities probabilities;
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
    float3 viewFresnel = fresnelSchlick(nDotV, f0);
    float specularWeight = enableReflection ? max(luminance(viewFresnel), 1e-5) : 0.0;
    float diffuseWeight = enableDiffuse ? max(luminance(surface.baseColor) * (1.0 - surface.metallic) *
                                  (1.0 - surface.transmission) * (1.0 - luminance(viewFresnel)),
                              0.0) : 0.0;
    float dielectricFresnel = fresnelDielectric(nDotV, etaI, etaT);
    float transmissionWeight = enableTransmission ? surface.transmission * (1.0 - surface.metallic) *
                               (1.0 - dielectricFresnel) * max(luminance(surface.baseColor), 0.05) : 0.0;
    float clearcoatWeight = enableReflection ? surface.clearcoat * 0.15 : 0.0;
    float sheenWeight = enableReflection ? sheenMaximumColor(surface) *
                                           (0.2 + 0.3 * surface.sheenRoughness) : 0.0;
    float total = specularWeight + diffuseWeight + transmissionWeight + clearcoatWeight + sheenWeight;
    if (total <= 1e-6)
    {
        probabilities.diffuse = 0.0;
        probabilities.specular = 0.0;
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

LobeProbabilities lobeProbabilities(SurfaceData surface, float3 viewDirection, float etaI, float etaT)
{
    return controlledLobeProbabilities(surface, viewDirection, etaI, etaT, true, true, true);
}

float3 evaluateBsdfWithIorsControlled(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                      float etaI, float etaT, bool enableDiffuse,
                                      bool enableReflection, bool enableTransmission)
{
    float signedNDotL = dot(surface.shadingNormal, lightDirection);
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return float3(0.0);
    if (signedNDotL < 0.0)
        return enableTransmission ? evaluateRoughTransmission(surface, viewDirection, lightDirection,
                                                               etaI, etaT) : float3(0.0);
    if (signedNDotL == 0.0)
        return float3(0.0);

    float3 halfwayVector = viewDirection + lightDirection;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return float3(0.0);
    float3 halfway = normalize(halfwayVector);
    float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float3 specular = !enableReflection ? float3(0.0) : abs(surface.anisotropy) > 1e-4
                          ? evaluateAnisotropicGgx(f0, surface, viewDirection, lightDirection)
                          : evaluateConductorGgx(f0, surface.roughness, surface.shadingNormal,
                                                 viewDirection, lightDirection);
    float3 diffuse = enableDiffuse
                         ? (1.0 - fresnel) * (1.0 - surface.metallic) *
                               (1.0 - surface.transmission) * evaluateLambert(surface.baseColor)
                         : float3(0.0);
    float attenuation = 1.0 - clearcoatFresnel(surface, viewDirection, lightDirection);
    float3 tirReflection = enableTransmission
                               ? evaluateRoughTransmissionTirReflection(surface, viewDirection, lightDirection,
                                                                         etaI, etaT)
                               : float3(0.0);
    float3 layeredBase = (diffuse + specular + tirReflection) * attenuation +
                         (enableReflection ? evaluateClearcoat(surface, viewDirection, lightDirection)
                                           : float3(0.0));
    return layeredBase * sheenEnergyScale(surface) +
           (enableReflection ? evaluateSheen(surface, viewDirection, lightDirection) : float3(0.0));
}

float3 evaluateBsdfWithIors(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                            float etaI, float etaT)
{
    return evaluateBsdfWithIorsControlled(surface, viewDirection, lightDirection, etaI, etaT,
                                           true, true, true);
}

float3 evaluateBsdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return evaluateBsdfWithIors(surface, viewDirection, lightDirection,
                                entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
}

float evaluateBsdfPdfWithIorsControlled(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                        float etaI, float etaT, bool enableDiffuse,
                                        bool enableReflection, bool enableTransmission)
{
    float signedNDotL = dot(surface.shadingNormal, lightDirection);
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return 0.0;
    LobeProbabilities probabilities = controlledLobeProbabilities(surface, viewDirection, etaI, etaT,
                                                                   enableDiffuse, enableReflection,
                                                                   enableTransmission);
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
    float tirPdf = probabilities.transmission * roughTransmissionTirReflectionPdf(
        surface, viewDirection, lightDirection, etaI, etaT);
    return probabilities.diffuse * diffusePdf + probabilities.specular * specularPdf + tirPdf +
           probabilities.clearcoat * evaluateClearcoatPdf(surface, viewDirection, lightDirection) +
           probabilities.sheen * sheenReflectionPdf(surface, viewDirection, lightDirection);
}

float evaluateBsdfPdfWithIors(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                              float etaI, float etaT)
{
    return evaluateBsdfPdfWithIorsControlled(surface, viewDirection, lightDirection, etaI, etaT,
                                              true, true, true);
}

float evaluateBsdfPdf(SurfaceData surface, float3 viewDirection, float3 lightDirection)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return evaluateBsdfPdfWithIors(surface, viewDirection, lightDirection,
                                   entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
}

BsdfSample sampleBsdfWithIorsControlled(SurfaceData surface, float3 viewDirection, float2 directionSample,
                                        float lobeSample, float etaI, float etaT, bool enableDiffuse,
                                        bool enableReflection, bool enableTransmission)
{
    BsdfSample sample = invalidBsdfSample();
    float nDotV = saturate(dot(surface.shadingNormal, viewDirection));
    if (nDotV <= 0.0)
        return sample;

    LobeProbabilities probabilities = controlledLobeProbabilities(surface, viewDirection, etaI, etaT,
                                                                   enableDiffuse, enableReflection,
                                                                   enableTransmission);
    if (probabilities.diffuse + probabilities.specular + probabilities.clearcoat +
        probabilities.transmission + probabilities.sheen <= 0.0)
        return sample;
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
        sample.delta = false;
    }
    else if (sampleClearcoat)
    {
        float clearcoatNDotV = saturate(dot(surface.clearcoatNormal, viewDirection));
        if (clearcoatNDotV <= 0.0)
            return sample;
        float3 clearcoatTangent = surface.tangent - surface.clearcoatNormal *
                                  dot(surface.clearcoatNormal, surface.tangent);
        if (dot(clearcoatTangent, clearcoatTangent) <= 1e-10)
            clearcoatTangent = cross(abs(surface.clearcoatNormal.z) < 0.999
                                         ? float3(0.0, 0.0, 1.0)
                                         : float3(0.0, 1.0, 0.0),
                                     surface.clearcoatNormal);
        clearcoatTangent = normalize(clearcoatTangent);
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
        sample.delta = false;
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
        sample.delta = false;
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
    sample.pdf = evaluateBsdfPdfWithIorsControlled(surface, viewDirection, sample.direction, etaI, etaT,
                                                    enableDiffuse, enableReflection, enableTransmission);
    if (sample.pdf <= 1e-8)
        return invalidBsdfSample();
    sample.weight = evaluateBsdfWithIorsControlled(surface, viewDirection, sample.direction, etaI, etaT,
                                                    enableDiffuse, enableReflection, enableTransmission) *
                    nDotL / sample.pdf;
    return sample;
}

BsdfSample sampleBsdfWithIors(SurfaceData surface, float3 viewDirection, float2 directionSample,
                              float lobeSample, float etaI, float etaT)
{
    return sampleBsdfWithIorsControlled(surface, viewDirection, directionSample, lobeSample, etaI, etaT,
                                         true, true, true);
}

BsdfSample sampleBsdf(SurfaceData surface, float3 viewDirection, float2 directionSample, float lobeSample)
{
    bool entering = dot(surface.geometricNormal, viewDirection) >= 0.0;
    return sampleBsdfWithIors(surface, viewDirection, directionSample, lobeSample,
                              entering ? 1.0 : surface.ior, entering ? surface.ior : 1.0);
}

float3 evaluateBsdfWithSubsurfaceMix(SurfaceData surface, float3 viewDirection, float3 lightDirection,
                                     float etaI, float etaT)
{
    float3 result = evaluateBsdfWithIors(surface, viewDirection, lightDirection, etaI, etaT);
    if (surface.subsurface <= 0.0 || dot(surface.shadingNormal, lightDirection) <= 0.0)
        return result;
    float3 halfwayVector = viewDirection + lightDirection;
    if (dot(halfwayVector, halfwayVector) <= 1e-10)
        return result;
    float3 halfway = normalize(halfwayVector);
    float3 f0 = lerp(float3(dielectricF0(surface.ior)), surface.baseColor, surface.metallic);
    float3 fresnel = fresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float3 diffuse = (1.0 - fresnel) * (1.0 - surface.metallic) *
                     (1.0 - surface.transmission) * evaluateLambert(surface.baseColor);
    float attenuation = 1.0 - clearcoatFresnel(surface, viewDirection, lightDirection);
    return max(result - diffuse * attenuation * sheenEnergyScale(surface) * surface.subsurface, 0.0);
}
