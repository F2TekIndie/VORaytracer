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

float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    bool entering = cosThetaI > 0.0;
    if (!entering)
    {
        float temporary = etaI;
        etaI = etaT;
        etaT = temporary;
        cosThetaI = abs(cosThetaI);
    }
    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0)
        return 1.0;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float parallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                     max((etaT * cosThetaI) + (etaI * cosThetaT), 1e-6);
    float perpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                          max((etaI * cosThetaI) + (etaT * cosThetaT), 1e-6);
    return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}
