import MaterialTypes;

struct MaterialTextureSamples
{
    float4 baseColor;
    float4 metallicRoughness;
    float4 normal;
    float4 emissive;
    float4 occlusion;
    float4 transmission;
    float4 clearcoat;
    float4 clearcoatRoughness;
    float4 clearcoatNormal;
    float4 anisotropy;
    float4 sheenColor;
    float4 sheenRoughness;
};

MaterialTextureSamples defaultMaterialTextureSamples()
{
    MaterialTextureSamples samples;
    samples.baseColor = float4(1.0);
    samples.metallicRoughness = float4(1.0);
    samples.normal = float4(0.5, 0.5, 1.0, 1.0);
    samples.emissive = float4(1.0);
    samples.occlusion = float4(1.0);
    samples.transmission = float4(1.0);
    samples.clearcoat = float4(1.0);
    samples.clearcoatRoughness = float4(1.0);
    samples.clearcoatNormal = float4(0.5, 0.5, 1.0, 1.0);
    samples.anisotropy = float4(1.0, 0.5, 1.0, 1.0);
    samples.sheenColor = float4(1.0);
    samples.sheenRoughness = float4(1.0);
    return samples;
}

SurfaceData applyBaseMaterialTextures(
    GpuMaterial material,
    SurfaceData surface,
    MaterialTextureSamples samples)
{
    float3 originalNormal = surface.shadingNormal;
    float3 originalTangent = surface.tangent;
    float3 originalBitangent = surface.bitangent;
    float tangentSign = dot(cross(originalNormal, originalTangent), originalBitangent) < 0.0 ? -1.0 : 1.0;
    surface.baseColor *= max(samples.baseColor.xyz, 0.0);
    surface.alpha *= saturate(samples.baseColor.w);
    surface.roughness = clamp(surface.roughness * saturate(samples.metallicRoughness.y), 0.02, 1.0);
    surface.metallic = saturate(surface.metallic * saturate(samples.metallicRoughness.z));
    surface.emissive *= max(samples.emissive.xyz, 0.0);
    surface.occlusion = lerp(1.0, saturate(samples.occlusion.x), saturate(material.surfaceParameters.z));
    surface.transmission = saturate(surface.transmission * saturate(samples.transmission.x));
    surface.clearcoat = saturate(surface.clearcoat * saturate(samples.clearcoat.x));
    surface.clearcoatRoughness = clamp(surface.clearcoatRoughness *
                                           saturate(samples.clearcoatRoughness.y),
                                       0.02, 1.0);

    if (material.textureIndices0.z != INVALID_TEXTURE_ID)
    {
        float3 tangentNormal = samples.normal.xyz * 2.0 - 1.0;
        tangentNormal.xy *= max(material.surfaceParameters.y, 0.0);
        tangentNormal = normalize(tangentNormal);
        surface.shadingNormal = normalize(surface.tangent * tangentNormal.x +
                                          surface.bitangent * tangentNormal.y +
                                          surface.shadingNormal * tangentNormal.z);
        float3 mappedTangent = originalTangent - surface.shadingNormal *
                                                   dot(surface.shadingNormal, originalTangent);
        if (dot(mappedTangent, mappedTangent) <= 1e-10)
            mappedTangent = cross(abs(surface.shadingNormal.z) < 0.999
                                      ? float3(0.0, 0.0, 1.0)
                                      : float3(0.0, 1.0, 0.0),
                                  surface.shadingNormal);
        surface.tangent = normalize(mappedTangent);
        surface.bitangent = normalize(cross(surface.shadingNormal, surface.tangent)) * tangentSign;
        surface.clearcoatNormal = surface.shadingNormal;
    }
    if (material.textureIndices2.x != INVALID_TEXTURE_ID)
    {
        float3 tangentNormal = normalize(samples.clearcoatNormal.xyz * 2.0 - 1.0);
        surface.clearcoatNormal = normalize(originalTangent * tangentNormal.x +
                                            originalBitangent * tangentNormal.y +
                                            originalNormal * tangentNormal.z);
    }
    if (material.textureIndices2.w != INVALID_TEXTURE_ID)
    {
        float2 direction = samples.anisotropy.xy * 2.0 - 1.0;
        if (dot(direction, direction) > 1e-8)
            surface.anisotropyRotation += atan2(direction.y, direction.x);
        surface.anisotropy = clamp(surface.anisotropy * saturate(samples.anisotropy.z), -0.99, 0.99);
    }
    surface.sheenColor *= max(samples.sheenColor.xyz, 0.0);
    surface.sheenRoughness = saturate(surface.sheenRoughness * saturate(samples.sheenRoughness.w));
    return surface;
}
