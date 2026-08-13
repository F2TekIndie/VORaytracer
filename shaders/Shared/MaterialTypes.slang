static const uint MATERIAL_FLAG_DOUBLE_SIDED = 1u << 0;
static const uint MATERIAL_FLAG_ALPHA_MASK = 1u << 1;
static const uint MATERIAL_FLAG_TRANSMISSION = 1u << 2;
static const uint MATERIAL_FLAG_CLEARCOAT = 1u << 3;
static const uint MATERIAL_FLAG_SHEEN = 1u << 4;
static const uint MATERIAL_FLAG_ANISOTROPY = 1u << 5;
static const uint MATERIAL_FLAG_EMISSIVE = 1u << 6;
static const uint MATERIAL_FLAG_SUBSURFACE = 1u << 7;
static const uint MATERIAL_FLAG_VOLUME = 1u << 8;
static const uint INVALID_TEXTURE_ID = 0xffffffffu;

struct GpuMaterial
{
    float4 baseColorFactor;
    float4 emissiveAndMetallic;
    float4 surfaceParameters;
    float4 transmissionClearcoat;
    float4 anisotropySheen;
    float4 sheenColorAbsorptionDistance;
    float4 absorptionColorSubsurface;
    float4 subsurfaceColorRadius;
    float4 volumeAbsorptionDensity;
    float4 volumeScatteringAnisotropy;
    uint4 materialFlags;
    uint4 textureIndices0;
    uint4 textureIndices1;
    uint4 textureIndices2;
};

struct SurfaceData
{
    float3 position;
    float3 geometricNormal;
    float3 shadingNormal;
    float3 clearcoatNormal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    float3 baseColor;
    float3 emissive;
    float alpha;
    float metallic;
    float roughness;
    float occlusion;
    float transmission;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    float anisotropy;
    float anisotropyRotation;
    float3 sheenColor;
    float sheenRoughness;
    float3 absorptionColor;
    float absorptionDistance;
    float subsurface;
    float3 subsurfaceColor;
    float subsurfaceRadius;
    float3 volumeAbsorption;
    float3 volumeScattering;
    float volumeDensity;
    float volumeAnisotropy;
    uint flags;
};

bool materialHasFlag(GpuMaterial material, uint flag)
{
    return (material.materialFlags.x & flag) != 0;
}

SurfaceData makeSurfaceData(
    GpuMaterial material,
    float3 position,
    float3 geometricNormal,
    float3 shadingNormal,
    float4 tangentAndSign,
    float2 uv)
{
    SurfaceData surface;
    surface.position = position;
    surface.geometricNormal = normalize(geometricNormal);
    surface.shadingNormal = normalize(shadingNormal);
    surface.clearcoatNormal = surface.shadingNormal;
    float3 tangent = tangentAndSign.xyz - surface.shadingNormal * dot(surface.shadingNormal, tangentAndSign.xyz);
    surface.tangent = length(tangent) > 1e-5 ? normalize(tangent) :
        normalize(cross(abs(surface.shadingNormal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0),
                        surface.shadingNormal));
    surface.bitangent = normalize(cross(surface.shadingNormal, surface.tangent)) *
                        (tangentAndSign.w < 0.0 ? -1.0 : 1.0);
    surface.uv = uv;
    surface.baseColor = max(material.baseColorFactor.xyz, 0.0);
    surface.alpha = saturate(material.baseColorFactor.w);
    surface.emissive = max(material.emissiveAndMetallic.xyz, 0.0);
    surface.metallic = saturate(material.emissiveAndMetallic.w);
    surface.roughness = clamp(material.surfaceParameters.x, 0.02, 1.0);
    surface.occlusion = 1.0;
    surface.transmission = saturate(material.transmissionClearcoat.x);
    surface.ior = max(material.transmissionClearcoat.y, 1.0001);
    surface.clearcoat = saturate(material.transmissionClearcoat.z);
    surface.clearcoatRoughness = clamp(material.transmissionClearcoat.w, 0.02, 1.0);
    surface.anisotropy = clamp(material.anisotropySheen.x, -0.99, 0.99);
    surface.anisotropyRotation = material.anisotropySheen.y;
    surface.sheenRoughness = saturate(material.anisotropySheen.z);
    surface.sheenColor = max(material.sheenColorAbsorptionDistance.xyz, 0.0);
    surface.absorptionDistance = material.sheenColorAbsorptionDistance.w;
    surface.absorptionColor = max(material.absorptionColorSubsurface.xyz, 1e-6);
    surface.subsurface = saturate(material.absorptionColorSubsurface.w);
    surface.subsurfaceColor = max(material.subsurfaceColorRadius.xyz, 0.0);
    surface.subsurfaceRadius = max(material.subsurfaceColorRadius.w, 1e-4);
    surface.volumeAbsorption = max(material.volumeAbsorptionDensity.xyz, 0.0);
    surface.volumeDensity = max(material.volumeAbsorptionDensity.w, 0.0);
    surface.volumeScattering = max(material.volumeScatteringAnisotropy.xyz, 0.0);
    surface.volumeAnisotropy = clamp(material.volumeScatteringAnisotropy.w, -0.99, 0.99);
    surface.flags = material.materialFlags.x;
    return surface;
}
