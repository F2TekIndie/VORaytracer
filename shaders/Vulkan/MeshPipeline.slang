import Pbr;

struct TaskPayload
{
    uint meshletIndices[32];
};

groupshared TaskPayload taskPayload;
groupshared uint visibleMeshletCount;

struct GpuVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 uv;
};

struct GpuMeshlet
{
    uint vertexOffset;
    uint triangleOffset;
    uint vertexCount;
    uint triangleCount;
    uint instanceIndex;
    uint padding0;
    uint padding1;
    uint padding2;
    float4 boundingSphere;
    float4 normalCone;
    float4 baseColorAndMetallic;
    float4 materialParameters;
};

struct GpuInstance
{
    float4x4 transform;
    float4x4 normalTransform;
    float4 scaleAndFlags;
    uint4 geometry;
    float4 baseColorAndMetallic;
    float4 emissiveAndRoughness;
};

struct FramePushConstants
{
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 cameraForwardAndFov;
    float4 cameraRightAndAspect;
    float4 cameraUp;
    float4 lightColorAndIntensity;
    float4 lightDirectionAndMode;
    float4 environmentParameters;
    float4 skyZenithAndVisibility;
    float4 skyHorizonAndIndirect;
    float4 skyGround;
    uint rayTracedShadows;
    uint rayTracedReflections;
    float exposure;
    uint meshletCount;
    uint instanceCount;
    uint showMeshlets;
    uint postProcessInput;
    uint environmentMipCount;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex> sceneVertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuMeshlet> sceneMeshlets;
[[vk::binding(2, 0)]] StructuredBuffer<uint> meshletVertexIndices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> meshletTriangleIndices;
[[vk::binding(4, 0)]] RaytracingAccelerationStructure sceneTlas;
[[vk::binding(5, 0)]] StructuredBuffer<GpuInstance> sceneInstances;
[[vk::binding(6, 0)]] StructuredBuffer<uint> geometryIndices;
[[vk::binding(7, 0)]] StructuredBuffer<float4> environmentPixels;
[[vk::push_constant]] ConstantBuffer<FramePushConstants> frame;

struct MeshVertex
{
    float4 position : SV_Position;
    float3 worldNormal : NORMAL0;
    float3 worldPosition : POSITION0;
    float2 uv : TEXCOORD0;
    nointerpolation float4 baseColorAndMetallic : TEXCOORD1;
    nointerpolation float roughness : TEXCOORD2;
    nointerpolation uint meshletIndex : TEXCOORD3;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
};

struct BackgroundVertexOutput
{
    float4 position : SV_Position;
    float3 direction : TEXCOORD0;
};

uint meshletHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float3 meshletDebugColor(uint meshletIndex)
{
    uint hash = meshletHash(meshletIndex + 1u);
    float3 color = float3(float(hash & 0xffu),
                          float((hash >> 8) & 0xffu),
                          float((hash >> 16) & 0xffu)) / 255.0;
    return 0.2 + color * 0.8;
}

float3 rotateEnvironmentDirection(float3 direction)
{
    float angle = frame.environmentParameters.w;
    float sine = sin(angle);
    float cosine = cos(angle);
    float3 yawRotated = float3(cosine * direction.x - sine * direction.z,
                               direction.y,
                               sine * direction.x + cosine * direction.z);
    float pitch = asin(clamp(frame.lightDirectionAndMode.y, -1.0, 1.0));
    float pitchSine = sin(pitch);
    float pitchCosine = cos(pitch);
    return float3(yawRotated.x,
                  pitchCosine * yawRotated.y + pitchSine * yawRotated.z,
                  -pitchSine * yawRotated.y + pitchCosine * yawRotated.z);
}

uint environmentMipOffset(uint mipLevel)
{
    uint offset = 0;
    uint width = uint(frame.environmentParameters.x);
    uint height = uint(frame.environmentParameters.y);
    for (uint mip = 0; mip < mipLevel; ++mip)
    {
        offset += width * height;
        width = max(width / 2, 1u);
        height = max(height / 2, 1u);
    }
    return offset;
}

float3 sampleHdrEnvironmentLevel(float3 rotatedDirection, uint mipLevel)
{
    uint width = max(uint(frame.environmentParameters.x) >> mipLevel, 1u);
    uint height = max(uint(frame.environmentParameters.y) >> mipLevel, 1u);
    if (width == 0 || height == 0)
        return float3(0.0);
    float2 uv = float2(atan2(rotatedDirection.z, rotatedDirection.x) / (2.0 * PI) + 0.5,
                            acos(clamp(rotatedDirection.y, -1.0, 1.0)) / PI);
    float2 texel = uv * float2(width, height) - 0.5;
    int2 base = int2(floor(texel));
    float2 fraction = frac(texel);
    uint x0 = uint((base.x % int(width) + int(width)) % int(width));
    uint x1 = (x0 + 1) % width;
    uint y0 = uint(clamp(base.y, 0, int(height) - 1));
    uint y1 = min(y0 + 1, height - 1);
    uint offset = environmentMipOffset(mipLevel);
    float3 a = lerp(environmentPixels[offset + y0 * width + x0].xyz,
                    environmentPixels[offset + y0 * width + x1].xyz, fraction.x);
    float3 b = lerp(environmentPixels[offset + y1 * width + x0].xyz,
                    environmentPixels[offset + y1 * width + x1].xyz, fraction.x);
    return lerp(a, b, fraction.y);
}

float3 sampleHdrEnvironmentLod(float3 direction, float mipLevel)
{
    if (frame.environmentMipCount == 0)
        return float3(0.0);
    float3 rotatedDirection = rotateEnvironmentDirection(normalize(direction));
    float clampedLevel = clamp(mipLevel, 0.0, float(frame.environmentMipCount - 1));
    uint lowerLevel = uint(floor(clampedLevel));
    uint upperLevel = min(lowerLevel + 1, frame.environmentMipCount - 1);
    float blend = frac(clampedLevel);
    return lerp(sampleHdrEnvironmentLevel(rotatedDirection, lowerLevel),
                sampleHdrEnvironmentLevel(rotatedDirection, upperLevel), blend);
}

float3 sampleHdrEnvironment(float3 direction)
{
    return sampleHdrEnvironmentLod(direction, 0.0);
}

float3 sampleGlobalEnvironment(float3 direction)
{
    uint mode = uint(frame.lightDirectionAndMode.w + 0.5);
    if (mode == 1)
        return sampleHdrEnvironment(direction) * frame.environmentParameters.z;
    if (mode == 2)
    {
        float3 normalizedDirection = normalize(direction);
        float upperBlend = pow(saturate(normalizedDirection.y), 0.45);
        float lowerBlend = pow(saturate(-normalizedDirection.y), 0.35);
        float3 sky = lerp(frame.skyHorizonAndIndirect.xyz, frame.skyZenithAndVisibility.xyz, upperBlend);
        sky = lerp(sky, frame.skyGround.xyz, lowerBlend);
        float sun = pow(saturate(dot(normalizedDirection, normalize(-frame.lightDirectionAndMode.xyz))), 512.0);
        return (sky + frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * sun) *
               frame.environmentParameters.z;
    }
    return float3(0.0);
}

float hdrSpecularMipLevel(float roughness)
{
    if (frame.environmentMipCount <= 1)
        return 0.0;
    float maximumMip = float(frame.environmentMipCount - 1);
    float angularFootprintInTexels = max(roughness * roughness * frame.environmentParameters.x /
                                         (2.0 * PI), 1.0);
    float footprintMip = log2(angularFootprintInTexels) + 0.5;
    float perceptualMip = roughness * maximumMip * 1.75;
    return min(max(footprintMip, perceptualMip), maximumMip);
}

float3 sampleDiffuseEnvironment(float3 normal)
{
    uint mode = uint(frame.lightDirectionAndMode.w + 0.5);
    if (mode == 1)
    {
        float hemisphereFootprint = max(frame.environmentParameters.x / (2.0 * PI), 1.0);
        float mipLevel = min(log2(hemisphereFootprint), float(max(frame.environmentMipCount, 1u) - 1));
        return sampleHdrEnvironmentLod(normal, mipLevel) * frame.environmentParameters.z;
    }
    return sampleGlobalEnvironment(normal);
}

float3 samplePrefilteredEnvironment(float3 reflectionDirection, float roughness)
{
    reflectionDirection = normalize(reflectionDirection);
    roughness = saturate(roughness);
    if (uint(frame.lightDirectionAndMode.w + 0.5) == 1)
        return sampleHdrEnvironmentLod(reflectionDirection, hdrSpecularMipLevel(roughness)) *
               frame.environmentParameters.z;
    return sampleGlobalEnvironment(reflectionDirection);
}

float hash01(float3 value)
{
    return frac(sin(dot(value, float3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float3 traceDiffuseBounce(float3 worldPosition, float3 normal)
{
    float coordinateScale = max(max(abs(worldPosition.x), abs(worldPosition.y)), abs(worldPosition.z));
    float rayBias = max(0.0001, coordinateScale * 0.00001);
    float random0 = hash01(worldPosition + normal * 17.0);
    float random1 = hash01(worldPosition.zxy * 3.1 + normal.yzx * 29.0);
    float phi = 2.0 * PI * random0;
    float cosTheta = sqrt(1.0 - random1);
    float sinTheta = sqrt(random1);
    float3 tangent = normalize(cross(abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0)
                                                               : float3(0.0, 1.0, 0.0), normal));
    float3 bitangent = cross(normal, tangent);
    float3 bounceDirection = normalize(tangent * cos(phi) * sinTheta +
                                       bitangent * sin(phi) * sinTheta + normal * cosTheta);
    RayDesc bounceRay;
    bounceRay.Origin = worldPosition + normal * rayBias;
    bounceRay.Direction = bounceDirection;
    bounceRay.TMin = rayBias * 0.5;
    bounceRay.TMax = 10000.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> bounceQuery;
    bounceQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, bounceRay);
    bounceQuery.Proceed();
    if (bounceQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return sampleDiffuseEnvironment(normal);

    GpuInstance hitInstance = sceneInstances[bounceQuery.CommittedInstanceID()];
    uint triangleOffset = hitInstance.geometry.y + bounceQuery.CommittedPrimitiveIndex() * 3;
    uint3 triangle = uint3(geometryIndices[triangleOffset], geometryIndices[triangleOffset + 1],
                           geometryIndices[triangleOffset + 2]);
    float2 barycentrics = bounceQuery.CommittedTriangleBarycentrics();
    float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
    float3 hitNormal = normalize(sceneVertices[hitInstance.geometry.x + triangle.x].normal.xyz * barycentric0 +
                                 sceneVertices[hitInstance.geometry.x + triangle.y].normal.xyz * barycentrics.x +
                                 sceneVertices[hitInstance.geometry.x + triangle.z].normal.xyz * barycentrics.y);
    hitNormal = normalize(mul((float3x3)hitInstance.normalTransform, hitNormal));
    float3 hitViewDirection = -bounceDirection;
    if (dot(hitNormal, hitViewDirection) < 0.0)
        hitNormal = -hitNormal;
    float3 hitPosition = bounceRay.Origin + bounceDirection * bounceQuery.CommittedRayT();
    float3 lightDirection = normalize(-frame.lightDirectionAndMode.xyz);
    float visibility = 1.0;
    if (frame.rayTracedShadows != 0 && uint(frame.lightDirectionAndMode.w + 0.5) != 1)
    {
        float hitScale = max(max(abs(hitPosition.x), abs(hitPosition.y)), abs(hitPosition.z));
        float hitBias = max(0.0001, hitScale * 0.00001);
        RayDesc shadowRay;
        shadowRay.Origin = hitPosition + hitNormal * hitBias;
        shadowRay.Direction = lightDirection;
        shadowRay.TMin = hitBias * 0.5;
        shadowRay.TMax = 10000.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> shadowQuery;
        shadowQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, shadowRay);
        shadowQuery.Proceed();
        visibility = shadowQuery.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
    }
    float3 directRadiance = uint(frame.lightDirectionAndMode.w + 0.5) == 1
                                ? float3(0.0)
                                : frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * visibility;
    float3 radiance = evaluateMetallicRoughness(hitInstance.baseColorAndMetallic.xyz,
                                                 hitInstance.baseColorAndMetallic.w,
                                                 hitInstance.emissiveAndRoughness.w,
                                                 hitNormal, hitViewDirection, lightDirection, directRadiance);
    radiance += sampleDiffuseEnvironment(hitNormal) * hitInstance.baseColorAndMetallic.xyz *
                (1.0 - hitInstance.baseColorAndMetallic.w) / PI;
    return radiance + hitInstance.emissiveAndRoughness.xyz;
}

[shader("vertex")]
BackgroundVertexOutput BackgroundVertex(uint vertexIndex : SV_VertexID)
{
    BackgroundVertexOutput output;
    float2 position = vertexIndex == 0 ? float2(-1.0, -1.0) :
                      vertexIndex == 1 ? float2(-1.0, 3.0) : float2(3.0, -1.0);
    output.position = float4(position, 0.0, 1.0);
    output.direction = normalize(frame.cameraForwardAndFov.xyz +
                                 frame.cameraRightAndAspect.xyz * position.x *
                                     frame.cameraRightAndAspect.w * frame.cameraForwardAndFov.w -
                                 frame.cameraUp.xyz * position.y * frame.cameraForwardAndFov.w);
    return output;
}

[shader("fragment")]
FragmentOutput BackgroundFragment(BackgroundVertexOutput input)
{
    FragmentOutput output;
    float3 color = frame.skyZenithAndVisibility.w != 0.0
                       ? sampleGlobalEnvironment(input.direction) : float3(0.0);
    if (frame.postProcessInput == 0)
    {
        color = max(color * exp2(frame.exposure), 0.0);
        color = color / (1.0 + color);
    }
    output.color = float4(color, 1.0);
    return output;
}

[shader("amplification")]
[numthreads(32, 1, 1)]
void TaskMain(
    uint3 groupIndex : SV_GroupID,
    uint threadIndex : SV_GroupThreadID)
{
    if (threadIndex == 0)
        visibleMeshletCount = 0;
    GroupMemoryBarrierWithGroupSync();

    uint meshletIndex = groupIndex.x * 32 + threadIndex;
    bool visible = meshletIndex < frame.meshletCount;
    if (visible)
    {
        GpuMeshlet meshlet = sceneMeshlets[meshletIndex];
        visible = meshlet.vertexCount <= 64 && meshlet.triangleCount <= 124 &&
                  meshlet.instanceIndex < frame.instanceCount;
        if (visible)
        {
            GpuInstance instance = sceneInstances[meshlet.instanceIndex];
            if (meshlet.boundingSphere.w > 0.0)
            {
                float3 worldCenter = mul(instance.transform, float4(meshlet.boundingSphere.xyz, 1.0)).xyz;
                float4 clipCenter = mul(frame.viewProjection, float4(worldCenter, 1.0));
                float radius = meshlet.boundingSphere.w * instance.scaleAndFlags.x;
                float radiusX = radius * length(float3(frame.viewProjection[0][0], frame.viewProjection[0][1], frame.viewProjection[0][2]));
                float radiusY = radius * length(float3(frame.viewProjection[1][0], frame.viewProjection[1][1], frame.viewProjection[1][2]));
                float radiusZ = radius * length(float3(frame.viewProjection[2][0], frame.viewProjection[2][1], frame.viewProjection[2][2]));
                float radiusW = radius * length(float3(frame.viewProjection[3][0], frame.viewProjection[3][1], frame.viewProjection[3][2]));
                visible = clipCenter.x + radiusX + radiusW >= -clipCenter.w &&
                          clipCenter.x - radiusX - radiusW <= clipCenter.w &&
                          clipCenter.y + radiusY + radiusW >= -clipCenter.w &&
                          clipCenter.y - radiusY - radiusW <= clipCenter.w &&
                          clipCenter.z + radiusZ >= 0.0 &&
                          clipCenter.z - radiusZ - radiusW <= clipCenter.w;

                float coneLength = length(meshlet.normalCone.xyz);
                if (visible && coneLength > 0.5 && meshlet.normalCone.w <= 1.0 &&
                    instance.scaleAndFlags.y >= instance.scaleAndFlags.x * 0.999)
                {
                    float3 cameraToMeshlet = normalize(worldCenter - frame.cameraPosition.xyz);
                    float3 worldConeAxis = normalize(mul((float3x3)instance.normalTransform, meshlet.normalCone.xyz));
                    visible = dot(worldConeAxis, cameraToMeshlet) < meshlet.normalCone.w;
                }
            }
        }
    }

    if (visible)
    {
        uint payloadIndex;
        InterlockedAdd(visibleMeshletCount, 1, payloadIndex);
        taskPayload.meshletIndices[payloadIndex] = meshletIndex;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex == 0)
        DispatchMesh(visibleMeshletCount, 1, 1, taskPayload);
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void MeshMain(
    uint3 groupIndex : SV_GroupID,
    uint threadIndex : SV_GroupThreadID,
    in payload TaskPayload taskPayload,
    out vertices MeshVertex vertices[64],
    out indices uint3 triangles[124])
{
    uint meshletIndex = taskPayload.meshletIndices[groupIndex.x];
    GpuMeshlet meshlet = sceneMeshlets[meshletIndex];
    GpuInstance instance = sceneInstances[meshlet.instanceIndex];
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);
    for (uint outputVertex = threadIndex; outputVertex < meshlet.vertexCount; outputVertex += 32)
    {
        uint vertexIndex = meshletVertexIndices[meshlet.vertexOffset + outputVertex];
        GpuVertex source = sceneVertices[vertexIndex];
        float4 worldPosition = mul(instance.transform, float4(source.position.xyz, 1.0));
        vertices[outputVertex].position = mul(frame.viewProjection, worldPosition);
        vertices[outputVertex].worldPosition = worldPosition.xyz;
        vertices[outputVertex].worldNormal = normalize(mul((float3x3)instance.normalTransform, source.normal.xyz));
        vertices[outputVertex].uv = source.uv.xy;
        vertices[outputVertex].baseColorAndMetallic = meshlet.baseColorAndMetallic;
        vertices[outputVertex].roughness = meshlet.materialParameters.x;
        vertices[outputVertex].meshletIndex = meshletIndex;
    }
    for (uint outputTriangle = threadIndex; outputTriangle < meshlet.triangleCount; outputTriangle += 32)
    {
        uint packed = meshletTriangleIndices[meshlet.triangleOffset + outputTriangle];
        triangles[outputTriangle] = uint3(packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff);
    }
}

[shader("fragment")]
FragmentOutput FragmentMain(MeshVertex input)
{
    FragmentOutput output;
    if (frame.showMeshlets != 0)
    {
        output.color = float4(meshletDebugColor(input.meshletIndex), 1.0);
        return output;
    }
    float3 normal = normalize(input.worldNormal);
    float3 viewDirection = normalize(frame.cameraPosition.xyz - input.worldPosition);
    if (dot(normal, viewDirection) < 0.0)
        normal = -normal;
    uint globalLightMode = uint(frame.lightDirectionAndMode.w + 0.5);
    float lightDistance = 10000.0;
    float3 lightDirection = normalize(-frame.lightDirectionAndMode.xyz);
    float3 baseColor = input.baseColorAndMetallic.xyz;
    float visibility = 1.0;
    if (frame.rayTracedShadows != 0 && globalLightMode != 1)
    {
        float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)), abs(input.worldPosition.z));
        float rayBias = max(0.0001, coordinateScale * 0.00001);
        RayDesc shadowRay;
        shadowRay.Origin = input.worldPosition + normal * rayBias;
        shadowRay.Direction = lightDirection;
        shadowRay.TMin = rayBias * 0.5;
        shadowRay.TMax = max(lightDistance - rayBias, rayBias);
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, shadowRay);
        query.Proceed();
        visibility = query.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
    }
    float3 directRadiance = globalLightMode == 1
                                ? float3(0.0)
                                : frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * visibility;
    float3 color = evaluateMetallicRoughness(baseColor, input.baseColorAndMetallic.w, input.roughness,
                                             normal, viewDirection, lightDirection, directRadiance);
    if (frame.skyHorizonAndIndirect.w != 0.0 && input.baseColorAndMetallic.w < 0.999)
    {
        float3 indirectRadiance = globalLightMode == 1
                                      ? sampleDiffuseEnvironment(normal)
                                      : traceDiffuseBounce(input.worldPosition, normal);
        color += indirectRadiance * baseColor *
                 (1.0 - input.baseColorAndMetallic.w);
    }
    if (frame.rayTracedReflections != 0)
    {
        float coordinateScale = max(max(abs(input.worldPosition.x), abs(input.worldPosition.y)), abs(input.worldPosition.z));
        float rayBias = max(0.0001, coordinateScale * 0.00001);
        RayDesc reflectionRay;
        reflectionRay.Origin = input.worldPosition + normal * rayBias;
        reflectionRay.Direction = normalize(reflect(-viewDirection, normal));
        reflectionRay.TMin = rayBias * 0.5;
        reflectionRay.TMax = 10000.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> reflectionQuery;
        reflectionQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, reflectionRay);
        reflectionQuery.Proceed();

        float3 environmentReflection = samplePrefilteredEnvironment(reflectionRay.Direction,
                                                                     input.roughness);
        float3 reflectedRadiance = environmentReflection;
        if (reflectionQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint instanceIndex = reflectionQuery.CommittedInstanceID();
            GpuInstance hitInstance = sceneInstances[instanceIndex];
            uint primitiveIndex = reflectionQuery.CommittedPrimitiveIndex();
            uint triangleOffset = hitInstance.geometry.y + primitiveIndex * 3;
            uint3 triangle = uint3(geometryIndices[triangleOffset],
                                   geometryIndices[triangleOffset + 1],
                                   geometryIndices[triangleOffset + 2]);
            float2 barycentrics = reflectionQuery.CommittedTriangleBarycentrics();
            float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
            float3 hitNormal = normalize(sceneVertices[hitInstance.geometry.x + triangle.x].normal.xyz * barycentric0 +
                                         sceneVertices[hitInstance.geometry.x + triangle.y].normal.xyz * barycentrics.x +
                                         sceneVertices[hitInstance.geometry.x + triangle.z].normal.xyz * barycentrics.y);
            hitNormal = normalize(mul((float3x3)hitInstance.normalTransform, hitNormal));
            float3 reflectedViewDirection = normalize(-reflectionRay.Direction);
            if (dot(hitNormal, reflectedViewDirection) < 0.0)
                hitNormal = -hitNormal;
            float3 hitPosition = reflectionRay.Origin + reflectionRay.Direction * reflectionQuery.CommittedRayT();
            float hitLightDistance = 10000.0;
            float3 hitLightDirection = normalize(-frame.lightDirectionAndMode.xyz);
            float hitVisibility = 1.0;
            if (frame.rayTracedShadows != 0 && globalLightMode != 1)
            {
                float hitCoordinateScale = max(max(abs(hitPosition.x), abs(hitPosition.y)), abs(hitPosition.z));
                float hitRayBias = max(0.0001, hitCoordinateScale * 0.00001);
                RayDesc reflectedShadowRay;
                reflectedShadowRay.Origin = hitPosition + hitNormal * hitRayBias;
                reflectedShadowRay.Direction = hitLightDirection;
                reflectedShadowRay.TMin = hitRayBias * 0.5;
                reflectedShadowRay.TMax = max(hitLightDistance - hitRayBias, hitRayBias);
                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> reflectedShadowQuery;
                reflectedShadowQuery.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xff, reflectedShadowRay);
                reflectedShadowQuery.Proceed();
                hitVisibility = reflectedShadowQuery.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
            }
            float3 hitDirectRadiance = globalLightMode == 1
                                           ? float3(0.0)
                                           : frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * hitVisibility;
            float3 localReflection = evaluateMetallicRoughness(hitInstance.baseColorAndMetallic.xyz,
                                                                hitInstance.baseColorAndMetallic.w,
                                                                hitInstance.emissiveAndRoughness.w,
                                                                hitNormal,
                                                                reflectedViewDirection,
                                                                hitLightDirection,
                                                                hitDirectRadiance);
            if (frame.skyHorizonAndIndirect.w != 0.0)
                localReflection += sampleDiffuseEnvironment(hitNormal) *
                                   hitInstance.baseColorAndMetallic.xyz *
                                   (1.0 - hitInstance.baseColorAndMetallic.w) / PI;
            localReflection += hitInstance.emissiveAndRoughness.xyz;

            // A single ray can only represent a sharp local reflection. Fade it into the
            // prefiltered environment once the material lobe is too wide for that sample.
            float localReflectionWeight = 1.0 - smoothstep(0.12, 0.58, saturate(input.roughness));
            reflectedRadiance = lerp(environmentReflection, localReflection, localReflectionWeight);
        }
        float3 reflectionF0 = lerp(float3(0.04), baseColor, input.baseColorAndMetallic.w);
        float3 reflectionWeight = fresnelSchlickRoughness(saturate(dot(normal, viewDirection)),
                                                          reflectionF0, input.roughness);
        color += reflectedRadiance * reflectionWeight;
    }
    if (frame.postProcessInput == 0)
    {
        color *= exp2(frame.exposure);
        color = max(color, 0.0);
        color = color / (1.0 + color);
    }
    output.color = float4(color, 1.0);
    return output;
}
