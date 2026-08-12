import Pbr;

struct TaskPayload
{
    uint meshletIndex;
};

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
    float4x4 model;
    float4 cameraPosition;
    float4 lightPosition;
    float4 lightColorAndIntensity;
    uint rayTracedShadows;
    uint rayTracedReflections;
    float exposure;
    uint meshletCount;
    uint instanceCount;
    uint showMeshlets;
    uint padding1;
    uint padding2;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex> sceneVertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuMeshlet> sceneMeshlets;
[[vk::binding(2, 0)]] StructuredBuffer<uint> meshletVertexIndices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> meshletTriangleIndices;
[[vk::binding(4, 0)]] RaytracingAccelerationStructure sceneTlas;
[[vk::binding(5, 0)]] StructuredBuffer<GpuInstance> sceneInstances;
[[vk::binding(6, 0)]] StructuredBuffer<uint> geometryIndices;
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

[shader("amplification")]
[numthreads(1, 1, 1)]
void TaskMain(
    uint3 groupIndex : SV_GroupID)
{
    TaskPayload taskPayload;
    uint meshletIndex = groupIndex.x;
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

    taskPayload.meshletIndex = meshletIndex;
    DispatchMesh(visible ? 1 : 0, 1, 1, taskPayload);
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
    GpuMeshlet meshlet = sceneMeshlets[taskPayload.meshletIndex];
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
        vertices[outputVertex].meshletIndex = taskPayload.meshletIndex;
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
    float3 lightVector = frame.lightPosition.xyz - input.worldPosition;
    float lightDistance = max(length(lightVector), 0.0001);
    float3 lightDirection = lightVector / lightDistance;
    float3 baseColor = input.baseColorAndMetallic.xyz;
    float visibility = 1.0;
    if (frame.rayTracedShadows != 0)
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
    float3 color = evaluateMetallicRoughness(baseColor, input.baseColorAndMetallic.w, input.roughness,
                                             normal, viewDirection, lightDirection,
                                             frame.lightColorAndIntensity.xyz * frame.lightColorAndIntensity.w * visibility);
    color += baseColor * 0.025;
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

        float3 reflectedRadiance;
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
            float3 hitLightVector = frame.lightPosition.xyz - hitPosition;
            float hitLightDistance = max(length(hitLightVector), 0.0001);
            float3 hitLightDirection = hitLightVector / hitLightDistance;
            float hitVisibility = 1.0;
            if (frame.rayTracedShadows != 0)
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
            reflectedRadiance = evaluateMetallicRoughness(hitInstance.baseColorAndMetallic.xyz,
                                                           hitInstance.baseColorAndMetallic.w,
                                                           hitInstance.emissiveAndRoughness.w,
                                                           hitNormal,
                                                           reflectedViewDirection,
                                                           hitLightDirection,
                                                           frame.lightColorAndIntensity.xyz *
                                                               frame.lightColorAndIntensity.w * hitVisibility);
            reflectedRadiance += hitInstance.baseColorAndMetallic.xyz * 0.025 +
                                 hitInstance.emissiveAndRoughness.xyz;
        }
        else
        {
            float skyFactor = saturate(reflectionRay.Direction.y * 0.5 + 0.5);
            reflectedRadiance = lerp(float3(0.012, 0.018, 0.035), float3(0.16, 0.38, 0.78), skyFactor);
        }
        float3 reflectionF0 = lerp(float3(0.04), baseColor, input.baseColorAndMetallic.w);
        float3 reflectionWeight = fresnelSchlick(saturate(dot(normal, viewDirection)), reflectionF0) *
                                  (1.0 - saturate(input.roughness) * 0.75);
        color += reflectedRadiance * reflectionWeight;
    }
    color *= exp2(frame.exposure);
    color = max(color, 0.0);
    color = color / (1.0 + color);
    output.color = float4(color, 1.0);
    return output;
}
