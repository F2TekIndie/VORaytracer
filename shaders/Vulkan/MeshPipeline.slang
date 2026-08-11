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
    float4 boundingSphere;
    float4 normalCone;
    float4 baseColorAndMetallic;
    float4 materialParameters;
};

struct FramePushConstants
{
    float4x4 viewProjection;
    float4x4 model;
    float4 cameraPosition;
    float4 lightPosition;
    float4 lightColorAndIntensity;
    uint rayTracedShadows;
    float exposure;
    uint meshletCount;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex> sceneVertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuMeshlet> sceneMeshlets;
[[vk::binding(2, 0)]] StructuredBuffer<uint> meshletVertexIndices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> meshletTriangleIndices;
[[vk::binding(4, 0)]] RaytracingAccelerationStructure sceneTlas;
[[vk::push_constant]] ConstantBuffer<FramePushConstants> frame;

struct MeshVertex
{
    float4 position : SV_Position;
    float3 worldNormal : NORMAL0;
    float3 worldPosition : POSITION0;
    float2 uv : TEXCOORD0;
    nointerpolation float4 baseColorAndMetallic : TEXCOORD1;
    nointerpolation float roughness : TEXCOORD2;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
};

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
        if (meshlet.boundingSphere.w > 0.0)
        {
            float4 center = float4(meshlet.boundingSphere.xyz, 1.0);
            float4 clipCenter = mul(frame.viewProjection, center);
            float radius = meshlet.boundingSphere.w;
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
            if (visible && coneLength > 0.5 && meshlet.normalCone.w <= 1.0)
            {
                float3 cameraToMeshlet = normalize(meshlet.boundingSphere.xyz - frame.cameraPosition.xyz);
                visible = dot(normalize(meshlet.normalCone.xyz), cameraToMeshlet) < meshlet.normalCone.w;
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
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);
    for (uint outputVertex = threadIndex; outputVertex < meshlet.vertexCount; outputVertex += 32)
    {
        uint vertexIndex = meshletVertexIndices[meshlet.vertexOffset + outputVertex];
        GpuVertex source = sceneVertices[vertexIndex];
        float4 worldPosition = mul(frame.model, float4(source.position.xyz, 1.0));
        vertices[outputVertex].position = mul(frame.viewProjection, worldPosition);
        vertices[outputVertex].worldPosition = worldPosition.xyz;
        vertices[outputVertex].worldNormal = normalize(mul((float3x3)frame.model, source.normal.xyz));
        vertices[outputVertex].uv = source.uv.xy;
        vertices[outputVertex].baseColorAndMetallic = meshlet.baseColorAndMetallic;
        vertices[outputVertex].roughness = meshlet.materialParameters.x;
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
    color *= exp2(frame.exposure);
    color = max(color, 0.0);
    color = color / (1.0 + color);
    output.color = float4(color, 1.0);
    return output;
}
