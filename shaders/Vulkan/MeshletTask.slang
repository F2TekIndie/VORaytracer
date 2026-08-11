struct TaskPayload
{
    uint meshletIndex;
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void TaskMain(
    uint threadIndex : SV_GroupThreadID,
    out payload TaskPayload taskPayload)
{
    taskPayload.meshletIndex = 0;
    DispatchMesh(1, 1, 1, taskPayload);
}
