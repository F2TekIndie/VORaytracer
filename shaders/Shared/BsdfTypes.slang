enum BsdfLobe : uint
{
    DiffuseLobe,
    SpecularReflectionLobe,
    SpecularTransmissionLobe,
    ClearcoatLobe,
    SheenLobe,
    SubsurfaceLobe,
};

struct BsdfSample
{
    float3 direction;
    float3 weight;
    float pdf;
    uint lobe;
    bool transmission;
    bool delta;
};

struct BsdfEvaluation
{
    float3 value;
    float pdf;
};

BsdfSample invalidBsdfSample()
{
    BsdfSample sample;
    sample.direction = float3(0.0);
    sample.weight = float3(0.0);
    sample.pdf = 0.0;
    sample.lobe = uint(BsdfLobe::DiffuseLobe);
    sample.transmission = false;
    sample.delta = false;
    return sample;
}

float misPowerHeuristic(float firstPdf, float secondPdf)
{
    float firstSquared = firstPdf * firstPdf;
    float secondSquared = secondPdf * secondPdf;
    return firstSquared / max(firstSquared + secondSquared, 1e-20);
}
