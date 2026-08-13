import Microfacet;

float henyeyGreenstein(float cosineTheta, float anisotropy)
{
    float g = clamp(anisotropy, -0.99, 0.99);
    float denominator = 1.0 + g * g - 2.0 * g * cosineTheta;
    return (1.0 - g * g) / max(4.0 * PI * denominator * sqrt(denominator), 1e-8);
}

float3 sampleHenyeyGreenstein(float3 incomingDirection, float anisotropy, float2 randomSample)
{
    float g = clamp(anisotropy, -0.99, 0.99);
    float cosineTheta;
    if (abs(g) < 1e-3)
        cosineTheta = 1.0 - 2.0 * randomSample.x;
    else
    {
        float fraction = (1.0 - g * g) / (1.0 - g + 2.0 * g * randomSample.x);
        cosineTheta = clamp((1.0 + g * g - fraction * fraction) / (2.0 * g), -1.0, 1.0);
    }
    float sineTheta = sqrt(max(1.0 - cosineTheta * cosineTheta, 0.0));
    float phi = 2.0 * PI * randomSample.y;
    float3 forward = normalize(incomingDirection);
    float3 tangent = normalize(cross(abs(forward.z) < 0.999 ? float3(0.0, 0.0, 1.0)
                                                               : float3(0.0, 1.0, 0.0),
                                     forward));
    float3 bitangent = cross(forward, tangent);
    return normalize(tangent * (cos(phi) * sineTheta) + bitangent * (sin(phi) * sineTheta) +
                     forward * cosineTheta);
}
