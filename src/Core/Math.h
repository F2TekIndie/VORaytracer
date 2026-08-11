#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace vor
{
constexpr float kPi = 3.14159265358979323846f;

struct Vec2
{
    float x{};
    float y{};
};

struct Vec3
{
    float x{};
    float y{};
    float z{};
};

struct alignas(16) Vec4
{
    float x{};
    float y{};
    float z{};
    float w{};
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float b) { return {a.x * b, a.y * b, a.z * b}; }
inline Vec3 operator/(Vec3 a, float b) { return {a.x / b, a.y / b, a.z / b}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(Vec3 v)
{
    const float size = length(v);
    return size > 0.0f ? v / size : Vec3{};
}

struct alignas(16) Mat4
{
    std::array<float, 16> m{};

    static Mat4 identity()
    {
        Mat4 result{};
        result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;
        return result;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 result{};
    for (std::uint32_t column = 0; column < 4; ++column)
    {
        for (std::uint32_t row = 0; row < 4; ++row)
        {
            for (std::uint32_t k = 0; k < 4; ++k)
                result.m[column * 4 + row] += a.m[k * 4 + row] * b.m[column * 4 + k];
        }
    }
    return result;
}

inline Mat4 perspective(float verticalFovRadians, float aspect, float nearPlane, float farPlane)
{
    const float f = 1.0f / std::tan(verticalFovRadians * 0.5f);
    Mat4 result{};
    result.m[0] = f / aspect;
    result.m[5] = -f; // Vulkan clip space has an inverted Y axis.
    result.m[10] = farPlane / (nearPlane - farPlane);
    result.m[11] = -1.0f;
    result.m[14] = (nearPlane * farPlane) / (nearPlane - farPlane);
    return result;
}

inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up)
{
    const Vec3 forward = normalize(target - eye);
    const Vec3 right = normalize(cross(forward, up));
    const Vec3 correctedUp = cross(right, forward);

    Mat4 result = Mat4::identity();
    result.m[0] = right.x;
    result.m[1] = correctedUp.x;
    result.m[2] = -forward.x;
    result.m[4] = right.y;
    result.m[5] = correctedUp.y;
    result.m[6] = -forward.y;
    result.m[8] = right.z;
    result.m[9] = correctedUp.z;
    result.m[10] = -forward.z;
    result.m[12] = -dot(right, eye);
    result.m[13] = -dot(correctedUp, eye);
    result.m[14] = dot(forward, eye);
    return result;
}
} // namespace vor
