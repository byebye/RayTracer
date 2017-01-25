#ifndef CUDA_TRIANGLES_CUUTILS_H
#define CUDA_TRIANGLES_CUUTILS_H

#include <cfloat>
#include <cstdint>
#include <cstdio>

#include <device_launch_parameters.h>
#include <host_defines.h>
#include <math_functions.h>

#include "common/Structures.h"

extern "C" {

__device__ bool isCloseToZero(float x)
{
  return abs(x) < DBL_EPSILON;
}

__device__ RGB& operator*=(RGB& rgb, float times)
{
  rgb.r *= times;
  rgb.g *= times;
  rgb.b *= times;
  return rgb;
}

__device__ RGB operator*(RGB rgb, float times)
{
  return rgb *= times;
}

__device__ RGB& operator+=(RGB& lhs, RGB rhs)
{
  lhs.r += rhs.r;
  lhs.g += rhs.g;
  lhs.b += rhs.b;
  return lhs;
}

__device__ RGB operator+(RGB lhs, RGB rhs)
{
  return lhs += rhs;
}

__device__ Point& operator-=(Point& a, Point const& b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__device__ Point operator-(Point a, Point const& b)
{
  return a -= b;
}

__device__ float distance(Point const& a, Point const& b)
{
  return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2) + pow(b.z - a.z, 2));
}

__device__ float vectorLen(Point const& vec)
{
  return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float dotProduct(Point const& a, Point const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vector crossProduct(Vector const& a, Vector const& b)
{
  Vector vec;
  vec.x = a.y * b.z - b.y * a.z;
  vec.y = a.z * b.x - a.x * b.z;
  vec.z = a.x * b.y - a.y * b.x;
  return vec;
}

__device__ void normalize(Point& vec)
{
  float len = vectorLen(vec);
  vec.x = vec.x / len;
  vec.y = vec.y / len;
  vec.z = vec.z / len;
}

__device__ Vector crossProduct(Vector const& a, Vector const& b)
{
  Vector vec;
  vec.x = a.y * b.z - b.y * a.z;
  vec.y = a.z * b.x - a.x * b.z;
  vec.z = a.x * b.y - a.y * b.x;
  return vec;
}

__device__ bool intersectsBoundingBox(Segment const& segment, BoundingBox const& box)
{
  Vector dir = normalize(segment.b - segment.a);
  float dirfracX = 1.0f / dir.x;
  float dirfracY = 1.0f / dir.y;
  float dirfracZ = 1.0f / dir.z;

  float t1 = (box.vMin.x - segment.a.x) * dirfracX;
  float t2 = (box.vMax.x - segment.a.x) * dirfracX;
  float t3 = (box.vMin.y - segment.a.y) * dirfracY;
  float t4 = (box.vMax.y - segment.a.y) * dirfracY;
  float t5 = (box.vMin.z - segment.a.z) * dirfracZ;
  float t6 = (box.vMax.z - segment.a.z) * dirfracZ;

  float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
  float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

  return 0 <= tmax && tmin <= tmax;
}

__device__ RGB calculateColorFromReflection(RGB currentColor, RGB reflectedColor,
                                            float reflectionCoefficient)
{
  return currentColor * (1.0f - reflectionCoefficient) + reflectedColor * reflectionCoefficient;
}
};
#endif // CUDA_TRIANGLES_CUUTILS_H
