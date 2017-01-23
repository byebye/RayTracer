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
};
#endif // CUDA_TRIANGLES_CUUTILS_H
