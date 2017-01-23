#include <cfloat>
#include <cstdio>
#include <cstdint>

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "common/Structures.h"
#include "cudaTriangles/CuUtils.h"

struct IntersectionResult
{
  Point intersectionPoint;
  bool intersects;
};

struct pairip
{
  int first;
  Point second;
};

extern "C" {

__device__ RGB processPixel(Segment const& ray,
                            int recurstionLevel,
                            Sphere* spheres,
                            int spheresNum,
                            Plane* planes,
                            int planesNum,
                            int maxRecursionLevel,
                            float ambientCoefficient,
                            Point const& light,
                            RGB const& background);

__device__ bool pointInShadow(Point const& point, Point const& light, Sphere const& sphere)
{
  Segment seg = {point, light};
  IntersectionResult const& res = intersection(seg, sphere);
  return res.intersects && distance(point, res.intersectionPoint) < distance(point, light);
}

__device__ bool pointInShadowP(Point const& point, Point const& light, Plane const& plane)
{
  Segment seg = {point, light};
  IntersectionResult const& res = intersectionP(seg, plane);
  return res.intersects && distance(point, res.intersectionPoint) < distance(point, light);
}

__device__ RGB processPixelOnBackground(RGB const& background)
{
  return background;
}

__device__ pairip findClosestSphereIntersection(Segment const& seg, Sphere* spheres, int spheresNum)
{
  Point closestPoint{};
  int sphereIndex = -1;
  float closestDistance = FLT_MAX;

  for (size_t i = 0; i < spheresNum; i++)
  {
    IntersectionResult const& res = intersection(seg, spheres[i]);

    if (!res.intersects)
      continue;

    float dist = distance(seg.a, res.intersectionPoint);
    if (dist < closestDistance)
    {
      closestDistance = dist;
      closestPoint = res.intersectionPoint;
      sphereIndex = i;
    }
  }
  return {sphereIndex, closestPoint};
}

__device__ pairip findClosestPlaneIntersection(Segment const& seg, Plane* planes, int planesNum)
{
  Point closestPoint{};
  int planeIndex = -1;
  float closestDistance = FLT_MAX;

  for (size_t i = 0; i < planesNum; i++)
  {
    IntersectionResult const& res = intersectionP(seg, planes[i]);

    if (!res.intersects)
      continue;

    float dist = distance(seg.a, res.intersectionPoint);
    if (dist < closestDistance)
    {
      closestDistance = dist;
      closestPoint = res.intersectionPoint;
      planeIndex = i;
    }
  }

  return {planeIndex, closestPoint};
}

__device__ RGB calculateColorFromReflection(RGB currentColor, RGB reflectedColor,
                                            float reflectionCoefficient)
{
  return currentColor * (1.0f - reflectionCoefficient) + reflectedColor * reflectionCoefficient;
}

__global__ void computePixel(Sphere* spheres,
                             int spheresNum,
                             Plane* planes,
                             int planesNum,
                             RGB* bitmap,
                             int imageX, int imageY, int imageZ,
                             int antiAliasing,
                             int maxRecursionLevel,
                             float ambientCoefficient,
                             float observerX, float observerY, float observerZ,
                             float lX, float lY, float lZ,
                             uint8_t R, uint8_t G, uint8_t B)
{
  int thidX = (blockIdx.x * blockDim.x) + threadIdx.x;
  int thidY = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (thidX < 2 * imageY && thidY < 2 * imageZ)
  {
    Point const observer = {observerX, observerY, observerZ};
    Point const light = {lX, lY, lZ};
    RGB background = {R, G, B};
    Point point{static_cast<float>(imageX),
                static_cast<float>(thidX - imageY) / antiAliasing,
                static_cast<float>(thidY - imageZ) / antiAliasing};

    Segment ray{observer, point};
    int idx = thidX * imageZ * 2 + thidY;
    bitmap[idx] = processPixel(ray, 0, spheres, spheresNum, planes, planesNum, maxRecursionLevel,
                               ambientCoefficient, light, background);
  }
}
}
