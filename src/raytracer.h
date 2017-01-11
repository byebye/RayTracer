#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "structures.h"


#include <array>
#include <vector>


class RayTracer
{
  // for antiAliasing = 4, 16 pixels are generated for each one from final scene
  static int const antiAliasing = 2;
  static int const maxRecursionLevel = 1;

  // We assume threadNumber < imageY
  static int const threadNumber = 2;

  Point const observer = {0, 0, 0};
  Point const light = {1000, 2000, 2500};

  // image is a rectangle with verticles (256, -+imageY/antiAliasing, -+imageZ/antiAliasing)
  static int const imageX = 512;
  Point const imageCenter = {imageX, 0, 0};
  static int const imageY = 384 * antiAliasing;
  static int const imageZ = 512 * antiAliasing;

  std::array<std::array<RGB, imageZ * 2>, imageY * 2> bitmap;

  double const diffuseCoefficient = 0.9;
  double const ambientCoefficient = 0.1;

  RGB processPixel(Segment const& ray, int recursionLevel);
  RGB processPixelOnBackground();
  RGB processPixelOnSphere(Point const& rayBeg, Point const& pointOnSphere, size_t sphereIndex,
                           int recursionLevel);
  RGB processPixelOnPlane(Point const& rayBeg, Point const& pointOnPlane, size_t planeIndex,
                          int recursionLevel);
  std::pair<int, Point> findClosestSphereIntersection(Segment const& seg);
  std::pair<int, Point> findClosestPlaneIntersection(Segment const& seg);
  void processPixelsThreads(int threadId);

public:
  void processPixels();
  void printBitmap();

  std::vector<Sphere> spheres;
  std::vector<Plane> planes;
};

#endif // RAYTRACER_H