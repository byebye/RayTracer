#include "common/RayTracerConfig.h"

#include <exception>
#include <fstream>
#include <iostream>

#include "common/StructuresOperators.h"

namespace
{
std::istream& operator>>(std::istream& in, RGB& rgb)
{
  short r, g, b;
  in >> r >> g >> b;
  rgb.r = r;
  rgb.g = g;
  rgb.b = b;
  return in;
}

Sphere parseSphere(std::ifstream& file)
{
  Sphere sphere{};
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "endSphere")
      break;
    else if (token == "center")
      file >> sphere.center.x >> sphere.center.y >> sphere.center.z;
    else if (token == "radius")
      file >> sphere.radius;
    else if (token == "color")
      file >> sphere.color;
    else if (token == "reflection")
      file >> sphere.reflectionCoefficient;
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return sphere;
}

Plane parsePlane(std::ifstream& file)
{
  Plane plane{};
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "endPlane")
      break;
    else if (token == "point")
      file >> plane.P.x >> plane.P.y >> plane.P.z;
    else if (token == "normalVector")
      file >> plane.normal.x >> plane.normal.y >> plane.normal.z;
    else if (token == "coef")
      file >> plane.d;
    else if (token == "color")
      file >> plane.color;
    else if (token == "reflection")
      file >> plane.reflectionCoefficient;
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return plane;
}

} // anonymous namespace

RayTracerConfig RayTracerConfig::fromFile(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open())
    throw std::invalid_argument("Unable to open file: " + path);

  RayTracerConfig config;
  std::string token;
  while (file >> token)
  {
    if (token[0] == '#') // comment, skip entire line
      std::getline(file, token);
    else if (token == "aa")
      file >> config.antiAliasing;
    else if (token == "ambient")
      file >> config.ambientCoefficient;
    else if (token == "maxRecursion")
      file >> config.maxRecursionLevel;
    else if (token == "depth")
      file >> config.imageX;
    else if (token == "height")
      file >> config.imageY;
    else if (token == "width")
      file >> config.imageZ;
    else if (token == "observer")
      file >> config.observer.x >> config.observer.y >> config.observer.z;
    else if (token == "light")
      file >> config.light.x >> config.light.y >> config.light.z;
    else if (token == "sphere")
      config.spheres.push_back(parseSphere(file));
    else if (token == "plane")
      config.planes.push_back(parsePlane(file));
    else
      throw std::invalid_argument("Unknown token '" + token + "'");
    if (!file.good())
      throw std::invalid_argument("Invalid config file format.");
  }
  return config;
}

std::ostream& operator<<(std::ostream& out, RayTracerConfig const& config)
{
  out << "antiAliasing: " << config.antiAliasing
      << "\nmaxRecursionLevel: " << config.maxRecursionLevel
      << "\nambientCoefficient: " << config.ambientCoefficient << "\nimageX: " << config.imageX
      << "\nimageY: " << config.imageY << "\nimageZ: " << config.imageZ
      << "\nobserver: " << config.observer << "\nlight: " << config.light;
  out << "\nspheres:\n";
  for (Sphere const& sphere : config.spheres)
    out << sphere << '\n';
  out << "planes:\n";
  for (Plane const& plane : config.planes)
    out << plane << '\n';
  return out;
}

RayTracerConfig RayTracerConfig::defaultConfig()
{
  RayTracerConfig config;

  config.antiAliasing = 2;
  config.maxRecursionLevel = 1;
  config.ambientCoefficient = 0.1;
  config.imageX = 512;
  config.imageY = 384 * config.antiAliasing;
  config.imageZ = 512 * config.antiAliasing;
  config.observer = {0, 0, 0};
  config.light = {1000, 2000, 2500};

  // red
  config.spheres.push_back({{2500, -200, -600}, 600, {200, 0, 0}, 0.3});
  // green
  config.spheres.push_back({{2000, 0, 800}, 400, {0, 200, 0}, 0.1});

  // Plane has one face!
  // front
  config.planes.push_back({{6000, 0, 0}, {-1, 0, 0}, 6000, {178, 170, 30}, 0.05});
  // back
  config.planes.push_back({{-2000, 0, 0}, {1, 0, 0}, 2000, {245, 222, 179}});
  // top
  config.planes.push_back({{0, 3000, 0}, {0, -1, 0}, 3000, {255, 105, 180}, 0.05});
  // bottom
  config.planes.push_back({{0, -800, 0}, {0, 1, 0}, 800, {100, 100, 200}, 0.05});
  // left
  config.planes.push_back({{0, 0, -2500}, {0, 0, 1}, 2500, {32, 178, 170}, 0.05});
  // right
  config.planes.push_back({{0, 0, 3500}, {0, 0, -1}, 3500, {32, 178, 170}, 0.05});

  return config;
}