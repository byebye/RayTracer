#include <cfloat>
#include <cstdio>
#include <cstdint>

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "common/Structures.h"
#include "cudaTriangles/CuUtils.h"
#include "cudaTriangles/KdTreeStructures.h"

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


__device__ RGB processPixelOnBackground(BaseConfig* config)
{
  return config->background;
}

__global__ void computePixel(RGB* bitmap,
                             BaseConfig* config,
                             int treeRoot,
                             int trianglesNum,
                             Triangle* triangles,
                             int leafNodesNum,
                             LeafNode* leafNodes,
                             int splitNodesNum,
                             SplitNode* splitNodes)
{
  int thidX = (blockIdx.x * blockDim.x) + threadIdx.x;
  int thidY = (blockIdx.y * blockDim.y) + threadIdx.y;

  KdTreeData treeData;
  treeData.treeRoot = treeRoot;
  treeData.triangles = triangles;
  treeData.trianglesNum = trianglesNum;
  treeData.leafNodes = leafNodes;
  treeData.leafNodesNum = leafNodesNum;
  treeData.splitNodes = splitNodes;
  treeData.splitNodesNum = splitNodesNum;

  if (thidX < 2 * config->imageY && thidY < 2 * config->imageZ)
  {
    Point point{static_cast<float>(config->imageX),
                static_cast<float>(thidX - config->imageY) / config->antiAliasing,
                static_cast<float>(thidY - config->imageZ) / config->antiAliasing};

    Segment ray{config->observer, point};
    int idx = thidX * config->imageZ * 2 + thidY;
    //bitmap[idx] = ...;
  }
}

}
