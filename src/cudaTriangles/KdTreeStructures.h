#ifndef CUDA_TRIANGLES_KDTREESTRUCTURES_H
#define CUDA_TRIANGLES_KDTREESTRUCTURES_H

#include "common/Structures.h"
#include "cudaTriangles/CuUtils.h"
#include <vector_types.h>

// a leaf node in the kd-tree
struct Leaf
{
  float triangleCount;
  int2 firstTriangle;
  int2 parentPointer;
};

// an internal node in the kd-tree
struct SplitNode
{
  float splitValue;
  int2 leftChildPointer;
  int2 rightChildPointer;
  // sign bits store child type (leaf/split) and split axis
};

// further data for kd-backtrack
struct ExtendedSplitNode
{
  float3 boundingBoxMin;
  float3 boundingBoxMax;
  float2 parentPointer;
};

// Per ray traversal state

struct TraversalState
{
  float2 nodePointer;
  float tmin;
  float tmax;
  // sign bits store node type (leaf, split)
  // and ray state (traverse, intersect, done)
};

struct IntersectState
{
  float2 triangleIndex;
  float triangleCount;
  float tmax;
};

struct HitState
{
  float2 bestTriangleIndex;
  float thit;
  float globalTmax;
};

// struct FindResult
//{
//  bool exists = false;
//  Triangle triangle;
//  Point point;
//};
//
// struct KdTree
//{
//  BoundingBox bb;
//
//  FindResult find(Segment seg, Triangle const& excludedTriangle)
//  {
//    FindResult res{};
//
//    if (!intersectionBoundingBox(seg, bb))
//      return res;
//
//    if (triangles.empty())
//      return findRecursive(seg, excludedTriangle);
//    else
//      return findInTriangles(seg, excludedTriangle);
//  }
//};


#endif // CUDA_TRIANGLES_KDTREESTRUCTURES_H
