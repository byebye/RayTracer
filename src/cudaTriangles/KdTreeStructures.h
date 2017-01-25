#ifndef CUDA_TRIANGLES_KDTREESTRUCTURES_H
#define CUDA_TRIANGLES_KDTREESTRUCTURES_H

#include "common/Structures.h"
#include "common/Utils.h"
#include "cudaTriangles/CuUtils.h"

#include <vector_types.h>

struct KdTreeData
{
  int treeRoot;

  int trianglesNum;
  Triangle* triangles;

  int leafNodesNum;
  LeafNode* leafNodes;

  int splitNodesNum;
  SplitNode* splitNodes;
};

// a leaf node in the kd-tree
struct LeafNode
{
  int triangleCount;
  int firstTriangle;
  int parent;
};

// an internal node in the kd-tree
struct SplitNode
{
  float splitValue;
  // > 0 - SplitNode, < 0 - LeafNode, points to position + 1
  int leftChild;
  int rightChild;
  BoundingBox bb;
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

#endif // CUDA_TRIANGLES_KDTREESTRUCTURES_H
