#ifndef CUDA_TRIANGLES_KDTREEBUILDER_H
#define CUDA_TRIANGLES_KDTREEBUILDER_H

#include "cudaTriangles/KdTreeStructures.h"

#include <vector>

struct KdTreeBuilder
{
  KdTreeBuilder(size_t const leafTrianglesLimit = 16)
    : trianglesInLeafBound(leafTrianglesLimit)
  {
  }

  std::vector<SplitNode> splitNodes;
  std::vector<LeafNode> leafNodes;
  std::vector<Triangle> treeTriangles;

  void clear()
  {
    splitNodes.clear();
    leafNodes.clear();
    treeTriangles.clear();
  }

  int build(std::vector<Triangle> const& triangles)
  {
    return build(triangles, -1, 0);
  }

private:
  size_t const trianglesInLeafBound;

  int build(std::vector<Triangle> const& triangles, int parent, int depth);

  int addLeaf(std::vector<Triangle> const& triangles, int parent);
};


#endif // CUDA_TRIANGLES_KDTREEBUILDER_H
