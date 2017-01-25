#ifndef CUDA_TRIANGLES_KDTREEBUILDER_H
#define CUDA_TRIANGLES_KDTREEBUILDER_H

#include "cudaTriangles/KdTreeStructures.h"

#include <vector>

struct KdTreeBuilder
{
  KdTreeBuilder(int const leafTrianglesLimit)
    : trianglesInLeafBound(leafTrianglesLimit)
  {
  }

  std::vector<SplitNode> splitNodes;
  std::vector<LeafNode> leafNodes;
  std::vector<Triangle> outTriangles;

  void clear()
  {
    splitNodes.clear();
    leafNodes.clear();
    outTriangles.clear();
  }

  int build(std::vector<Triangle> const& triangles, int parent, int depth = 0);

private:
  int const trianglesInLeafBound;

  int addLeaf(std::vector<Triangle> const& triangles, int parent);
};


#endif // CUDA_TRIANGLES_KDTREEBUILDER_H
