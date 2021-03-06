#include "cuda/RayTracerCuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define CU_CHECK(ans)                                                                              \
  {                                                                                                \
    cuAssert((ans), __FILE__, __LINE__);                                                           \
  }
inline void cuAssert(CUresult code, const char* file, int line, bool abort = true)
{
  if (code != CUDA_SUCCESS)
  {
    const char* error_string;
    cuGetErrorString(code, &error_string);
    std::cerr << file << ":" << line << " - CUDA error (" << code << "): " << error_string
              << std::endl;
    if (abort)
      exit(code);
  }
}

void RayTracerCuda::processPixelsCuda()
{
  cuInit(0);

  CUdevice cuDevice;
  CU_CHECK(cuDeviceGet(&cuDevice, 0));

  CUcontext cuContext;
  CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  CUmodule cuModule = (CUmodule) 0;
  CU_CHECK(cuModuleLoad(&cuModule, "raytracing.ptx"));

  CUfunction computePixel;
  CU_CHECK(cuModuleGetFunction(&computePixel, cuModule, "computePixel"));

  RGB* bitmapTab = bitmap.data();
  CU_CHECK(cuMemHostRegister(bitmapTab, sizeof(RGB) * bitmap.size(), 0));

  CUdeviceptr bitmapDev;
  CU_CHECK(cuMemAlloc(&bitmapDev, sizeof(RGB) * bitmap.size()));

  cudaError code = cudaDeviceSetLimit (cudaLimitStackSize, 2048*2);
  if(code != cudaSuccess)
  {
    std::cerr << "Setting stack limit failed " << code << " " <<std::endl;
    exit(code);
  }

  int planesNum = config.planes.size();
  Plane* planesTab = const_cast<Plane*>(config.planes.data());
  CUdeviceptr planesDev;
  if (planesNum != 0)
  {
    CU_CHECK(cuMemHostRegister(planesTab, sizeof(Plane) * planesNum, 0));
    CU_CHECK(cuMemAlloc(&planesDev, sizeof(Plane) * (planesNum)));
    CU_CHECK(cuMemcpyHtoD(planesDev, planesTab, sizeof(Plane) * (planesNum)));
  }

  int spheresNum = config.spheres.size();
  Sphere* spheresTab = const_cast<Sphere*>(config.spheres.data());
  CUdeviceptr spheresDev;
  if (spheresNum != 0)
  {
    CU_CHECK(cuMemHostRegister(spheresTab, sizeof(Sphere) * spheresNum, 0));
    CU_CHECK(cuMemAlloc(&spheresDev, sizeof(Sphere) * (spheresNum)));
    CU_CHECK(cuMemcpyHtoD(spheresDev, spheresTab, sizeof(Sphere) * (spheresNum)));
  }

  int iX = config.imageX;
  int iY = bitmap.rows / 2;
  int iZ = bitmap.cols / 2;
  int aA = config.antiAliasing;
  int mRL = config.maxRecursionLevel;
  float aC = config.ambientCoefficient;
  float oX = config.observer.x;
  float oY = config.observer.y;
  float oZ = config.observer.z;
  float lX = config.light.x;
  float lY = config.light.y;
  float lZ = config.light.z;
  uint8_t R = 0;
  uint8_t G = 0;
  uint8_t B = 0;

  void* args[] = {&spheresDev, &spheresNum, &planesDev, &planesNum, &bitmapDev, &iX, &iY,
                  &iZ,         &aA,         &mRL,       &aC,        &oX,        &oY, &oZ,
                  &lX,         &lY,         &lZ,        &R,         &G,         &B};
  int threadsNum = 16;
  int blocks_per_grid_x = (bitmap.rows + threadsNum - 1) / threadsNum;
  int blocks_per_grid_y = (bitmap.cols + threadsNum - 1) / threadsNum;
  int threads_per_block_x = threadsNum;
  int threads_per_block_y = threadsNum;

  CU_CHECK(cuLaunchKernel(computePixel, blocks_per_grid_x, blocks_per_grid_y, 1,
                          threads_per_block_x, threads_per_block_y, 1, 0, 0, args, 0));

  CU_CHECK(cuMemcpyDtoH(bitmapTab, bitmapDev, sizeof(RGB) * bitmap.size()));

  CU_CHECK(cuMemHostUnregister(bitmapTab));
  if (spheresNum != 0)
  {
    CU_CHECK(cuMemHostUnregister(spheresTab));
  }
  if (planesNum != 0)
  {
    CU_CHECK(cuMemHostUnregister(planesTab));
  }

  cuCtxDestroy(cuContext);
}
