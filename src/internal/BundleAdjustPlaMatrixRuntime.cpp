#include "BundleAdjustPlaMatrixRuntime.h"

#include "BundleAdjustPlaMatrix.h"

#include <plamatrix/internal/opencl/runtime.h>
#include <plamatrix/internal/vulkan/capabilities.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <stdexcept>
#include <string>

namespace plabundle::internal
{

    plamatrix::internal::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend)
    {
        switch (backend)
        {
        case BABackend::PlaMatrixCpu:
            return plamatrix::internal::hasSparseDirectSchurSolver()
                       ? plamatrix::internal::SchurComplementLinearBackend::SparseCpu
                       : plamatrix::internal::SchurComplementLinearBackend::DenseCpu;
        case BABackend::PlaMatrixCuda:
            return plamatrix::internal::SchurComplementLinearBackend::Cuda;
        case BABackend::PlaMatrixVulkan:
            return plamatrix::internal::SchurComplementLinearBackend::Vulkan;
        case BABackend::PlaMatrixOpenCl:
            return plamatrix::internal::SchurComplementLinearBackend::OpenCl;
        case BABackend::Auto:
            break;
        }
        throw std::invalid_argument("PlaMatrix solve requires a concrete backend");
    }

    const char* plaMatrixLinearBackendName(plamatrix::internal::SchurComplementLinearBackend backend)
    {
        switch (backend)
        {
        case plamatrix::internal::SchurComplementLinearBackend::Cpu:
            return "block_jacobi_pcg_cpu";
        case plamatrix::internal::SchurComplementLinearBackend::DenseCpu:
            return "dense_cholesky_cpu";
        case plamatrix::internal::SchurComplementLinearBackend::SparseCpu:
            return "sparse_cholesky_native_cpu";
        case plamatrix::internal::SchurComplementLinearBackend::Cuda:
            return "block_jacobi_pcg_cuda";
        case plamatrix::internal::SchurComplementLinearBackend::OpenCl:
            return "block_jacobi_pcg_opencl";
        case plamatrix::internal::SchurComplementLinearBackend::Vulkan:
            return "block_jacobi_pcg_vulkan";
        }
        return "unknown";
    }

    bool isPlaMatrixBackendAvailable(BABackend backend, int deviceIndex, std::string* message)
    {
        if (backend == BABackend::PlaMatrixCpu)
        {
            return true;
        }
        if (deviceIndex < 0)
        {
            if (message)
            {
                *message = "PlaMatrix device index must be non-negative";
            }
            return false;
        }
        if (backend == BABackend::PlaMatrixCuda)
        {
#ifdef PLAMATRIX_WITH_CUDA
            int device_count = 0;
            const cudaError_t error = cudaGetDeviceCount(&device_count);
            if (error == cudaSuccess && deviceIndex < device_count)
            {
                return true;
            }
            if (message)
            {
                *message = error == cudaSuccess
                               ? "PlaMatrix CUDA device index is out of range"
                               : std::string("PlaMatrix CUDA runtime is unavailable: ") + cudaGetErrorString(error);
            }
#else
            if (message)
            {
                *message = "PlaMatrix was built without CUDA support";
            }
#endif
            return false;
        }
        if (backend == BABackend::PlaMatrixOpenCl)
        {
            if (!plamatrix::internal::opencl::hasUsableOpenClDevice())
            {
                if (message)
                {
                    *message = "PlaMatrix has no usable OpenCL GPU with an online compiler";
                }
                return false;
            }
            const int selected_index = plamatrix::internal::opencl::selectedOpenClDeviceIndex();
            if (deviceIndex != selected_index)
            {
                if (message)
                {
                    *message = "PlaMatrix OpenCL selected device " + std::to_string(selected_index) +
                               ", which does not match requested device " + std::to_string(deviceIndex);
                }
                return false;
            }
#ifdef PLAMATRIX_WITH_OPENCL
            if (!plamatrix::internal::opencl::OpenClRuntime::instance().supportsFp64())
            {
                if (message)
                {
                    *message = "PlaMatrix OpenCL device " + plamatrix::internal::opencl::selectedOpenClDeviceName() +
                               " does not support the FP64 arithmetic required by PlaBundle";
                }
                return false;
            }
#endif
            return true;
        }
        if (backend == BABackend::PlaMatrixVulkan)
        {
            if (deviceIndex != 0)
            {
                if (message)
                {
                    *message = "PlaMatrix Vulkan automatic runtime currently selects device index 0";
                }
                return false;
            }
            if (!plamatrix::internal::vulkan::hasUsableVulkanDevice())
            {
                if (message)
                {
                    *message = "PlaMatrix has no usable Vulkan compute device";
                }
                return false;
            }
            return true;
        }
        if (message)
        {
            *message = "backend is not a concrete PlaMatrix backend";
        }
        return false;
    }

} // namespace plabundle::internal
