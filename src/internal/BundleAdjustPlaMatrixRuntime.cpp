#include "BundleAdjustPlaMatrixRuntime.h"

#include "BundleAdjustPlaMatrix.h"

#include <plamatrix/opencl/runtime.h>

#ifdef PLAMATRIX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <stdexcept>
#include <string>

namespace plabundle::internal
{

    plamatrix::SchurComplementLinearBackend plaMatrixLinearBackend(BABackend backend)
    {
        switch (backend)
        {
        case BABackend::PlaMatrixCpu:
            return plamatrix::hasSparseDirectSchurSolver() ? plamatrix::SchurComplementLinearBackend::SparseCpu
                                                           : plamatrix::SchurComplementLinearBackend::DenseCpu;
        case BABackend::PlaMatrixCuda:
            return plamatrix::SchurComplementLinearBackend::Cuda;
        case BABackend::PlaMatrixOpenCl:
            return plamatrix::SchurComplementLinearBackend::OpenCl;
        case BABackend::Auto:
            break;
        }
        throw std::invalid_argument("PlaMatrix solve requires a concrete backend");
    }

    const char* plaMatrixLinearBackendName(plamatrix::SchurComplementLinearBackend backend)
    {
        switch (backend)
        {
        case plamatrix::SchurComplementLinearBackend::Cpu:
            return "block_jacobi_pcg_cpu";
        case plamatrix::SchurComplementLinearBackend::DenseCpu:
            return "dense_cholesky_cpu";
        case plamatrix::SchurComplementLinearBackend::SparseCpu:
            return "sparse_cholesky_native_cpu";
        case plamatrix::SchurComplementLinearBackend::Cuda:
            return "block_jacobi_pcg_cuda";
        case plamatrix::SchurComplementLinearBackend::OpenCl:
            return "block_jacobi_pcg_opencl";
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
            if (!plamatrix::opencl::hasUsableOpenClDevice())
            {
                if (message)
                {
                    *message = "PlaMatrix has no usable OpenCL GPU with an online compiler";
                }
                return false;
            }
            const int selected_index = plamatrix::opencl::selectedOpenClDeviceIndex();
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
            if (!plamatrix::opencl::OpenClRuntime::instance().supportsFp64())
            {
                if (message)
                {
                    *message = "PlaMatrix OpenCL device " + plamatrix::opencl::selectedOpenClDeviceName() +
                               " does not support the FP64 arithmetic required by PlaBundle";
                }
                return false;
            }
#endif
            return true;
        }
        if (message)
        {
            *message = "backend is not a concrete PlaMatrix backend";
        }
        return false;
    }

} // namespace plabundle::internal
