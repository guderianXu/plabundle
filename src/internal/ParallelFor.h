#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace plabundle::internal
{

    template <typename Function>
    void parallelForIndices(std::size_t count, std::size_t requestedWorkerCount, Function&& function)
    {
        if (count == 0)
        {
            return;
        }

        const std::size_t worker_count = std::min(count, std::max<std::size_t>(1, requestedWorkerCount));
        if (worker_count == 1)
        {
            for (std::size_t index = 0; index < count; ++index)
            {
                function(index);
            }
            return;
        }

        std::atomic<std::size_t> next_index{0};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::exception_ptr error;
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index)
        {
            workers.emplace_back(
                [&]()
                {
                    while (!failed.load(std::memory_order_acquire))
                    {
                        const std::size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
                        if (index >= count)
                        {
                            return;
                        }
                        try
                        {
                            function(index);
                        }
                        catch (...)
                        {
                            {
                                std::lock_guard lock(error_mutex);
                                if (!error)
                                {
                                    error = std::current_exception();
                                }
                            }
                            failed.store(true, std::memory_order_release);
                            return;
                        }
                    }
                });
        }
        workers.clear();
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

} // namespace plabundle::internal
