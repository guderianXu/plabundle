#include "BundleAdjustPlaMatrixProblem.h"

#include <plamatrix/internal/ops/statistics.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace plabundle::internal::plamatrix_ba
{
    namespace
    {

        std::size_t parameterIndex(BAIntrinsicParameter parameter)
        {
            return static_cast<std::size_t>(parameter);
        }

        double medianOr(std::vector<double> values, double fallback)
        {
            return plamatrix::internal::finiteMedian(std::move(values)).value_or(fallback);
        }

        struct GroupSamples
        {
            std::array<std::vector<double>, kBAIntrinsicParameterCount> initial;
            std::array<std::vector<double>, kBAIntrinsicParameterCount> reference;
            std::vector<double> imageWidths;
            std::vector<double> imageHeights;
            CameraParameterMask supported;

            GroupSamples()
            {
                supported.fill(true);
            }
        };

        void appendSamples(const CameraState& camera, const CameraState& reference, GroupSamples* samples)
        {
            const auto intrinsics = camera.intrinsics();
            const auto reference_intrinsics = reference.intrinsics();
            const auto parameters = camera.parameterBlock();
            const auto reference_parameters = reference.parameterBlock();
            const auto supported = camera.supportedParameters();
            samples->initial[0].push_back(parameters[0]);
            samples->reference[0].push_back(reference_parameters[0]);
            samples->initial[1].push_back(std::exp(parameters[1]));
            samples->reference[1].push_back(std::exp(reference_parameters[1]));
            samples->initial[2].push_back(intrinsics.principalX - reference_intrinsics.principalX);
            samples->initial[3].push_back(intrinsics.principalY - reference_intrinsics.principalY);
            samples->reference[2].push_back(0.0);
            samples->reference[3].push_back(0.0);
            for (std::size_t index = 4; index < parameters.size(); ++index)
            {
                samples->initial[index].push_back(parameters[index]);
                samples->reference[index].push_back(reference_parameters[index]);
            }
            for (std::size_t index = 0; index < supported.size(); ++index)
            {
                samples->supported[index] = samples->supported[index] && supported[index];
            }
            const auto image_size = reference.imageSize().has_value() ? reference.imageSize() : camera.imageSize();
            samples->imageWidths.push_back(image_size.has_value() ? static_cast<double>(image_size->samples)
                                                                  : 2.0 * reference_intrinsics.principalX);
            samples->imageHeights.push_back(image_size.has_value() ? static_cast<double>(image_size->lines)
                                                                   : 2.0 * reference_intrinsics.principalY);
        }

    } // namespace

    bool hasSharedIntrinsics(const BAOptions& options)
    {
        for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
        {
            if (sharedIntrinsicParameterEnabled(options, static_cast<BAIntrinsicParameter>(index)))
            {
                return true;
            }
        }
        return false;
    }

    std::vector<IntrinsicGroupState> initializeIntrinsicGroups(const std::vector<CameraState>& cameras,
                                                               const BAOptions& options,
                                                               const ActiveProblem& active)
    {
        std::vector<IntrinsicGroupState> groups(static_cast<std::size_t>(active.intrinsicBlockCount));
        if (groups.empty())
        {
            return groups;
        }
        const auto& references =
            options.sharedIntrinsicReferenceCameras.empty() ? cameras : options.sharedIntrinsicReferenceCameras;
        std::vector<GroupSamples> samples(groups.size());
        for (std::size_t camera_index = 0; camera_index < cameras.size(); ++camera_index)
        {
            appendSamples(cameras[camera_index],
                          references[camera_index],
                          &samples[static_cast<std::size_t>(active.calibrationGroupByCamera[camera_index])]);
        }

        const double low_order_scale = std::max(1.0, options.sharedLowOrderDistortionScale);
        for (std::size_t group_index = 0; group_index < groups.size(); ++group_index)
        {
            auto& group = groups[group_index];
            group.lower.fill(-std::numeric_limits<double>::infinity());
            group.upper.fill(std::numeric_limits<double>::infinity());
            for (std::size_t parameter = 0; parameter < kBAIntrinsicParameterCount; ++parameter)
            {
                group.enabled[parameter] =
                    sharedIntrinsicParameterEnabled(options, static_cast<BAIntrinsicParameter>(parameter)) &&
                    samples[group_index].supported[parameter];
            }
            group.focalReference = medianOr(samples[group_index].reference[0], 1.0);
            group.aspectReference = medianOr(samples[group_index].reference[1], 1.0);
            group.parameters[0] = std::max(1e-12, medianOr(samples[group_index].initial[0], group.focalReference));
            group.parameters[1] =
                std::log(std::max(1e-12, medianOr(samples[group_index].initial[1], group.aspectReference)));
            group.prior[0] = group.focalReference;
            group.prior[1] = std::log(group.aspectReference);
            for (std::size_t parameter = 2; parameter < kBAIntrinsicParameterCount; ++parameter)
            {
                group.prior[parameter] = medianOr(samples[group_index].reference[parameter], 0.0);
                group.parameters[parameter] = medianOr(samples[group_index].initial[parameter], group.prior[parameter]);
            }

            group.lower[0] = group.focalReference * options.minSharedFocalScale;
            group.upper[0] = group.focalReference * options.maxSharedFocalScale;
            group.lower[1] = std::log(group.aspectReference * options.minSharedFocalAspectScale);
            group.upper[1] = std::log(group.aspectReference * options.maxSharedFocalAspectScale);
            const double principal_limit = group.focalReference * options.maxSharedPrincipalPointOffsetFraction;
            group.lower[2] = group.lower[3] = -principal_limit;
            group.upper[2] = group.upper[3] = principal_limit;
            const std::array<double, 9> distortion_limits{{options.maxSharedRadialK1Abs * low_order_scale,
                                                           options.maxSharedRadialK2Abs,
                                                           options.maxSharedRadialK3Abs,
                                                           options.maxSharedTangentialP1Abs * low_order_scale,
                                                           options.maxSharedTangentialP2Abs * low_order_scale,
                                                           group.focalReference * options.maxSharedSkewFraction,
                                                           options.maxSharedRadialK4Abs,
                                                           options.maxSharedTangentialP3Abs,
                                                           options.maxSharedTangentialP4Abs}};
            for (std::size_t index = 0; index < distortion_limits.size(); ++index)
            {
                group.lower[index + 4] = -distortion_limits[index];
                group.upper[index + 4] = distortion_limits[index];
            }

            const double principal_sigma =
                std::max(1e-6, group.focalReference * options.sharedPrincipalPointPriorSigmaFraction);
            const std::array<double, kBAIntrinsicParameterCount> sigma{
                {options.sharedFocalPriorSigma * group.focalReference,
                 options.sharedFocalAspectPriorSigma,
                 principal_sigma,
                 principal_sigma,
                 options.sharedRadialK1PriorSigma * low_order_scale,
                 options.sharedRadialK2PriorSigma,
                 options.sharedRadialK3PriorSigma,
                 options.sharedTangentialP1PriorSigma * low_order_scale,
                 options.sharedTangentialP2PriorSigma * low_order_scale,
                 group.focalReference * options.sharedSkewPriorSigmaFraction,
                 options.sharedRadialK4PriorSigma,
                 options.sharedTangentialP3PriorSigma,
                 options.sharedTangentialP4PriorSigma}};
            for (std::size_t parameter = 0; parameter < kBAIntrinsicParameterCount; ++parameter)
            {
                group.parameters[parameter] =
                    std::clamp(group.parameters[parameter], group.lower[parameter], group.upper[parameter]);
                group.inverseSigma[parameter] = group.enabled[parameter] ? 1.0 / std::max(1e-9, sigma[parameter]) : 0.0;
            }

            if (options.useReferenceCalibrationTransitionPrior)
            {
                group.usesReferenceTransitionPrior = true;
                // 参考模型没有独立的像素宽高比参数；该扩展继续沿用原弱先验。
                constexpr std::array<std::size_t, kBAIntrinsicParameterCount - 1> supported_parameters{
                    {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
                for (const std::size_t parameter : supported_parameters)
                {
                    group.inverseSigma[parameter] = 0.0;
                }

                bool mask_changed = false;
                for (const std::size_t parameter : supported_parameters)
                {
                    mask_changed = mask_changed || (group.enabled[parameter] !=
                                                    options.referencePreviousIntrinsicParameterMask[parameter]);
                }
                const double image_width = medianOr(samples[group_index].imageWidths, 0.0);
                const double image_height = medianOr(samples[group_index].imageHeights, 0.0);
                const double focal_y = group.focalReference * group.aspectReference;
                const double normalized_x = 0.5 * image_width / group.focalReference;
                const double normalized_y = 0.5 * image_height / focal_y;
                const double radius = std::sqrt(normalized_x * normalized_x + normalized_y * normalized_y);
                if (mask_changed && std::isfinite(radius) && radius > 0.0)
                {
                    const double radius2 = radius * radius;
                    const double radius3 = radius2 * radius;
                    const double radius5 = radius3 * radius2;
                    const double radius7 = radius5 * radius2;
                    const double radius9 = radius7 * radius2;
                    const std::array<double, kBAIntrinsicParameterCount> base_sigma{
                        {1.0 / radius,
                         0.0,
                         1.0,
                         1.0,
                         1.0 / (radius3 * group.focalReference),
                         1.0 / (radius5 * group.focalReference),
                         1.0 / (radius7 * group.focalReference),
                         1.0 / (radius2 * group.focalReference),
                         1.0 / (radius2 * group.focalReference),
                         1.0 / radius,
                         1.0 / (radius9 * group.focalReference),
                         1.0,
                         1.0}};
                    for (const std::size_t parameter : supported_parameters)
                    {
                        if (!group.enabled[parameter])
                        {
                            continue;
                        }
                        const double sigma_scale =
                            options.referencePreviousIntrinsicParameterMask[parameter] ? 1.0e6 : 0.01;
                        group.transitionWeight[parameter] = 1.0 / (base_sigma[parameter] * sigma_scale);
                    }
                }
            }
        }
        return groups;
    }

    BAIntrinsicParameterMask
    activeIntrinsicParameters(const BAOptions& options, const BAIntrinsicParameterMask& enabled, int iteration)
    {
        static_cast<void>(options);
        static_cast<void>(iteration);
        return enabled;
    }

    int intrinsicStageCount(const BAOptions& options, const BAIntrinsicParameterMask& enabled)
    {
        static_cast<void>(options);
        static_cast<void>(enabled);
        return 1;
    }

    void applyIntrinsicStep(const ActiveProblem& active,
                            const std::vector<double>& primary_step,
                            std::vector<IntrinsicGroupState>* groups)
    {
        for (std::size_t group_index = 0; group_index < groups->size(); ++group_index)
        {
            auto& group = (*groups)[group_index];
            const int block = active.cameraBlockCount + static_cast<int>(group_index);
            for (std::size_t parameter = 0; parameter < kBAIntrinsicParameterCount; ++parameter)
            {
                if (!group.enabled[parameter])
                {
                    continue;
                }
                group.parameters[parameter] = std::clamp(
                    group.parameters[parameter] +
                        primary_step[static_cast<std::size_t>(block * kPrimaryBlockSize + static_cast<int>(parameter))],
                    group.lower[parameter],
                    group.upper[parameter]);
            }
        }
    }

    void publishIntrinsics(const std::vector<CameraState>& input_cameras,
                           const BAOptions& options,
                           const ActiveProblem& active,
                           const std::vector<IntrinsicGroupState>& groups,
                           const BAIntrinsicParameterMask& committed_parameters,
                           BAResult* result)
    {
        if (groups.empty())
        {
            return;
        }
        const auto& references =
            options.sharedIntrinsicReferenceCameras.empty() ? input_cameras : options.sharedIntrinsicReferenceCameras;
        double focal_scale_sum = 0.0;
        double aspect_scale_sum = 0.0;
        std::array<double, 2 + kBAIntrinsicParameterCount - 4> remaining_sums{};
        for (std::size_t camera_index = 0; camera_index < result->refinedCameras.size(); ++camera_index)
        {
            auto& camera = result->refinedCameras[camera_index];
            const auto source = input_cameras[camera_index].intrinsics();
            const auto reference = references[camera_index].intrinsics();
            const auto source_parameters = input_cameras[camera_index].parameterBlock();
            auto published_parameters = source_parameters;
            const auto& group = groups[static_cast<std::size_t>(active.calibrationGroupByCamera[camera_index])];
            const auto publishes = [&](std::size_t parameter) {
                return group.enabled[parameter] &&
                       (!group.usesReferenceTransitionPrior || committed_parameters[parameter]);
            };
            const double focal = publishes(0) ? group.parameters[0] : source.focalX;
            const double source_aspect = source.focalY / source.focalX;
            const double aspect = publishes(1) ? std::exp(group.parameters[1]) : source_aspect;
            const double principal_x = publishes(2) ? reference.principalX + group.parameters[2] : source.principalX;
            const double principal_y = publishes(3) ? reference.principalY + group.parameters[3] : source.principalY;
            published_parameters[0] = focal;
            published_parameters[1] = std::log(aspect);
            published_parameters[2] = principal_x;
            published_parameters[3] = principal_y;
            for (std::size_t index = 4; index < published_parameters.size(); ++index)
            {
                if (publishes(index))
                {
                    published_parameters[index] = group.parameters[index];
                }
            }
            const CameraParameterMask supported = camera.supportedParameters();
            (void)camera.setParameterBlock(published_parameters, supported);
            focal_scale_sum += focal / group.focalReference;
            aspect_scale_sum += aspect / group.aspectReference;
            remaining_sums[0] += principal_x - reference.principalX;
            remaining_sums[1] += principal_y - reference.principalY;
            for (std::size_t index = 0; index < remaining_sums.size() - 2; ++index)
            {
                remaining_sums[index + 2] += published_parameters[index + 4];
            }
            bool changed = false;
            for (std::size_t index = 0; index < source_parameters.size(); ++index)
            {
                const double tolerance =
                    index < 2 ? 1.0e-8 * std::max(1.0, std::abs(source_parameters[index])) : 1.0e-10;
                changed = changed || std::abs(published_parameters[index] - source_parameters[index]) > tolerance;
            }
            if (changed)
            {
                ++result->refinedIntrinsicCount;
            }
        }
        const double count = static_cast<double>(result->refinedCameras.size());
        result->refinedCalibrationGroupCount = static_cast<int>(groups.size());
        result->refinedSharedFocalScale = focal_scale_sum / count;
        result->refinedSharedFocalAspectScale = aspect_scale_sum / count;
        result->refinedSharedPrincipalOffsetX = remaining_sums[0] / count;
        result->refinedSharedPrincipalOffsetY = remaining_sums[1] / count;
        result->refinedSharedRadialK1 = remaining_sums[2] / count;
        result->refinedSharedRadialK2 = remaining_sums[3] / count;
        result->refinedSharedRadialK3 = remaining_sums[4] / count;
        result->refinedSharedTangentialP1 = remaining_sums[5] / count;
        result->refinedSharedTangentialP2 = remaining_sums[6] / count;
        result->refinedSharedSkewB2 = remaining_sums[7] / count;
        result->refinedSharedRadialK4 = remaining_sums[8] / count;
        result->refinedSharedTangentialP3 = remaining_sums[9] / count;
        result->refinedSharedTangentialP4 = remaining_sums[10] / count;
    }

} // namespace plabundle::internal::plamatrix_ba
