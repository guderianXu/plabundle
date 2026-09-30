#pragma once

#include <plabundle/options.h>
#include <plabundle/problem.h>

#include <array>
#include <string>
#include <vector>

namespace plabundle
{

    struct AdaptiveCameraModelAssessment
    {
        bool valid = false;
        int cameraCount = 0;
        int activeCameraCount = 0;
        int trackCount = 0;
        int observationCount = 0;
        int multiViewTrackCount = 0;
        int occupiedPeripheralSectors = 0;
        bool hasAbsoluteGeometryConstraint = false;
        bool unanchoredParallelAerialGuardApplied = false;
        double opticalAxisConcentration = 1.0;
        double medianTriangulationAngleDegrees = 0.0;
        double multiViewTrackRatio = 0.0;
        double normalizedRadiusP90 = 0.0;
        double maximumNormalizedRadius = 0.0;
        double peripheralRadiusThreshold = 0.30;
        double lowOrderDistortionScale = 1.0;
        double geometryStrength = 0.0;
        double observationSupport = 0.0;
        double peripheralCoverage = 0.0;
        double sectorCoverage = 0.0;
        double imageAxisBalance = 0.0;
        std::array<double, kIntrinsicParameterCount> incrementalInformationScore{};
        std::array<double, kIntrinsicParameterCount> sensitivity{};
        std::array<double, kIntrinsicParameterCount> reliability{};
        IntrinsicParameterMask enabled{};
        std::string modelName = "fixed";
        std::string reason;
    };

    const char* intrinsicParameterName(IntrinsicParameter parameter) noexcept;
    int enabledIntrinsicParameterCount(const IntrinsicParameterMask& mask) noexcept;
    std::string adaptiveCameraModelName(const IntrinsicParameterMask& mask);
    AdaptiveCameraModelAssessment assessAdaptiveCameraModel(const Problem& problem, const Options* options = nullptr);
    AdaptiveCameraModelAssessment assessAdaptiveCameraModel(const Problem& problem, const SolveOptions& options);
    bool applyAdaptiveCameraModel(const AdaptiveCameraModelAssessment& assessment, Options* options);
    bool applyAdaptiveCameraModel(const AdaptiveCameraModelAssessment& assessment, SolveOptions* options);
    bool restoreInactiveAdaptiveIntrinsics(std::vector<placamera::FramePinholeNumericState>* cameras,
                                           const std::vector<placamera::FramePinholeNumericState>& stableReferences,
                                           const IntrinsicParameterMask& activeMask);

} // namespace plabundle
