#include <plabundle/options.h>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>

TEST(PlaBundleOptionsTest, StructuredDefaultsMatchCompatibilityDefaults)
{
    const plabundle::SolveOptions structured;
    const plabundle::Options compatibility = plabundle::makeCompatibilityOptions(structured);

    EXPECT_EQ(compatibility.maxIterations, plabundle::Options{}.maxIterations);
    EXPECT_EQ(compatibility.imageRobustLoss, plabundle::ImageRobustLoss::LeastSquares);
    EXPECT_DOUBLE_EQ(compatibility.imageRobustLossScalePixels, 1.0);
    EXPECT_EQ(compatibility.refineCameraPose, plabundle::Options{}.refineCameraPose);
    EXPECT_EQ(compatibility.controlPointWeight, plabundle::Options{}.controlPointWeight);
    EXPECT_EQ(compatibility.backend, plabundle::Options{}.backend);
    EXPECT_EQ(compatibility.enableBackendQualityGate, plabundle::Options{}.enableBackendQualityGate);
    EXPECT_EQ(plabundle::BackendOptions::kAutoPolicyVersion, plabundle::Options::kAutoBackendPolicyVersion);
    EXPECT_TRUE(plabundle::validateOptions(structured));
    EXPECT_TRUE(plabundle::validateOptions(compatibility));
}

TEST(PlaBundleOptionsTest, StructuredConfigurationRoundTripsWithoutLosingGroups)
{
    plabundle::SolveOptions structured;
    structured.solver.maxIterations = 37;
    structured.solver.imageRobustLoss = plabundle::ImageRobustLoss::Cauchy;
    structured.solver.imageRobustLossScalePixels = 2.75;
    structured.solver.referenceArmijoCoefficient = 2.0e-3;
    structured.solver.enablePointFilter = false;
    structured.solver.numThreads = 3;
    structured.solver.cancelFlag = std::make_shared<std::atomic<bool>>(false);
    structured.calibration.refineCameraPose = false;
    structured.calibration.refineSharedFocalLength = true;
    structured.calibration.refineSharedModelCoefficients = true;
    structured.calibration.refineSharedMetashapeParameters = true;
    structured.calibration.useSharedIntrinsicParameterMask = true;
    structured.calibration.sharedIntrinsicParameterMask.fill(false);
    structured.calibration.sharedIntrinsicParameterMask[0] = true;
    structured.calibration.sharedIntrinsicParameterMask[12] = true;
    structured.calibration.maxSharedSkewFraction = 0.12;
    structured.calibration.maxSharedRadialK4Abs = 0.28;
    structured.calibration.maxSharedTangentialP3Abs = 0.22;
    structured.calibration.maxSharedTangentialP4Abs = 0.18;
    structured.calibration.sharedSkewPriorSigmaFraction = 0.03;
    structured.calibration.sharedRadialK4PriorSigma = 0.12;
    structured.calibration.sharedTangentialP3PriorSigma = 0.11;
    structured.calibration.sharedTangentialP4PriorSigma = 0.10;
    structured.constraints.controlPointWeight = 12.0;
    structured.constraints.controlPointHuberDeltaMeters = 0.4;
    structured.backend.requested = plabundle::Backend::Auto;
    structured.backend.plaMatrixDevice = 2;
    structured.backend.minPlaMatrixCudaCameras = 17;
    structured.backend.enablePlaMatrixMixedPrecision = true;
    structured.backend.allowFallback = false;
    structured.quality.enabled = false;
    structured.quality.minAcceptedValidTrackRatio = 0.75;

    const plabundle::Options compatibility = plabundle::makeCompatibilityOptions(structured);
    const plabundle::SolveOptions round_trip = plabundle::makeSolveOptions(compatibility);

    EXPECT_EQ(round_trip.solver.maxIterations, 37);
    EXPECT_EQ(round_trip.solver.imageRobustLoss, plabundle::ImageRobustLoss::Cauchy);
    EXPECT_DOUBLE_EQ(round_trip.solver.imageRobustLossScalePixels, 2.75);
    EXPECT_DOUBLE_EQ(round_trip.solver.referenceArmijoCoefficient, 2.0e-3);
    EXPECT_FALSE(round_trip.solver.enablePointFilter);
    EXPECT_EQ(round_trip.solver.numThreads, 3);
    EXPECT_EQ(round_trip.solver.cancelFlag, structured.solver.cancelFlag);
    EXPECT_FALSE(round_trip.calibration.refineCameraPose);
    EXPECT_TRUE(round_trip.calibration.refineSharedFocalLength);
    EXPECT_TRUE(round_trip.calibration.refineSharedModelCoefficients);
    EXPECT_TRUE(round_trip.calibration.refineSharedMetashapeParameters);
    EXPECT_TRUE(round_trip.calibration.useSharedIntrinsicParameterMask);
    EXPECT_TRUE(round_trip.calibration.sharedIntrinsicParameterMask[0]);
    EXPECT_TRUE(round_trip.calibration.sharedIntrinsicParameterMask[12]);
    EXPECT_DOUBLE_EQ(round_trip.calibration.maxSharedSkewFraction, 0.12);
    EXPECT_DOUBLE_EQ(round_trip.calibration.maxSharedRadialK4Abs, 0.28);
    EXPECT_DOUBLE_EQ(round_trip.calibration.maxSharedTangentialP3Abs, 0.22);
    EXPECT_DOUBLE_EQ(round_trip.calibration.maxSharedTangentialP4Abs, 0.18);
    EXPECT_DOUBLE_EQ(round_trip.calibration.sharedSkewPriorSigmaFraction, 0.03);
    EXPECT_DOUBLE_EQ(round_trip.calibration.sharedRadialK4PriorSigma, 0.12);
    EXPECT_DOUBLE_EQ(round_trip.calibration.sharedTangentialP3PriorSigma, 0.11);
    EXPECT_DOUBLE_EQ(round_trip.calibration.sharedTangentialP4PriorSigma, 0.10);
    EXPECT_DOUBLE_EQ(round_trip.constraints.controlPointWeight, 12.0);
    EXPECT_DOUBLE_EQ(round_trip.constraints.controlPointHuberDeltaMeters, 0.4);
    EXPECT_EQ(round_trip.backend.requested, plabundle::Backend::Auto);
    EXPECT_EQ(round_trip.backend.plaMatrixDevice, 2);
    EXPECT_EQ(round_trip.backend.minPlaMatrixCudaCameras, 17);
    EXPECT_TRUE(round_trip.backend.enablePlaMatrixMixedPrecision);
    EXPECT_FALSE(round_trip.backend.allowFallback);
    EXPECT_FALSE(round_trip.quality.enabled);
    EXPECT_DOUBLE_EQ(round_trip.quality.minAcceptedValidTrackRatio, 0.75);
}

TEST(PlaBundleOptionsTest, StructuredValidationReportsTheOwningGroupFailure)
{
    plabundle::SolveOptions options;
    options.backend.plaMatrixPreconditionerClusterSize = 0;
    std::string error;
    EXPECT_FALSE(plabundle::validateOptions(options, &error));
    EXPECT_NE(error.find("preconditioner"), std::string::npos);

    options = {};
    options.calibration.minSharedFocalScale = -1.0;
    EXPECT_FALSE(plabundle::validateOptions(options, &error));
    EXPECT_NE(error.find("focal"), std::string::npos);

    options = {};
    options.solver.imageRobustLossScalePixels = 0.0;
    EXPECT_FALSE(plabundle::validateOptions(options, &error));
    EXPECT_NE(error.find("robust loss scale"), std::string::npos);

    options = {};
    options.solver.imageRobustLoss = static_cast<plabundle::ImageRobustLoss>(99);
    EXPECT_FALSE(plabundle::validateOptions(options, &error));
    EXPECT_NE(error.find("robust loss value"), std::string::npos);
}

TEST(PlaBundleOptionsTest, ZeroDisablesOnlyTheImageRmsGrowthGate)
{
    plabundle::SolveOptions structured;
    structured.quality.maxAcceptedRmsGrowth = 0.0;
    EXPECT_TRUE(plabundle::validateOptions(structured));

    const plabundle::Options compatibility =
        plabundle::makeCompatibilityOptions(structured);
    EXPECT_DOUBLE_EQ(compatibility.maxAcceptedRmsGrowth, 0.0);
    EXPECT_TRUE(compatibility.enableBackendQualityGate);
    EXPECT_TRUE(plabundle::validateOptions(compatibility));

    structured.quality.maxAcceptedRmsGrowth = 0.5;
    EXPECT_FALSE(plabundle::validateOptions(structured));
}
