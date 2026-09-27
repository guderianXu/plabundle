#include <plabundle/photo_alignment.h>

#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

namespace
{

    plabundle::FrameCamera makeCamera(double center_x)
    {
        plabundle::FrameCamera camera;
        camera.cameraCenter = {center_x, 0.0, 0.0};
        camera.focalXPixels = 800.0;
        camera.focalYPixels = 800.0;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        camera.imageSize = plabundle::ImageSize{1024, 768};
        return camera;
    }

    plabundle::PhotoAlignmentProblem makeProblem()
    {
        plabundle::PhotoAlignmentProblem alignment;
        alignment.cameraIds = {"image-left", "image-right"};
        alignment.trackIds = {"tie-42"};
        alignment.problem.cameras = {makeCamera(-1.0), makeCamera(1.0)};
        plabundle::Track track;
        track.initialPoint = {0.0, 0.0, 8.0};
        track.observations = {{0, 412.0, 384.0, 1.0, 1.0}, {1, 612.0, 384.0, 1.0, 1.0}};
        alignment.problem.tracks.push_back(track);
        return alignment;
    }

    plabundle::Result makeResult(const plabundle::PhotoAlignmentProblem& problem)
    {
        plabundle::Result result;
        result.status = plabundle::SolveStatus::Success;
        result.solutionUsable = true;
        result.requestedBackend = plabundle::Backend::Auto;
        result.usedBackend = plabundle::Backend::PlaMatrixCpu;
        result.backendMessage = "CONVERGENCE";
        result.backendSelectionReason = "small_problem_cpu";
        result.quality.meanRmsBefore = 2.0;
        result.quality.meanRmsAfter = 0.25;
        result.refinedCameras = problem.problem.cameras;
        result.points.resize(problem.problem.tracks.size());
        result.points[0].valid = true;
        result.points[0].converged = true;
        result.points[0].point = {0.0, 0.0, 8.0};
        result.points[0].rmsBefore = 2.0;
        result.points[0].rmsAfter = 0.25;
        return result;
    }

} // namespace

TEST(PhotoAlignmentTest, ValidatesStableIdentifierBinding)
{
    plabundle::PhotoAlignmentProblem problem = makeProblem();
    std::string error;
    EXPECT_TRUE(plabundle::validatePhotoAlignmentProblem(problem, &error)) << error;

    problem.cameraIds[1] = problem.cameraIds[0];
    EXPECT_FALSE(plabundle::validatePhotoAlignmentProblem(problem, &error));
    EXPECT_NE(error.find("unique"), std::string::npos);

    problem = makeProblem();
    problem.trackIds.clear();
    EXPECT_FALSE(plabundle::validatePhotoAlignmentProblem(problem, &error));
    EXPECT_NE(error.find("align"), std::string::npos);
}

TEST(PhotoAlignmentTest, NormalizesPlaBundleResultWithoutLosingExternalIdentity)
{
    const plabundle::PhotoAlignmentProblem problem = makeProblem();
    const plabundle::Result result = makeResult(problem);
    plabundle::PhotoAlignmentOutcome outcome;
    std::string error;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(problem, result, &outcome, &error)) << error;

    EXPECT_EQ(outcome.terminationStatus, plabundle::SolveStatus::Success);
    EXPECT_EQ(outcome.terminationReason, "CONVERGENCE");
    EXPECT_EQ(outcome.requestedBackend, plabundle::Backend::Auto);
    EXPECT_EQ(outcome.usedBackend, plabundle::Backend::PlaMatrixCpu);
    ASSERT_EQ(outcome.cameras.size(), 2U);
    EXPECT_EQ(outcome.cameras[0].id, "image-left");
    ASSERT_EQ(outcome.tracks.size(), 1U);
    EXPECT_EQ(outcome.tracks[0].id, "tie-42");
    EXPECT_TRUE(outcome.tracks[0].point.valid);
}

TEST(PhotoAlignmentTest, ComparesOutcomesByIdentifierInsteadOfStorageOrder)
{
    const plabundle::PhotoAlignmentProblem problem = makeProblem();
    plabundle::PhotoAlignmentOutcome reference;
    std::string error;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(problem, makeResult(problem), &reference, &error)) << error;
    plabundle::PhotoAlignmentOutcome candidate = reference;
    std::reverse(candidate.cameras.begin(), candidate.cameras.end());

    plabundle::PhotoAlignmentComparison comparison;
    ASSERT_TRUE(plabundle::comparePhotoAlignmentOutcomes(
        reference, candidate, plabundle::PhotoAlignmentComparisonTolerance{}, &comparison, &error))
        << error;
    EXPECT_TRUE(comparison.comparable);
    EXPECT_TRUE(comparison.equivalent);
    EXPECT_TRUE(comparison.validTrackMaskMismatchIds.empty());
    ASSERT_EQ(comparison.cameraDifferences.size(), 2U);
    EXPECT_EQ(comparison.cameraDifferences[0].id, "image-left");
}

TEST(PhotoAlignmentTest, ReportsNumericMaskTerminationAndBackendDifferences)
{
    const plabundle::PhotoAlignmentProblem problem = makeProblem();
    plabundle::PhotoAlignmentOutcome reference;
    std::string error;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(problem, makeResult(problem), &reference, &error)) << error;
    plabundle::PhotoAlignmentOutcome candidate = reference;
    candidate.terminationStatus = plabundle::SolveStatus::NoConvergence;
    candidate.terminationReason = "NO_CONVERGENCE";
    candidate.usedBackend = plabundle::Backend::PlaMatrixOpenCl;
    candidate.meanRmsAfter += 0.1;
    candidate.cameras[0].camera.cameraCenter[0] += 0.2;
    constexpr double kHalfPi = 1.57079632679489661923;
    candidate.cameras[1].camera.cameraToWorldRotation = {
        std::cos(kHalfPi), -std::sin(kHalfPi), 0.0, std::sin(kHalfPi), std::cos(kHalfPi), 0.0, 0.0, 0.0, 1.0};
    candidate.tracks[0].point.valid = false;

    plabundle::PhotoAlignmentComparison comparison;
    ASSERT_TRUE(plabundle::comparePhotoAlignmentOutcomes(
        reference, candidate, plabundle::PhotoAlignmentComparisonTolerance{}, &comparison, &error))
        << error;
    EXPECT_FALSE(comparison.equivalent);
    EXPECT_FALSE(comparison.terminationStatusMatches);
    EXPECT_FALSE(comparison.terminationReasonMatches);
    EXPECT_FALSE(comparison.usedBackendMatches);
    EXPECT_NEAR(comparison.rmsAfterDifference, 0.1, 1.0e-12);
    EXPECT_NEAR(comparison.maximumCameraCenterDifferenceMeters, 0.2, 1.0e-12);
    EXPECT_NEAR(comparison.maximumCameraRotationDifferenceDegrees, 90.0, 1.0e-12);
    EXPECT_EQ(comparison.validTrackMaskMismatchIds, std::vector<std::string>{"tie-42"});
}

TEST(PhotoAlignmentTest, RejectsMismatchedIdentifierSetsAndInvalidTolerance)
{
    const plabundle::PhotoAlignmentProblem problem = makeProblem();
    plabundle::PhotoAlignmentOutcome reference;
    std::string error;
    ASSERT_TRUE(plabundle::makePhotoAlignmentOutcome(problem, makeResult(problem), &reference, &error)) << error;
    plabundle::PhotoAlignmentOutcome candidate = reference;
    candidate.cameras[0].id = "another-image";

    plabundle::PhotoAlignmentComparison comparison;
    EXPECT_FALSE(plabundle::comparePhotoAlignmentOutcomes(
        reference, candidate, plabundle::PhotoAlignmentComparisonTolerance{}, &comparison, &error));
    EXPECT_NE(error.find("same camera and track ids"), std::string::npos);

    candidate = reference;
    plabundle::PhotoAlignmentComparisonTolerance tolerance;
    tolerance.rms = -1.0;
    EXPECT_FALSE(plabundle::comparePhotoAlignmentOutcomes(reference, candidate, tolerance, &comparison, &error));
    EXPECT_NE(error.find("non-negative"), std::string::npos);
}
