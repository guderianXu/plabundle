#include <plabundle/constraints.h>

#include "control_point_internal.h"
#include "pose_prior_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>

namespace
{
    template <int Dimension>
    void expectWhitenedIdentity(const std::array<double, 36>& weight,
                                const std::array<double, 36>& covariance,
                                const std::array<int, 6>& components)
    {
        for (int row = 0; row < Dimension; ++row)
        {
            for (int column = 0; column < Dimension; ++column)
            {
                double value = 0.0;
                for (int left = 0; left < Dimension; ++left)
                {
                    for (int right = 0; right < Dimension; ++right)
                    {
                        value += weight[static_cast<std::size_t>(row * 6 + components[left])] *
                                 covariance[static_cast<std::size_t>(components[left] * 6 + components[right])] *
                                 weight[static_cast<std::size_t>(column * 6 + components[right])];
                    }
                }
                EXPECT_NEAR(value, row == column ? 1.0 : 0.0, 1.0e-11);
            }
        }
    }
} // namespace

TEST(PlaBundleCovarianceWhiteningTest, ControlPointCovarianceProducesIdentityInformation)
{
    plabundle::ControlPointConstraint constraint;
    constraint.uncertainty = plabundle::ControlPointUncertainty::Covariance;
    constraint.uncertaintyMatrix = {{4.0, 2.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 9.0}};

    plabundle::internal::ControlPointWhitening whitening;
    std::string error;
    ASSERT_TRUE(plabundle::internal::makeControlPointWhitening(constraint, &whitening, &error)) << error;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            double value = 0.0;
            for (int left = 0; left < 3; ++left)
            {
                for (int right = 0; right < 3; ++right)
                {
                    value += whitening.matrix[static_cast<std::size_t>(row * 3 + left)] *
                             constraint.uncertaintyMatrix[static_cast<std::size_t>(left * 3 + right)] *
                             whitening.matrix[static_cast<std::size_t>(column * 3 + right)];
                }
            }
            EXPECT_NEAR(value, row == column ? 1.0 : 0.0, 1.0e-12);
        }
    }

    double rms_uncertainty = 0.0;
    ASSERT_TRUE(plabundle::internal::controlPointRmsUncertaintyMeters(constraint, &rms_uncertainty, &error)) << error;
    EXPECT_NEAR(rms_uncertainty, std::sqrt(18.0), 1.0e-12);
}

TEST(PlaBundleCovarianceWhiteningTest, ControlPointRejectsNearSingularCovarianceAndInformation)
{
    plabundle::ControlPointConstraint constraint;
    constraint.uncertainty = plabundle::ControlPointUncertainty::Covariance;
    constraint.uncertaintyMatrix = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-14}};
    std::string error;
    EXPECT_FALSE(plabundle::validateControlPointConstraint(constraint, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);

    constraint.uncertainty = plabundle::ControlPointUncertainty::SqrtInformation;
    constraint.uncertaintyMatrix = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}};
    EXPECT_FALSE(plabundle::validateControlPointConstraint(constraint, &error));
    EXPECT_NE(error.find("positive definite"), std::string::npos);
}

TEST(PlaBundleCovarianceWhiteningTest, ControlPointAcceptsCorrelatedSqrtInformation)
{
    plabundle::ControlPointConstraint constraint;
    constraint.uncertainty = plabundle::ControlPointUncertainty::SqrtInformation;
    constraint.uncertaintyMatrix = {{2.0, 0.5, 0.0, 0.0, 3.0, -0.25, 0.0, 0.0, 4.0}};

    plabundle::internal::ControlPointWhitening whitening;
    std::string error;
    ASSERT_TRUE(plabundle::internal::makeControlPointWhitening(constraint, &whitening, &error)) << error;
    EXPECT_EQ(whitening.matrix, constraint.uncertaintyMatrix);
}

TEST(PlaBundleCovarianceWhiteningTest, PosePriorSixDimensionalCovarianceWhitensCorrelations)
{
    plabundle::CameraPosePrior prior;
    prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
    const std::array<double, 6> sigma{{0.2, 0.3, 0.4, 2.0, 3.0, 4.0}};
    std::array<double, 36> lower{};
    for (int row = 0; row < 6; ++row)
    {
        lower[static_cast<std::size_t>(row * 6 + row)] = sigma[static_cast<std::size_t>(row)];
    }
    lower[3 * 6] = 0.05;
    lower[5 * 6 + 4] = -0.4;
    for (int row = 0; row < 6; ++row)
    {
        for (int column = 0; column < 6; ++column)
        {
            for (int inner = 0; inner < 6; ++inner)
            {
                prior.uncertaintyMatrix[static_cast<std::size_t>(row * 6 + column)] +=
                    lower[static_cast<std::size_t>(row * 6 + inner)] *
                    lower[static_cast<std::size_t>(column * 6 + inner)];
            }
        }
    }

    plabundle::internal::PosePriorWhitening whitening;
    std::string error;
    ASSERT_TRUE(plabundle::internal::makePosePriorWhitening(prior, &whitening, &error)) << error;
    EXPECT_EQ(whitening.residualSize, 6);
    expectWhitenedIdentity<6>(whitening.matrix, prior.uncertaintyMatrix, whitening.componentIndices);
}

TEST(PlaBundleCovarianceWhiteningTest, PositionOnlyPriorSelectsThePositionSubmatrix)
{
    plabundle::CameraPosePrior prior;
    prior.components = plabundle::PosePriorComponents::Position;
    prior.uncertainty = plabundle::PosePriorUncertainty::Covariance;
    prior.uncertaintyMatrix[3 * 6 + 3] = 4.0;
    prior.uncertaintyMatrix[4 * 6 + 4] = 9.0;
    prior.uncertaintyMatrix[5 * 6 + 5] = 16.0;
    prior.uncertaintyMatrix[3 * 6 + 4] = 1.0;
    prior.uncertaintyMatrix[4 * 6 + 3] = 1.0;

    plabundle::internal::PosePriorWhitening whitening;
    std::string error;
    ASSERT_TRUE(plabundle::internal::makePosePriorWhitening(prior, &whitening, &error)) << error;
    EXPECT_EQ(whitening.residualSize, 3);
    expectWhitenedIdentity<3>(whitening.matrix, prior.uncertaintyMatrix, whitening.componentIndices);
}

TEST(PlaBundleCovarianceWhiteningTest, PositionOnlyPriorAcceptsCorrelatedSqrtInformation)
{
    plabundle::CameraPosePrior prior;
    prior.components = plabundle::PosePriorComponents::Position;
    prior.uncertainty = plabundle::PosePriorUncertainty::SqrtInformation;
    prior.uncertaintyMatrix[3 * 6 + 3] = 2.0;
    prior.uncertaintyMatrix[3 * 6 + 4] = 0.5;
    prior.uncertaintyMatrix[4 * 6 + 4] = 3.0;
    prior.uncertaintyMatrix[5 * 6 + 5] = 4.0;

    plabundle::internal::PosePriorWhitening whitening;
    std::string error;
    ASSERT_TRUE(plabundle::internal::makePosePriorWhitening(prior, &whitening, &error)) << error;
    EXPECT_EQ(whitening.residualSize, 3);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            const int component = whitening.componentIndices[static_cast<std::size_t>(column)];
            EXPECT_DOUBLE_EQ(whitening.matrix[static_cast<std::size_t>(row * 6 + component)],
                             prior.uncertaintyMatrix[static_cast<std::size_t>((row + 3) * 6 + component)]);
        }
    }
}
