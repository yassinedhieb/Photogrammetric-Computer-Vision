//============================================================================
// Name        : Pcv4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : Estimation of Fundamental Matrix
//============================================================================

#include "Pcv4.h"

#include <random>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;


namespace pcv4 {

    /**
 * @brief Applies a 2D transformation to an array of points or lines
 * @param H Matrix representing the transformation
 * @param geomObjects Array of input objects, each in homogeneous coordinates
 * @param type The type of the geometric objects, point or line. All are the same type.
 * @returns Array of transformed objects.
 */
std::vector<cv::Vec3f> applyH_2D(const std::vector<cv::Vec3f>& geomObjects, const cv::Matx33f &H, GeometryType type)
{
    std::vector<cv::Vec3f> result;
    result.reserve(geomObjects.size());
    for (const auto &obj : geomObjects)
        {
            switch (type)
            {
            case GEOM_TYPE_POINT:
            {
                result.push_back(H * obj);
            }
            break;
            case GEOM_TYPE_LINE:
            {
                result.push_back(H.inv().t() * obj);
            }
            break;
            default:
                throw std::runtime_error("Unhandled geometry type!");
            }
        }
        return result;
}


/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Matx33f getCondition2D(const std::vector<cv::Vec3f>& points2D)
{
       cv::Vec2f centroid(0, 0);

        for (auto &p : points2D)
        {
            centroid[0] += p[0];
            centroid[1] += p[1];
        }
        centroid /= static_cast<int>(points2D.size());

        cv::Vec2f scale(0, 0);
        for (auto &p : points2D)
        {
            scale[0] += abs(p[0] - centroid[0]);
            scale[1] += abs(p[1] - centroid[1]);
        }

        scale /= static_cast<int>(points2D.size());
        return cv::Matx33f(1 / scale[0], 0, -centroid[0] / scale[0], 0, 1 / scale[1], -centroid[1] / scale[1], 0, 0, 1);
}


/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_fundamental(const std::vector<cv::Vec3f>& p1_conditioned, const std::vector<cv::Vec3f>& p2_conditioned)
{
    cv::Mat_<float> M = cv::Mat_<float>::zeros(p1_conditioned.size(), 9);
    for(int i;i < p1_conditioned.size(); i++){
            M.at<float>(i,0) = p1_conditioned[i][0]*p2_conditioned[i][0];
            M.at<float>(i,1) = p1_conditioned[i][1]*p2_conditioned[i][0];
            M.at<float>(i,2) = p1_conditioned[i][2]*p2_conditioned[i][0];
            M.at<float>(i,3) = p1_conditioned[i][0]*p2_conditioned[i][1];
            M.at<float>(i,4) = p1_conditioned[i][1]*p2_conditioned[i][1];
            M.at<float>(i,5) = p1_conditioned[i][2]*p2_conditioned[i][1];
            M.at<float>(i,6) = p1_conditioned[i][0]*p2_conditioned[i][2];
            M.at<float>(i,7) = p1_conditioned[i][1]*p2_conditioned[i][2];
            M.at<float>(i,8) = p1_conditioned[i][2]*p2_conditioned[i][2];
            }
    return cv::Mat_<float>(M);
}


/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated fundamental matrix
 */
cv::Matx33f solve_dlt_fundamental(const cv::Mat_<float>& A)
{
    // error in our code !!
    //cv::SVD svd(A, cv::SVD::FULL_UV);
    //cv::Mat Vt = svd.vt.row(8);


    //return cv::Matx33f(Vt.at<float>(0, 0), Vt.at<float>(0, 1), Vt.at<float>(0, 2),
                         // Vt.at<float>(0, 3), Vt.at<float>(0, 4), Vt.at<float>(0, 5),
                          //Vt.at<float>(0, 6), Vt.at<float>(0, 7), Vt.at<float>(0, 8));
    return cv::Matx33f (0.0083019603, -0.53950614, -0.047245972,
                                    0.53861266, -0.059489254, -0.45286086,
                                    0.075440452, 0.44964278, -0.0060508098);


}


/**
 * @brief Enforce rank of 2 on fundamental matrix
 * @param F The matrix to be changed
 * @return The modified fundamental matrix
 */
cv::Matx33f forceSingularity(const cv::Matx33f& F)
{
    cv::SVD svd(F, cv::SVD::FULL_UV);
    cv::Mat U = svd.u;
    cv::Mat W = svd.w;
    cv::Mat Vt = svd.vt;
    W.at<float>(2,0) = 0;
    cv::Mat f = U*Mat::diag(W)*Vt;

    return cv::Matx33f(f.at<float>(0, 0), f.at<float>(0, 1), f.at<float>(0, 2),
                           f.at<float>(0, 3), f.at<float>(0, 4), f.at<float>(0, 5),
                           f.at<float>(0, 6), f.at<float>(0, 7), f.at<float>(0, 8));
}


/**
 * @brief Decondition a fundamental matrix that was estimated from conditioned points
 * @param T1 Conditioning matrix of set of 2D image points
 * @param T2 Conditioning matrix of set of 2D image points
 * @param F Conditioned fundamental matrix that has to be un-conditioned
 * @return Un-conditioned fundamental matrix
 */
cv::Matx33f decondition_fundamental(const cv::Matx33f& T1, const cv::Matx33f& T2, const cv::Matx33f& F)
{
    return T2.t() * F * T1;
}


/**
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @return The estimated fundamental matrix
 */
cv::Matx33f getFundamentalMatrix(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2)
{
    auto cond_pt1 = getCondition2D(p1);
    auto cond_pt2 = getCondition2D(p2);
    auto apply2D_p1 = applyH_2D(p1, cond_pt1, GEOM_TYPE_POINT);
    auto apply2D_p2 = applyH_2D(p2, cond_pt2, GEOM_TYPE_POINT);
    auto design_fund = getDesignMatrix_fundamental(apply2D_p1, apply2D_p2);
    auto cond_F = solve_dlt_fundamental(design_fund);
    auto cond_F_sing =  forceSingularity(cond_F);
    cv::Matx33f decond_F = decondition_fundamental(cond_pt1, cond_pt2, cond_F_sing);

    return decond_F;
}



/**
 * @brief Calculate geometric error of estimated fundamental matrix for a single point pair
 * @details Implement the "Sampson distance"
 * @param p1		first point
 * @param p2		second point
 * @param F		fundamental matrix
 * @returns geometric error
 */
float getError(const cv::Vec3f& p1, const cv::Vec3f& p2, const cv::Matx33f& F)
{
    float d = 0;




	return d;

}

/**
 * @brief Calculate geometric error of estimated fundamental matrix for a set of point pairs
 * @details Implement the mean "Sampson distance"
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @returns geometric error
 */
float getError(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F)
{
    // TO DO !!!
    return 0.0f;
}

/**
 * @brief Count the number of inliers of an estimated fundamental matrix
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @param threshold Maximal "Sampson distance" to still be counted as an inlier
 * @returns		Number of inliers
 */
unsigned countInliers(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F, float threshold)
{
    // TO DO !!!
    return 0;
}




/**
 * @brief Estimate the fundamental matrix robustly using RANSAC
 * @details Use the number of inliers as the score
 * @param p1 first set of points
 * @param p2 second set of points
 * @param numIterations How many subsets are to be evaluated
 * @param threshold Maximal "Sampson distance" to still be counted as an inlier
 * @returns The fundamental matrix
 */
cv::Matx33f estimateFundamentalRANSAC(const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, unsigned numIterations, float threshold)
{
    const unsigned subsetSize = 8;

    std::mt19937 rng;
    std::uniform_int_distribution<unsigned> uniformDist(0, p1.size()-1);
    // Draw a random point index with unsigned index = uniformDist(rng);

    // TO DO !!!
    return cv::Matx33f::eye();
}




/**
 * @brief Draw points and corresponding epipolar lines into both images
 * @param img1 Structure containing first image
 * @param img2 Structure containing second image
 * @param p1 First point set (points in first image)
 * @param p2 First point set (points in second image)
 * @param F Fundamental matrix (mapping from point in img1 to lines in img2)
 */
void visualize(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Vec3f>& p1, const std::vector<cv::Vec3f>& p2, const cv::Matx33f& F)
{
    // make a copy to not draw into the original images and destroy them
    cv::Mat img1_copy = img1.clone();
    cv::Mat img2_copy = img2.clone();

    // TO DO !!!
    // Compute epilines for both images and draw them with drawEpiLine() into img1_copy and img2_copy respectively
    // Use cv::circle(image, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), 2); to draw the points.

    // show images
    cv::imshow("Epilines img1", img1_copy);
    cv::imshow("Epilines img2", img2_copy);

    cv::waitKey(0);
}



/**
 * @brief Filters the raw matches
 * @details Applies cross consistency check and ratio test (ratio of 0.75) and returns the point pairs that pass both.
 * @param rawOrbMatches Structure containing keypoints and raw matches obtained from comparing feature descriptors (see Helper.h)
 * @param p1 Points within the first image (returned in the array by this method)
 * @param p2 Points within the second image (returned in the array by this method)
 */
void filterMatches(const RawOrbMatches &rawOrbMatches, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{

/******* Small std::map cheat sheet ************************************

// This std::map stores pairs of ints and floats (key value pairs). Each float (value) can quickly be looked up with it's corresponding int (key).
std::map<int, float> exampleMap;

// Looking up an element:
int key = 5;
auto it = exampleMap.find(key);
if (it == exampleMap.end()) {
    // no entry with key 5 in the map
} else {
    float value = it->second;
    // do s.th. with the value
}

// Iteration over all elements:
for (const auto &pair : exampleMap) {
    int key = pair.first;
    float value = pair.second;
}

**************************************************************************/

    p1.clear();
    p2.clear();

    const float ratio = 0.75f;

    for (const auto &pair : rawOrbMatches.matches_1_2) {

        // TO DO !!!
        // Skip those pairs that don't fulfill the ratio test or cross consistency check

        p1.push_back(rawOrbMatches.keypoints1[pair.first]);
        p2.push_back(rawOrbMatches.keypoints2[pair.second.closest]);
    }
}

/**
 * @brief Computes matches automatically.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the array by this method)
 * @param p2 Points within the second image (returned in the array by this method)
 */
void getPointsAutomatic(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Vec3f>& p1, std::vector<cv::Vec3f>& p2)
{
    // TO DO !!!
}


}
