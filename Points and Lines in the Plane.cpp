//============================================================================
// Name        : Pcv2test.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description :
//============================================================================

#include "Pcv2.h"

namespace pcv2 {


/**
 * @brief get the conditioning matrix of given points
 * @param the points as matrix
 * @returns the condition matrix (already allocated)
 */
cv::Matx33f getCondition2D(const std::vector<cv::Vec3f> &points)
    {
        float tx = 0, ty = 0;
        for (auto &p : points)
        {
            tx += p[0];
            ty += p[1];
        }
        tx /= 4;
        ty /= 4;

        float sx = 0, sy = 0;
        for (auto &p : points)
        {
            sx += abs(p[0] - tx);
            sy += abs(p[1] - ty);
        }

        sx /= 4;
        sy /= 4;

        return cv::Matx33f(1 / sx, 0, -tx / sx, 0, 1 / sy, -ty / sy, 0, 0, 1);
    }


/**
 * @brief define the design matrix as needed to compute 2D-homography
 * @param conditioned_base first set of conditioned points x' --> x' = H * x
 * @param conditioned_attach second set of conditioned points x --> x' = H * x
 * @returns the design matrix to be computed
 */
cv::Mat_<float> getDesignMatrix_homography2D(const std::vector<cv::Vec3f> &conditioned_base, const std::vector<cv::Vec3f> &conditioned_attach)
{
    int i=0, j=0;
    int n = conditioned_base.size();

    cv::Mat_<float> M = cv::Mat_<float>::zeros(9, 9);

            for(i,j;i<4, j<2*n;i++,j++){
            M.at<float>(j,0) = -conditioned_base[i][2]*conditioned_attach[i][0];
            M.at<float>(j,1) = -conditioned_base[i][2]*conditioned_attach[i][1];
            M.at<float>(j,2) = -conditioned_base[i][2]*conditioned_attach[i][2];
            M.at<float>(j,3) = 0.f;
            M.at<float>(j,4) = 0.f;
            M.at<float>(j,5) = 0.f;
            M.at<float>(j,6) = conditioned_base[i][0]*conditioned_attach[i][0];
            M.at<float>(j,7) = conditioned_base[i][0]*conditioned_attach[i][1];
            M.at<float>(j,8) = conditioned_base[i][0]*conditioned_attach[i][2];

            M.at<float>(j+1,0) = 0.f;
            M.at<float>(j+1,1) = 0.f;
            M.at<float>(j+1,2) = 0.f;
            M.at<float>(j+1,3) = -conditioned_base[i][2]*conditioned_attach[i][0];
            M.at<float>(j+1,4) = -conditioned_base[i][2]*conditioned_attach[i][1];
            M.at<float>(j+1,5) = -conditioned_base[i][2]*conditioned_attach[i][2];
            M.at<float>(j+1,6) = conditioned_base[i][1]*conditioned_attach[i][0];
            M.at<float>(j+1,7) = conditioned_base[i][1]*conditioned_attach[i][1];
            M.at<float>(j+1,8) = conditioned_base[i][1]*conditioned_attach[i][2];
            j+=1;
            }

return cv::Mat_<float>(M);
}
/**
 * @brief solve homogeneous equation system by usage of SVD
 * @param A the design matrix
 * @returns solution of the homogeneous equation system
 */
cv::Matx33f solve_dlt_homography2D(const cv::Mat_<float> &A)
{
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Mat Vt = -svd.vt.row(8);


    return cv::Matx33f(Vt.at<float>(0, 0), Vt.at<float>(0, 1), Vt.at<float>(0, 2),
                           Vt.at<float>(0, 3), Vt.at<float>(0, 4), Vt.at<float>(0, 5),
                           Vt.at<float>(0, 6), Vt.at<float>(0, 7), Vt.at<float>(0, 8));
}


/**
 * @brief decondition a homography that was estimated from conditioned point clouds
 * @param T_base conditioning matrix T' of first set of points x'
 * @param T_attach conditioning matrix T of second set of points x
 * @param H conditioned homography that has to be un-conditioned (in-place)
 */
cv::Matx33f decondition_homography2D(const cv::Matx33f &T_base, const cv::Matx33f &T_attach, const cv::Matx33f &H)
{

    return T_base.inv() * H * T_attach;
}


/**
 * @brief compute the homography
 * @param base first set of points x'
 * @param attach second set of points x
 * @returns homography H, so that x' = Hx
 */

cv::Matx33f homography2D(const std::vector<cv::Vec3f> &base, const std::vector<cv::Vec3f> &attach)
{
        cv::Matx33f base_Cond = getCondition2D(base);
        cv::Matx33f attach_Cond = getCondition2D(attach);
        std::vector<cv::Vec3f> applyH_2D_base = applyH_2D(base, base_Cond, GEOM_TYPE_POINT);
        std::vector<cv::Vec3f> applyH_2D_Attach = applyH_2D(attach, attach_Cond, GEOM_TYPE_POINT);
        cv::Mat_<float> design_Mat_homog2D = getDesignMatrix_homography2D(applyH_2D_base, applyH_2D_Attach);
        cv::Matx33f homog_cond_H = solve_dlt_homography2D(design_Mat_homog2D);
        cv::Matx33f decond_H = decondition_homography2D(base_Cond, attach_Cond, homog_cond_H);

return decond_H ;
}


// Functions from exercise 1
// Reuse your solutions from the last exercise here

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

    /******* Small std::vector cheat sheet ************************************/
    /*
     *   Number of elements in vector:                 a.size()
     *   Access i-th element (reading or writing):     a[i]
     *   Resize array:                                 a.resize(count);
     *   Append an element to an array:                a.push_back(element);
     *     \-> preallocate memory for e.g. push_back:  a.reserve(count);
     */
    /**************************************************************************/

        for (int i=0; i < geomObjects.size(); i++)
        {
            switch (type)
            {
            case GEOM_TYPE_POINT:
            {
                result.resize(i);
                result.reserve(i);
                result.push_back(H*geomObjects[i]);
            }
            break;
            case GEOM_TYPE_LINE:
            {
                result.resize(i);
                result.reserve(i);
                result.push_back(H.inv().t() * geomObjects[i]);
            }
            break;
            default:
                throw std::runtime_error("Unhandled geometry type!");
            }
        }
        return result;
}


/**
 * @brief Convert a 2D point from Euclidean to homogeneous coordinates
 * @param p The point to convert (in Euclidean coordinates)
 * @returns The same point in homogeneous coordinates
 */
cv::Vec3f eucl2hom_point_2D(const cv::Vec2f& p)
{

    return cv::Vec3f(p[0], p[1], 1);
}

}
