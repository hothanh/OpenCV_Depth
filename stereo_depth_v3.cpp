#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>
#include <stdio.h>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

void GetDepth(Mat *imgL, Mat *imR, Mat *depth_map, Mat *filtered_disp_vis)
{
	Ptr<StereoSGBM> sgbm_left = StereoSGBM::create(0, 16, 3);

	sgbm_left->setPreFilterCap(61);
	sgbm_left->setBlockSize(3);
	sgbm_left->setP1(8 * 1*3*3);
	sgbm_left->setP2(32 * 1*3*3);
	sgbm_left->setMinDisparity(0);
	sgbm_left->setNumDisparities(16);
	sgbm_left->setUniquenessRatio(10);
	sgbm_left->setSpeckleWindowSize(100);
	sgbm_left->setSpeckleRange(32);
	sgbm_left->setDisp12MaxDiff(1);
	sgbm_left->setMode(StereoSGBM::MODE_SGBM);
	Ptr<StereoMatcher> right_matcher = createRightMatcher(sgbm_left);
	
	Mat disp, right_disp, disp8;
	Mat filtered_disp;
	
	sgbm_left->compute(*imgL, *imgR, *disp);
	right_matcher->compute(*imgR, *imgL, right_disp);
	Ptr<DisparityWLSFilter> wls_filter;
	wls_filter = createDisparityWLSFilter(sgbm_left);
	
	//! [filtering]
	double lambda = 8000.0;
	double sigma = 1.5;
	
	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	wls_filter->filter(disp, *imgL, filtered_disp, right_disp);
	//Visualize disparity map
	getDisparityVis(filtered_disp, *filtered_disp_vis, 10.0);
	
	Mat imgDisparity32F;
	filtered_disp.convertTo(imgDisparity32F, CV_32F, 1. / 16);
	
	depth_map->setTo(0);

        cv::Mat Q_32F;
        Q.convertTo(Q_32F, CV_32F);

        float inv_baseline = Q_32F.at<float>(3, 2); // this valye is 1/baseline_length
        float focal_length = Q_32F.at<float>(2, 3); // this if camera focal length

        float min_ = 1.0e8;
        float max_ = -1.0e8;

        // calculate depth from each disparity value 
        for (int y = 0; y < imgDisparity32F.rows; y++)
        {
            for (int x = 0; x < imgDisparity32F.cols; x++)
            {
                if (imgDisparity32F.at<float>(y, x) < 0.5) continue;
                float depth = focal_length / (inv_baseline * imgDisparity32F.at<float>(y, x));
                depth_map->at<float>(y, x) = depth;

     		if (depth > max_) max_ = depth;
        	if (depth < min_) min_ = depth;
      	    }
  	}

}
