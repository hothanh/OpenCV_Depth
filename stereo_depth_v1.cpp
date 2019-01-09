// stereo_depth.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"

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

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match [--imagemode=image|folder] [--left=<left_image>] [--right=<right_image>] [--algorithm=sgbm|sgbm3way] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[-o=<disparity_image>] [-od=<depth_data(.matbin)>]\n");
}

void serializeMatbin(cv::Mat& mat, std::string filename)
{
	if (!mat.isContinuous()) {
		std::cout << "Not implemented yet" << std::endl;
		exit(1);
	}

	FILE* FP = fopen(filename.c_str(), "wb");

	int elemSizeInBytes = (int)mat.elemSize();
	int elemType = (int)mat.type();
	int dataSize = (int)(mat.cols * mat.rows * mat.elemSize());

	int sizeImg[4] = { mat.cols, mat.rows, elemSizeInBytes, elemType };
	fwrite(/* buffer */ sizeImg, /* how many elements */ 4, /* size of each element */ sizeof(int), /* file */ FP);
	fwrite(mat.data, mat.cols * mat.rows, elemSizeInBytes, FP);
	fclose(FP);
}

cv::Mat deserializeMatbin(std::string filename)
{
	FILE* fp = fopen(filename.c_str(), "rb");

	int header[4];
	fread(header, sizeof(int), 4, fp);
	int cols = header[0];
	int rows = header[1];
	int elemSizeInBytes = header[2];
	int elemType = header[3];

	cv::Mat outputMat = cv::Mat::ones(rows, cols, elemType);
	size_t result = fread(outputMat.data, elemSizeInBytes, (size_t)(cols * rows), fp);

	if (result != (size_t)(cols * rows)) {
		fputs("Reading error", stderr);
	}

	fclose(fp);
	return outputMat;
}


int main(int argc, char** argv)
{
	std::string loc="";
	std::string img1_filename = "";
	std::string img2_filename = "";
	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string disparity_filename = "disparity.tif";
	std::string depth_filename = "depth.matbin";

	enum { STEREO_SGBM, STEREO_3WAY };

	int alg = STEREO_SGBM;
	int SADWindowSize, numberOfDisparities;

	cv::CommandLineParser parser(argc, argv, "{imagemode||}{left||}{right||}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{scale|1|}{i||}{e||}{o||}{od||}{loc||}");

	if (parser.has("help"))
	{
		print_help();
		return 0;
	}

	if (parser.has("loc"))
	{	loc = parser.get<std::string>("loc");
	}
	int img_mode = 0;
	if (parser.has("imagemode"))
	{
		std::string process_mode = parser.get<std::string>("imagemode");
		img_mode = process_mode == "image" ? 0 : 1;
	}

	if(img_mode != 1)
	{
		img1_filename = parser.get<std::string>("left");
		img2_filename = parser.get<std::string>("right");
		cout << img1_filename << endl;
		cout << img2_filename << endl;
	}
	
	if (parser.has("algorithm"))
	{
		std::string _alg = parser.get<std::string>("algorithm");
		alg = _alg == "sgbm" ? STEREO_SGBM :
			_alg == "sgbm3way" ? STEREO_3WAY : -1;
	}
	numberOfDisparities = parser.get<int>("max-disparity");
	SADWindowSize = parser.get<int>("blocksize");
	
	if (parser.has("i"))
		intrinsic_filename = parser.get<std::string>("i");
	if (parser.has("e"))
		extrinsic_filename = parser.get<std::string>("e");
	if (parser.has("o"))
		disparity_filename = parser.get<std::string>("o");
	if (parser.has("od"))
		depth_filename = parser.get<std::string>("od");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	if (alg < 0)
	{
		printf("Command-line parameter error: Unknown stereo algorithm\n\n");
		print_help();
		return -1;
	}
	if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
	{
		printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
		print_help();
		return -1;
	}
	if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
	{
		printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
		return -1;
	}
	if ((img1_filename.empty() || img2_filename.empty()) && img_mode==0)
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}
	if ((!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()))
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		return -1;
	}

	int color_mode = -1;

	Mat img1;
	Mat img2;
	Size img_size;

	if(img_mode == 0)
	{
		img1 = imread(img1_filename, color_mode);
		img2 = imread(img2_filename, color_mode);

		img_size = img1.size();

		if (img1.empty())
		{
			printf("Command-line parameter error: could not load the first input image file\n");
			return -1;
		}
		if (img2.empty())
		{
			printf("Command-line parameter error: could not load the second input image file\n");
			return -1;
		}
	}
	else {
		img1_filename = "img/left_image_338.png";
		img2_filename = "img/right_image_338.png";
		img1 = imread(img1_filename, color_mode);
		img2 = imread(img2_filename, color_mode);

		img_size = img1.size();
	}

	Rect roi1, roi2;
	Mat Q;

	if (!intrinsic_filename.empty())
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		fs.open(extrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename.c_str());
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
	}
	
	// for processing all images at once, i am here using static values. 
	// if you want to process other stereo images, you have to match file names and index etc. 

	for(int i = 1; i < 5000; i++)
	{
		if (img_mode == 1) 
		{
			stringstream strleft, strright;
			strleft << loc << "left_image" << i << ".png";
			strright << loc << "right_image" << i << ".png";

			img1_filename = strleft.str();
			img2_filename = strright.str();

			img1 = imread(img1_filename, color_mode);
			img2 = imread(img2_filename, color_mode);

			if (img1.empty() || img2.empty()) 
			{
			
				cout << "File " << img1_filename << " Not Found " << endl;
				continue;
			}

		}

		numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;
		//numberOfDisparities = 64;

		cout << numberOfDisparities << endl;

		int cn = img1.channels();

		Ptr<StereoSGBM> sgbm_left = StereoSGBM::create(0, 16, 3);
	
		sgbm_left->setPreFilterCap(61);
		int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
		sgbm_left->setBlockSize(sgbmWinSize);
		sgbm_left->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
		sgbm_left->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
		sgbm_left->setMinDisparity(0);
		sgbm_left->setNumDisparities(numberOfDisparities);
		sgbm_left->setUniquenessRatio(10);
		sgbm_left->setSpeckleWindowSize(100);
		sgbm_left->setSpeckleRange(32);
		sgbm_left->setDisp12MaxDiff(1);

		//scale down input to speed up process
		Mat LScaleImage, RScaleImage;
		resize(img1, LScaleImage, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
		resize(img2, RScaleImage, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
		
		img1 = LScaleImage;
		img2 = RScaleImage;

		if (alg == STEREO_SGBM)
			sgbm_left->setMode(StereoSGBM::MODE_SGBM);
		else if (alg == STEREO_3WAY)
			sgbm_left->setMode(StereoSGBM::MODE_SGBM_3WAY);

		Ptr<StereoMatcher> right_matcher = createRightMatcher(sgbm_left);

		Mat disp, right_disp, disp8;
		Mat filtered_disp;

		int64 t = getTickCount();

		sgbm_left->compute(img1, img2, disp);
		right_matcher->compute(img2, img1, right_disp);

		t = getTickCount() - t;
		printf("SGMB processing time: %fms\n\n", t * 1000 / getTickFrequency());

		double matching_time, filtering_time;

		Ptr<DisparityWLSFilter> wls_filter;
		wls_filter = createDisparityWLSFilter(sgbm_left);
	
		//! [filtering]
		double lambda = 8000.0;
		double sigma = 1.5;

		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		//filtering_time = (double)getTickCount();
		wls_filter->filter(disp, img1, filtered_disp, right_disp);
		//filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
		//! [filtering]

		//Mat conf_map = Mat(img1.rows, img1.cols, CV_8U);
		//conf_map = Scalar(255);
		//Deepak
		//Visualize disparity map
		Mat filtered_disp_vis;
		getDisparityVis(filtered_disp, filtered_disp_vis, 10.0);
	
		Mat imgDisparity32F;
		filtered_disp.convertTo(imgDisparity32F, CV_32F, 1. / 16);
       		// convert to 3d depth map
       		Mat map_3d;
       		reprojectImageTo3D(imgDisparity32F, map_3d, Q, true);

       		const double max_z = 1.0e4;
       		imgDisparity32F.setTo(0);
		t = getTickCount();
       		// filter out noise pixels and get only 2d depth map
        	for (int y = 0; y < map_3d.rows; y++)
        	{
        	    for (int x = 0; x < map_3d.cols; x++)
        	    {
        	        Vec3f point = map_3d.at<Vec3f>(y, x);
        	        // filter out noise pixels 
        	        if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
        	        // we are only interested in depth values
        	        imgDisparity32F.at<float>(y, x) = point[2];
        	    }
        	}


		/*
		cv::Mat_<cv::Vec3f> XYZ(imgDisparity32F.rows, imgDisparity32F.cols);   // Output point cloud
		cv::Mat_<float> vec_tmp(4, 1);
		cv::Mat Q_32F;
		Q.convertTo(Q_32F, CV_32F);
		printf("q32(0,3):%4.2f\n",Q_32F.at<float>(0,3));
		t = getTickCount();

		for (int y = 0; y < imgDisparity32F.rows; y++)
		{
			for (int x = 0; x < imgDisparity32F.cols; x++)
			{
				vec_tmp(0) = x; 
				vec_tmp(1) = y; 
				vec_tmp(2) = imgDisparity32F.at<float>(y, x) <= 0 ? -1 : imgDisparity32F.at<float>(y, x);
				vec_tmp(3) = 1;
			
				vec_tmp = Q_32F*vec_tmp;
				vec_tmp /= vec_tmp(3);
			
				//cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
				//point[0] = vec_tmp(0);
				//point[1] = vec_tmp(1);
				//point[2] = vec_tmp(2);

				imgDisparity32F.at<float>(y, x) = vec_tmp(2);
				//printf("(%d,%d): %4.2f\n",y,x,vec_tmp(2));
			}
		}
		*/
		t = getTickCount() - t;
		printf("depth processing time: %fms\n\n", t * 1000 / getTickFrequency());

		if (img_mode == 0) {
			serializeMatbin(imgDisparity32F, depth_filename);
		}
		else
		{
			stringstream str1;
			str1 << loc << "../result/depth_bin/depth_image" << i << ".matbin";
			depth_filename = str1.str();
			serializeMatbin(imgDisparity32F, depth_filename);
		}

        	Mat img_depth8u;
        	// << ****** just checking if depth map saved correctly and loaded as it is ****** >>
        	//Mat imgDepth32f = deserializeMatbin("depth.matbin");
        	//imgDepth32f.convertTo(img_depth8u, CV_16S);
        	// << ****** just checking if depth map saved correctly and loaded as it is ****** >>

        	//! [display & save depth color map for result verification] 
        	Mat depth_color;
        	cv::normalize(imgDisparity32F, img_depth8u, 255, 0, NORM_MINMAX);
        	img_depth8u.convertTo(img_depth8u,  CV_8U);
        	applyColorMap(img_depth8u, depth_color, COLORMAP_JET);
        	stringstream strcolor;
        	strcolor << loc << "../result/depth_color_map/depth_color_map" << i << ".png";
        	imwrite(strcolor.str(), depth_color);
		
	

//		imshow("left", img1);
//		imshow("right", img2);
//		imshow("depth-map", imgDepth16S);
//		imshow("disparity-map", filtered_disp_vis);
		
		//waitKey(1);
		
		if (img_mode == 0) {
			imwrite(disparity_filename, filtered_disp_vis);
			break;
		}
		else {
			stringstream str1;
			str1 << loc << "../result/dispa_image/dispa" << i << ".png";
			disparity_filename = str1.str();
			imwrite(disparity_filename, filtered_disp_vis);			
		}
	}

	printf("press any key to continue...");
	fflush(stdout);
	//waitKey();
	printf("\n");

	return 0;
}
