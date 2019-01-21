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
#include <ctype.h>
#include <time.h>
#include <sys/time.h>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;


static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match [--imagemode=image|folder] [--left=<left_image>] [--right=<right_image>] [--algorithm=sgbm|sgbm3way] [--sad-win-size=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[-o=<disparity_image>] [-od=<depth_data(.matbin)>]\n"
		"[--pre-filter-cap=<prefiltercap>] [--min-disparity=<mininum_disparity] [--P1=<P1>] [--P2=<P2>] [--uniqe-ratio=<unique_ratio]\n"
		"[--speckle-range=<speckle_range>] [--speckle-win-size=<speckle_window_size] [--disp-12max-diff=<disparity_12max_diff>]\n"
		"[--timing=yes|no] [--matbin-dir=<matbin_directory>]");
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


Mat buildColorMap(Mat map) 
{
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    max /= 8;

    cv::Mat adjMap;
    // expand your range to 0..255. Similar to histEq();
    map.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);

    // this is great. It converts your grayscale image into a tone-mapped one, 
    // much more pleasing for the eye, but the values are normalized and are false
    // so we name it falseColorMap
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);

    return falseColorsMap;
}
   
int main(int argc, char** argv)
{
	std::string loc="";
	std::string img1_filename = "";
	std::string img2_filename = "";
	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string disparity_filename = "disparity.png";
	std::string depth_filename = "depth.matbin";
	int start_index = 0;
	int end_index = 52000;
	bool show_time = false;
	int sgbm_mode=0;
	double scale_factor=1;
	double lambda=8000.0;
	double sigma = 1.5;
	std::string matbin_dir="./";

	enum { STEREO_SGBM, STEREO_3WAY };

	int alg = STEREO_SGBM;
	int SADWindowSize, numberOfDisparities;
	int PreFilterCap, MininumDisparity, P1, P2, UniqeRatio, SpeckleRange, SpeckleWinSize, Disp12MaxDiff;

	cv::CommandLineParser parser(argc, argv, "{imagemode||}{left||}{right||}{help h||}{algorithm||}{max-disparity|0|}{sad-win-size|0|}{scale|1|}{i||}{e||}{o||}{od||}{loc||}{pre-filter-cap|61|}{min-disparity|0|}{P1|1100|}{P2|1100|}{uniqe-ratio|10|}{speckle-range|32|}{speckle-win-size|100|}{disp-12max-diff|1|}{timing||}{matbin-dir||}{sgbm-mode|0|}{scale-factor|1|}{start-inx|0|}{end-inx|52000|}{lambda|8000|}{sigma|1.5|}");

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
	SADWindowSize = parser.get<int>("sad-win-size");
	PreFilterCap	= parser.get<int>("pre-filter-cap");
	MininumDisparity = parser.get<int>("min-disparity");
	P1 = parser.get<int>("P1");
	P2 = parser.get<int>("P2");
	scale_factor = parser.get<double>("scale-factor");
	lambda = parser.get<double>("lambda");
	sigma = parser.get<double>("sigma");
	sgbm_mode = parser.get<int>("sgbm-mode");
	UniqeRatio = parser.get<int>("uniqe-ratio");
	SpeckleRange = parser.get<int>("speckle-range");
	SpeckleWinSize = parser.get<int>("speckle-win-size");
	Disp12MaxDiff = parser.get<int>("disp-12max-diff");
	start_index = parser.get<int>("start-inx");
	end_index = parser.get<int>("end-inx");
	show_time = (parser.get<std::string>("timing") == "yes") ? true : false;
	matbin_dir = parser.get<std::string>("matbin-dir");
	
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
		printf("Command-line parameter error: The block size (--sad-win-size=<...>) must be a positive odd number\n");
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

	struct timeval t1, t2, t3, t4;
	double t_process;
	int frame_count=0;
	for(int i = start_index; i <= end_index; i++)
	{
		if (img_mode == 1) 
		{
			stringstream strleft, strright;
			strleft << loc << "left_image_" << i << ".png";
			strright << loc << "right_image_" << i << ".png";

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

		int cn = img1.channels();

		if (show_time) gettimeofday(&t1, NULL);

		Ptr<StereoSGBM> sgbm_left = cv::StereoSGBM::create(0, 160, 13, 4056, 16224, 1, 0, 15, 400, 200, 1);
	
		sgbm_left->setPreFilterCap(PreFilterCap);
		int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
		sgbm_left->setBlockSize(sgbmWinSize);
		sgbm_left->setP1(P1);
		sgbm_left->setP2(P2);
		sgbm_left->setMinDisparity(MininumDisparity);
		sgbm_left->setNumDisparities(numberOfDisparities);
		sgbm_left->setUniquenessRatio(UniqeRatio);
		sgbm_left->setSpeckleWindowSize(SpeckleWinSize);
		sgbm_left->setSpeckleRange(SpeckleRange);
		sgbm_left->setDisp12MaxDiff(Disp12MaxDiff);

		//scale down input to speed up process
		Mat LScaleImage, RScaleImage;
		resize(img1, LScaleImage, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
		resize(img2, RScaleImage, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
		
		//img1 = LScaleImage;
		//img2 = RScaleImage;

		//if (alg == STEREO_SGBM)
		//	sgbm_left->setMode(StereoSGBM::MODE_SGBM);
		//else if (alg == STEREO_3WAY)
		//	sgbm_left->setMode(StereoSGBM::MODE_SGBM_3WAY);
		sgbm_left->setMode(sgbm_mode);

		Ptr<StereoMatcher> right_matcher = createRightMatcher(sgbm_left);

		Mat disp, right_disp, disp8;
		Mat filtered_disp;

		//int64 t = getTickCount();

		sgbm_left->compute(LScaleImage, RScaleImage, disp);
		right_matcher->compute(RScaleImage, LScaleImage, right_disp);

		//t = getTickCount() - t;
		//printf("SGMB processing time: %fms\n\n", t * 1000 / getTickFrequency());


		Ptr<DisparityWLSFilter> wls_filter;
		wls_filter = createDisparityWLSFilter(sgbm_left);
	
		//! [filtering]

		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		wls_filter->filter(disp, LScaleImage, filtered_disp, right_disp);

		//Deepak
		//Visualize disparity map
		Mat filtered_disp_vis;
		getDisparityVis(filtered_disp, filtered_disp_vis, 10.0);
	
		if (show_time)
		{
			gettimeofday(&t2, NULL);
			t_process = (t2.tv_sec - t1.tv_sec) * 1000.0;
			t_process += (t2.tv_usec - t1.tv_usec) / 1000.0;
			cout << "T_disparity_calculation: " << t_process  << "ms"   << endl;
		}
		Mat imgDisparity32F;
		disp.convertTo(imgDisparity32F, CV_32F, 1. / 16);

        	cv::Mat depth_map(imgDisparity32F.rows, imgDisparity32F.cols, CV_32F);
        	depth_map.setTo(0);

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
        	        // depth = (focal_length * baseline) / disparity
        	        float depth = focal_length / (inv_baseline * imgDisparity32F.at<float>(y, x));
        	        depth_map.at<float>(y, x) = depth;

        	        if (depth > max_) max_ = depth;
        	        if (depth < min_) min_ = depth;
        	    }
        	}

		if (show_time)
		{
			gettimeofday(&t3, NULL);
			t_process = (t3.tv_sec - t2.tv_sec) * 1000.0;
			t_process += (t3.tv_usec - t2.tv_usec) / 1000.0;
			cout << "T_depth_calculation: " << t_process  << "ms"   << endl;
		}

		if (img_mode == 0) {
			serializeMatbin(depth_map, depth_filename);
		}
		else
		{
			stringstream str1;
			str1 << matbin_dir << "depth_image_" << i << ".matbin";
			depth_filename = str1.str();
			serializeMatbin(depth_map, depth_filename);	
		}
		if (show_time)
		{
			gettimeofday(&t4, NULL);
			t_process = (t4.tv_sec - t3.tv_sec) * 1000.0;
			t_process += (t4.tv_usec - t3.tv_usec) / 1000.0;
			cout << "T_write_matbin: " << t_process  << "ms"   << endl << endl;
		}

        	// << ****** just checking if depth map saved correctly and loaded as it is ****** >>
        	//Mat imgDepth32f = deserializeMatbin("depth.matbin");
        	//imgDepth32f.convertTo(img_depth8u, CV_16S);
        	// << ****** just checking if depth map saved correctly and loaded as it is ****** >>

        	//! [display & save depth color map for result verification] 
        	//Mat depth_color_map = buildColorMap(depth_map);
        	//stringstream strcolor;
        	//strcolor << loc << "../result/depth_color_map/depth_color_map" << i << ".png";
        	//imwrite(strcolor.str(), depth_color_map);
		if (img_mode == 0) {
			imwrite(disparity_filename, filtered_disp_vis);
			//break;
		}
		else {
			stringstream str1;
			str1 << loc << "../result/dispa_image/disparity_image_" << i << ".png";
			disparity_filename = str1.str();
			imwrite(disparity_filename, disp);			
		}
		
		frame_count++;
	}

	//printf("press any key to continue...");
	//fflush(stdout);
	//waitKey();
	//printf("\n");

	return 0;
}
