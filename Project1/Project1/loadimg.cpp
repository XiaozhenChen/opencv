#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;



int DrawPoly()
{
	Mat image = Mat::zeros(400, 400, CV_8UC3);

	line(image, Point(50, 20), Point(70, 50), Scalar(110, 220, 0), 2, 4);
	circle(image, Point(100, 100), 20, Scalar(110, 220, 0), 2, 8);
	ellipse(image, Point(200, 200), Size(100.0, 160.0), 45, 0, 360, Scalar(255, 0, 0), 1, 8);
	rectangle(image,Point(15,20),Point(70,50),Scalar(0,5,255),2,8,0);
	


	int w = 400;
	// Draw a circle 
	/** Create some points */
	Point rook_points[1][20];
	rook_points[0][0] = Point(w / 4.0, 7 * w / 8.0);
	rook_points[0][1] = Point(3 * w / 4.0, 7 * w / 8.0);
	rook_points[0][2] = Point(3 * w / 4.0, 13 * w / 16.0);
	rook_points[0][3] = Point(11 * w / 16.0, 13 * w / 16.0);
	rook_points[0][4] = Point(19 * w / 32.0, 3 * w / 8.0);
	rook_points[0][5] = Point(3 * w / 4.0, 3 * w / 8.0);
	rook_points[0][6] = Point(3 * w / 4.0, w / 8.0);
	rook_points[0][7] = Point(26 * w / 40.0, w / 8.0);
	rook_points[0][8] = Point(26 * w / 40.0, w / 4.0);
	rook_points[0][9] = Point(22 * w / 40.0, w / 4.0);
	rook_points[0][10] = Point(22 * w / 40.0, w / 8.0);
	rook_points[0][11] = Point(18 * w / 40.0, w / 8.0);
	rook_points[0][12] = Point(18 * w / 40.0, w / 4.0);
	rook_points[0][13] = Point(14 * w / 40.0, w / 4.0);
	rook_points[0][14] = Point(14 * w / 40.0, w / 8.0);
	rook_points[0][15] = Point(w / 4.0, w / 8.0);
	rook_points[0][16] = Point(w / 4.0, 3 * w / 8.0);
	rook_points[0][17] = Point(13 * w / 32.0, 3 * w / 8.0);
	rook_points[0][18] = Point(5 * w / 16.0, 13 * w / 16.0);
	rook_points[0][19] = Point(w / 4.0, 13 * w / 16.0);

	const Point* ppt[1] = { rook_points[0] };
	int npt[] = { 20 };

	fillPoly(image,ppt,npt,1,Scalar(255,255,255),8);

	putText(image, "Hi all...", Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200), 4);
	imshow("Image", image);


	waitKey(0);
	return(0);
}

int LoadImage()
{
	Mat image;
	image = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		cout << "Cannot load image!" << endl;
		return -1;
	}

	

	//imwrite("C:/CXZ/cvTest/Project1/Resource/result.jpg", image);




	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Create a new matrix to hold the gray image
	Mat gray;

	// convert RGB image to gray
	cvtColor(image, gray, CV_BGR2GRAY);

	//namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", image);

	namedWindow("Result window", CV_WINDOW_AUTOSIZE);
	imshow("Result window", gray);


	waitKey(0);
	return 0;
}




/*
int Threshold_main(int argc, char** argv)
{
	int threshold_value = 0;
	int threshold_type = 3;;
	int const max_value = 255;
	int const max_type = 4;
	int const max_BINARY_value = 255;

	Mat src, src_gray, dst;

	char window_name[] = "Threshold Demo";

	char trackbar_type[] = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
	char trackbar_value[] = "Value";
	/// Load an image
	src = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", 1);

	/// Convert the image to Gray
	cvtColor(src, src_gray, CV_RGB2GRAY);

	/// Create a window to display results
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold
	createTrackbar(trackbar_type,
		window_name, &threshold_type,
		max_type, Threshold_Demo);

	createTrackbar(trackbar_value,
		window_name, &threshold_value,
		max_value, Threshold_Demo);

	/// Call the function to initialize
	Threshold_Demo(0, 0);

	/// Wait until user finishes program
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}
}


void Threshold_Demo(int, void*)
{
	 0: Binary
	   1: Binary Inverted
	   2: Threshold Truncated
	   3: Threshold to Zero
	   4: Threshold to Zero Inverted


	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}
	 */


int Filter()
{
	Mat src = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat dst;
	GaussianBlur(src, dst, Size(5, 5), 0, 0);
	
	imshow("source",src );
	imshow("Result", dst);

	waitKey(0);
	return 0;
}


void conv2(Mat src, int kernel_size)
{
	Mat dst, kernel;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	/// Apply filter
	filter2D(src, dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
	namedWindow("filter2D Demo", CV_WINDOW_AUTOSIZE);
	imshow("filter2D Demo", dst);
}

int conv2_main(int argc, char** argv)
{
	Mat src;

	/// Load an image
	src = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg");
	if (!src.data) { return -1; }

	conv2(src, 3);
	conv2(src, 30);

	waitKey(0);
	return 0;
}



int Edge_detect_main(int argc, char** argv)
{
	Mat src, gray, dst, abs_dst;
	src = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", CV_LOAD_IMAGE_COLOR);

	/// Remove noise by blurring with a Gaussian filter
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(src, gray, CV_RGB2GRAY);

	/// Apply Laplace function
	Laplacian(gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	imshow("result", abs_dst);

	waitKey(0);
	return 0;
}



int Hough_Circle_Detection_main()
{
	Mat src, gray;
	src = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", 1); resize(src, src, Size(640, 480));
	cvtColor(src, gray, CV_BGR2GRAY);

	// Reduce the noise so we avoid false circle detection
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;

	// Apply the Hough Transform to find the circles
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 50, 0, 0);

	// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center     
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);// circle outline
		cout << "center : " << center << "\nradius : " << radius << endl;
	}

	// Show your results
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", src);

	waitKey(0);
	return 0;
}

int Histogram_main(int, char**)
{
	Mat gray = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", 0);
	namedWindow("Gray", 1);    imshow("Gray", gray);

	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Show the calculated histogram in command window
	double total;
	total = gray.rows * gray.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		cout << " " << binVal;
	}

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	namedWindow("Result", 1);    imshow("Result", histImage);

	waitKey(0);
	return 0;
}

int Erode_main()
{

	Mat image, dst;
	image = imread("C:/CXZ/cvTest/Project1/Resource/lena.jpg", CV_LOAD_IMAGE_COLOR);

	// Create a structuring element
	int erosion_size = 6;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	// Apply erosion or dilation on the image
	erode(image, dst, element);  // dilate(image,dst,element);

	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	imshow("Display window", image);

	namedWindow("Result window", CV_WINDOW_AUTOSIZE);
	imshow("Result window", dst);

	waitKey(0);
	return 0;
}

int Bitwise_main()
{
	Mat drawing1 = Mat::zeros(Size(400, 200), CV_8UC1);
	Mat drawing2 = Mat::zeros(Size(400, 200), CV_8UC1);
	Mat res;
	drawing1(Range(0, drawing1.rows), Range(0, drawing1.cols / 2)) = 255 ;
	imshow("drawing1", drawing1);

	drawing2(Range(100, 150), Range(150,350)) = 255;
	imshow("drawing2", drawing2);

	bitwise_and(drawing1, drawing2,res); 
	imshow("And", res);

	bitwise_or(drawing1, drawing2, res);      imshow("OR", res);
	bitwise_xor(drawing1, drawing2, res);     imshow("XOR", res);
	bitwise_not(drawing1, res);              imshow("NOT", res);

	//drawing1

		waitKey(0);
	return 0;
}


