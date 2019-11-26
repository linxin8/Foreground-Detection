#define _CRT_SECURE_NO_WARNINGS
#include <iostream>  
#include <algorithm>
#include <opencv2/opencv.hpp> 
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<string>

using namespace cv;

const int gaussianSize = 5;
const float backgroundThreshold = 0.7f;
float alpha = 0.002f;
const float D = 3;
const float standardOrginal = 15;
Mat result;
struct Gaussian
{
	Mat weight;
	Mat mean;
	Mat standard; 
};
struct GaussianList
{
	Gaussian gaussianTable[gaussianSize];
	Gaussian* begin() { return gaussianTable; }
	Gaussian* end() { return gaussianTable + gaussianSize; }
	Gaussian& operator[](int i) { return gaussianTable[i]; }
};
GaussianList gaussianList;
Mat backgroud;

void init(const Mat& frame)
{
	for (auto&g : gaussianList)
	{
		g.weight = Mat::zeros(frame.size(), CV_32FC1);
		g.mean = Mat::zeros(frame.size(), CV_8UC1); 
		g.standard = Mat::zeros(frame.size(), CV_32FC1);
		g.standard.setTo(standardOrginal);
	}
	gaussianList[0].weight.setTo(1.0);
	frame.copyTo(gaussianList[0].mean);
	backgroud = Mat::zeros(frame.size(), CV_8UC1);
	result = Mat::zeros(frame.size(), CV_8UC1);
} 

#define ALIAS_MARCO									\
auto& frameData = frame.at<uchar>(i, j);			\
auto& meanData = g.mean.at<uchar>(i, j);			\
auto& weightData = g.weight.at<float>(i, j);		\
auto& standardData = g.standard.at<float>(i, j);	 


void train(Mat& frame)
{
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			auto hasFindGaussian = false;
			for (auto&g : gaussianList)
			{
				ALIAS_MARCO;
				auto off = std::abs(frameData - meanData);
				if (off <= D * standardData)
				{
					weightData = (1 - alpha) * weightData + alpha;
					auto p = alpha / weightData;
					meanData = (uchar)((1 - p) * meanData + p * frameData);
					off = std::abs(frameData - meanData);
					standardData = std::sqrt((1 - p) * standardData * standardData + p * off * off);
					hasFindGaussian = true;
				}
				else
				{
					weightData = (1 - alpha)* weightData;
				}
			}
			if (!hasFindGaussian)
			{
				auto gp = std::find_if(gaussianList.begin(), gaussianList.end(), [&](Gaussian& g) {ALIAS_MARCO; return weightData == 0; });
				if (gp == gaussianList.end()) gp = &gaussianList[gaussianSize - 1];
				auto& g = *gp;
				ALIAS_MARCO;
				meanData = frameData;
				standardData = standardOrginal;
				weightData = 0.05f;
			}
			float totalWeight = 0;
			for (auto&g : gaussianList)
			{
				ALIAS_MARCO;
				totalWeight += weightData;
			}
			for (auto&g : gaussianList)
			{
				ALIAS_MARCO;
				weightData /= totalWeight;
			}
			for (int m = 0; m < gaussianSize; m++)
			{
				for (int n = m + 1; n < gaussianSize; n++)
				{
					auto rankLeft = gaussianList[m].weight.at<float>(i, j) / gaussianList[m].standard.at<float>(i, j);
					auto rankRight = gaussianList[n].weight.at<float>(i, j) / gaussianList[n].standard.at<float>(i, j);
					if (rankLeft < rankRight)
					{
						std::swap(gaussianList[m].mean.at<uchar>(i, j), gaussianList[n].mean.at<uchar>(i, j));
						std::swap(gaussianList[m].weight.at<float>(i, j), gaussianList[n].weight.at<float>(i, j));
						std::swap(gaussianList[m].standard.at<float>(i, j), gaussianList[n].standard.at<float>(i, j));
					}
				}
			}
		}
	}
}

void updateBackground()
{
	for (int i = 0; i < gaussianList[0].weight.rows; i++)
	{
		for (int j = 0; j < gaussianList[0].weight.cols; j++)
		{
			float sum = 0;
			for (int k = 0; k < gaussianSize; k++)
			{
				sum += gaussianList[k].weight.at<float>(i, j);
				if (sum >= backgroundThreshold)
				{
					backgroud.at<uchar>(i, j) = k + 1;
					break;
				}
			}
		}
	}
}
 
void test(Mat& frame)
{
	for (int i = 0; i < frame.rows; ++i)
	{
		for (int j = 0; j < frame.cols; ++j)
		{
			bool hasFind = false;
			for (int k = 0; k < backgroud.at<uchar>(i, j); k++)
			{
				auto& g = gaussianList[k];
				ALIAS_MARCO;
				auto off = std::abs(frameData - meanData);
				if (off < D*standardData)
				{
					result.at<uchar>(i, j) = 0;
					hasFind = true;
					break;
				}
			}
			if (!hasFind)
			{
				result.at<uchar>(i, j) = 255; 
			}
		}
	} 
}

Mat readImage(int index)
{
	char str[30];
	std::sprintf(str, "image/b%05d.bmp", index);
	return imread(str, 0);
}

int main()
{
	init(readImage(0)); 
	for (int i = 0; i <= 200; i++)
	{
		if (i < 100)
		{
			alpha = 0.05f;
		}
		else
		{ 
			alpha = 0.005f;
		}
		auto image = readImage(i);
		train(image);
		std::cout << "train " << i << std::endl;
	}
	updateBackground();  
	Mat window = Mat::zeros(result.rows, result.cols * 2, result.type());
	Rect left(0, 0, result.cols, result.rows);
	Rect right(0, 0, result.cols, result.rows); 
	right.x = result.cols; 
	namedWindow("window", WINDOW_FREERATIO); 
	VideoWriter writer("ouput.mp4", VideoWriter::fourcc('M', 'P', '4', '2'),3.0, window.size());
	for (int i = 0; i <= 286; i++)
	{
		auto image = readImage(i); 
		test(image);
		image.copyTo(window(left));			
		erode(result, result, cv::Mat()); 
		dilate(result, result, cv::Mat()); 
		dilate(result, result, cv::Mat());
		result.copyTo(window(right));
		imshow("window", window);  
		writer.write(window);
		waitKey(50);
		std::cout << "display " << i << std::endl;
	}
	return 0;
}
