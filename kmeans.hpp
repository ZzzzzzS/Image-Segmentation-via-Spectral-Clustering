#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <random>

/**
 * @brief image sigmentation with kmeans method
 * 
 */
class kmeans
{
public:
/**
 * @brief Construct a new kmeans object
 * 
 * @param k k classes
 * @param max_iter max iteration for kmeans
 * @param attempts Flag to specify the number of times the algorithm is executed using different initial labellings.
 */
	kmeans(int k, int max_iter, int attempts)
		:k(k),max_iter(max_iter),attempts(attempts)
	{

	}
	/**
	 * @brief Destroy the kmeans object
	 * 
	 */
	~kmeans() {}

	/**
	 * @brief operator() perform kmeans sigmentation
	 * 
	 * @param src image input
	 * @param mask sigmentation output
	 */
	void operator()(const cv::Mat& src, cv::Mat& mask)
	{
		std::vector<cv::Point3d> centers;
		cv::Mat dst;
		this->KmeansOnce(src, dst, centers); //perform kmeans once

		//remap color for better visualization
		std::vector<int> colors(k);
		int setp=255/k;
		for (size_t i = 0; i < k; i++)
		{
			colors[i] = i * setp + setp / 2;
		}

		this->ColorRemap(dst, mask, colors);
	}

private:
	int k;
	int max_iter;
	int attempts;

private:
	/**
	 * @brief perform kmeans once
	 * 
	 * @param src image input
	 * @param dst segmentation output
	 * @param centers clustering centers
	 */
	void KmeansOnce(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point3d>& centers)
	{
		cv::Mat LocalLabels, BestLabels; //labels of each pixel
		std::vector<cv::Point3d> LocalCenters, BestCenters; //clustering centers
		double LocalDistance, BestDistance;
		//perform kmeans multiple times and choose the best result
		for (size_t i = 0; i < attempts; i++)
		{
			iter(src, LocalLabels, LocalCenters); //perform kmeans once
			LocalDistance = ComputeDistance(src, LocalLabels, LocalCenters); //compute distance
			if (i == 0)
			{
				BestLabels = LocalLabels;
				BestCenters = LocalCenters;
				BestDistance = LocalDistance;
			}
			else
			{
				if (LocalDistance < BestDistance)
				{
					BestLabels = LocalLabels;
					BestCenters = LocalCenters;
					BestDistance = LocalDistance;
				}
			}
		}
		dst = BestLabels;
		centers = BestCenters;
	}

	/**
	 * @brief perform kmeans once
	 * 
	 * @param src image input
	 * @param dst segmentation output
	 * @param centers clustering centers
	 */
	void iter(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Point3d>& centers)
	{
		//generate random centers in uniform distribution
		centers.resize(k);
		std::random_device rd;  // get random device
		std::mt19937 gen(rd()); // use mt19937
		std::uniform_int_distribution<> dis(0, 255); 
		for (int i = 0; i < k; i++)
		{
			centers[i].x = dis(gen);
			centers[i].y = dis(gen);
			centers[i].z = dis(gen);
		}

		//iterate
		size_t iter_=0;
		dst = cv::Mat::zeros(src.size(), CV_8UC1);
		while (iter_<max_iter)
		{
			//update labels
			for (size_t i = 0; i < src.rows; i++)
			{
				for (size_t j = 0; j < src.cols; j++)
				{
					cv::Point3d p;
					p.x = src.at<cv::Vec3b>(i, j)[0];
					p.y = src.at<cv::Vec3b>(i, j)[1];
					p.z = src.at<cv::Vec3b>(i, j)[2];
					double min_dist = 100000000;
					int min_idx = -1;
					for (size_t k = 0; k < centers.size(); k++)
					{
						double dist = cv::norm(p - centers[k]);
						if (dist < min_dist)
						{
							min_dist = dist;
							min_idx = k;
						}
					}
					dst.at<uchar>(i, j) = min_idx;
				}
			}

			//update centers
			std::vector<cv::Point3d> counts(k, {0,0,0});
			std::vector<int> labelsize(k,0);
			for (size_t i = 0; i < src.rows; i++)
			{
				for (size_t j = 0; j < src.cols; j++)
				{
					int idx = dst.at<uchar>(i, j);
					labelsize[idx]++;
					counts[idx].x += src.at<cv::Vec3b>(i, j)[0];
					counts[idx].y += src.at<cv::Vec3b>(i, j)[1];
					counts[idx].z += src.at<cv::Vec3b>(i, j)[2];
				}
			}
			for (size_t i = 0; i < k; i++)
			{
				centers[i]=counts[i]/labelsize[i];
			}

			iter_++;
		}
	}


	/**
	 * @brief compute distance between each pixel and its center
	 * 
	 * @details The function returns the compactness measure that is computed as
	 	distance=\sum _i \| \texttt{samples} _i - \texttt{centers} _{ \texttt{labels} _i} \| ^2
	 	after every attempt. The best (minimum) value is chosen and the corresponding 
	 	labels and the compactness value are returned by the function.

	 * @param src image input
	 * @param labels labels input
	 * @param centers clustering centers input
	 * @return double sum distance
	 */
	double ComputeDistance(const cv::Mat& src, const cv::Mat& labels, const std::vector<cv::Point3d>& centers)
	{
		double Distance = 0;
		for (size_t i = 0; i < src.rows; i++)
		{
			for (size_t j = 0; j < src.cols; j++)
			{
				int idx = labels.at<uchar>(i, j);
				cv::Point3d p;
				p.x = src.at<cv::Vec3b>(i, j)[0];
				p.y = src.at<cv::Vec3b>(i, j)[1];
				p.z = src.at<cv::Vec3b>(i, j)[2];
				Distance += cv::norm(p - centers[idx]); //compute distance for each pixel
			}
		}
		return Distance;
	}

	/**
	 * @brief remap color
	 * 
	 * @param src segmentation output without color remapping
	 * @param dst result with color remapping
	 * @param colors color lists
	 */
	void ColorRemap(const cv::Mat& src, cv::Mat& dst, std::vector<int>& colors)
	{
		dst= cv::Mat::zeros(src.size(), CV_8UC1);
		for (size_t i = 0; i < src.rows; i++)
		{
			for (size_t j = 0; j < src.cols; j++)
			{
				int idx = src.at<uchar>(i, j);
				dst.at<uchar>(i, j) = colors[idx];
			}
		}
	}

};