#include <iostream>
#include <opencv2/opencv.hpp>
#include "io.hpp"
#include "kmeans.hpp"
#include "Ncut.hpp"

int main(int, char**) {
	std::cout << "Hello, world!\n";

	//read files
	io io;
	std::vector<std::string> path;
	std::vector<int> label;
	//download from https://www.cs.toronto.edu/~kriz/cifar.html
	path.push_back("./cifar-10-batches-bin/data_batch_1.bin");
	path.push_back("./cifar-10-batches-bin/data_batch_2.bin");
	path.push_back("./cifar-10-batches-bin/data_batch_3.bin");
	path.push_back("./cifar-10-batches-bin/data_batch_4.bin");
	path.push_back("./cifar-10-batches-bin/data_batch_5.bin");
	label.push_back(0);//airplane
	label.push_back(4);//deer
	label.push_back(7);//horse
	io.read_cifar10_bin(path, label);

	//show image
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	int i = 1;
	int j = 4;
	cv::imshow("image", io.Images[j][i]);

	//perform kmeans
	kmeans kmeans_t = kmeans(4, 100, 10);
	cv::Mat mask;
	kmeans_t(io.Images[j][i], mask);
	cv::namedWindow("mask", cv::WINDOW_NORMAL);
	cv::imshow("mask", mask);

	//perform Normalized Cuts
	Ncut ncut = Ncut(5, 0.01, 4.0);
	cv::Mat mask2;
	ncut(io.Images[j][i], mask2);
	cv::namedWindow("mask2", cv::WINDOW_NORMAL);
	cv::imshow("mask2", mask2);
	cv::waitKey(0);

	return 0;
}
