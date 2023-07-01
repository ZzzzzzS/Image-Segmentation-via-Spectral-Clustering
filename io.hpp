# pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

/**
 * @brief io class, read and write images
 * 
 */
class io
{

public:
/**
 * @brief Construct a new io object
 * 
 */
    io()
    {
        Images.resize(10);
        SigmentResult.resize(10);
    }
    ~io() {}

    /**
     * @brief Images[Label][ImageIndex] images with different labels
     * 
     */
    std::vector<std::vector<cv::Mat>> Images;

    /**
     * @brief labels of images
     * 
     */
    std::vector<int> LabelIndex;

    /**
     * @brief SigmentResult[Label][ImageIndex] result of sigment, current no use
     * 
     */
    std::vector<std::vector<cv::Mat>> SigmentResult;

public:
/**
 * @brief read_cifar10_bin
 * 
 * @param path path of cifar10 binary file
 * @param Label label of cifar10 binary file
 * @return true 
 * @return false 
 */
    bool read_cifar10_bin(std::vector<std::string>& path,std::vector<int>& Label)
    {   //read cifar10 binary file
        //read multiple files
        LabelIndex.clear();
        LabelIndex = Label;
        Images.clear();
        Images.resize(10);
        SigmentResult.clear();
        SigmentResult.resize(10);

        for(auto&& i : path)
        {
            if(!_read_cifar10_bin(i))
            {
                return false;
            }
        }
        return true;
    }
private:
    bool _read_cifar10_bin(std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (file) {
            cv::Mat labels;
            std::vector<cv::Mat> images;
            // read the binary data into a buffer
            char* buffer=new char[10000 * 3073];
            file.read(buffer, 10000 * 3073);
            // reshape the buffer into 10000 images of size 3072 (32x32x3)
            cv::Mat data(10000, 32 * 32 * 3+1, CV_8UC1, buffer);
            // split the data into image and label arrays
            labels = data.col(0).clone();
            data = data.colRange(1, data.cols);
            // create the vector of images
            images.reserve(10000);
            for (int i = 0; i < data.rows; i++) {
                cv::Mat R = cv::Mat(32, 32, CV_8UC1, data.ptr(i));
                cv::Mat G = cv::Mat(32, 32, CV_8UC1, data.ptr(i) + 1024);
                cv::Mat B  =cv::Mat(32, 32, CV_8UC1, data.ptr(i)+2048);
                cv::Mat image;
                cv::merge(std::vector<cv::Mat>{B, G, R}, image);

                //cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                images.push_back(image);
            }
            delete[] buffer;

            for (size_t i = 0; i < labels.rows; i++) //seperate images by labels
            {
                Images[labels.at<uchar>(i)].push_back(images[i]);
            }
        }
        else //file open failed
        {
            std::cout<<"file open failed:"<<path<<std::endl;
            return false;
        }
    }
};
