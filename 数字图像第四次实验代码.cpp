#include<iostream>
#include<cmath>
#include "opencv.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;		
using namespace std;


// 全局阈值
Mat global_threshold(Mat img, double delta)
{
    // 获取数据
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    double old_threshold = mean(img)[0];

    // 深拷贝图像
    Mat img_return = img.clone();

    // 图像处理
    // 求最终阈值
    int sum_low_pixel = 0;
    double sum_low_pixel_value = 0;
    int sum_high_pixel = 0;
    double sum_high_pixel_value = 0;
    double pixel_value;
    while(1)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width * channel; col++)
            {
                pixel_value = img_return.ptr<uchar>(row)[col];
                // 记录大小像素
                if(pixel_value >= old_threshold)
                {
                    sum_high_pixel = sum_high_pixel + 1;
                    sum_high_pixel_value = sum_high_pixel_value + pixel_value;
                }
                else
                {
                    sum_low_pixel = sum_low_pixel + 1;
                    sum_low_pixel_value = sum_low_pixel_value + pixel_value;
                }
                // cout<<pixel_value;
            }
        }

        double new_threshold = (sum_high_pixel_value / sum_high_pixel + sum_low_pixel_value / sum_low_pixel) / 2;
        if(abs(old_threshold - new_threshold) <= delta)
            break;
        else
            old_threshold = new_threshold;
    }

    // 二值化
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            pixel_value = img_return.ptr<uchar>(row)[col];
            // 记录大小像素
            if(pixel_value >= old_threshold)
            {
                img_return.ptr<uchar>(row)[col] = 255;
            }
            else
            {
                img_return.ptr<uchar>(row)[col] = 0;
            }
            // cout<<pixel_value;
        }
    }
    
    // 返回
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
        }
    }    
    return img_return;
}


// QTUS
Mat qtus_threshold(Mat img)
{
    // 定义参数
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
	int T = 0; //Otsu算法阈值
	double varValue = 0; //类间方差中间值保存
	double w0 = 0; //前景像素点数所占比例
	double w1 = 0; //背景像素点数所占比例
	double u0 = 0; //前景平均灰度
	double u1 = 0; //背景平均灰度
	double Histogram[256] = {0}; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
	uchar *data = img.data;
	double totalNum = img.rows*img.cols; //像素总数

	//计算灰度直方图分布，Histogram数组下标是灰度值，保存内容是灰度值对应像素点数
	for(int i=0;i<img.rows;i++)   //为表述清晰，并没有把rows和cols单独提出来
	{
		for(int j=0;j<img.cols;j++)
		{
			Histogram[data[i*img.step+j]]++;
		}
	}

	for(int i=0;i<255;i++)
	{
		//每次遍历之前初始化各变量
		w1=0;		u1=0;		w0=0;		u0=0;
		//***********背景各分量值计算**************************
		for(int j=0;j<=i;j++) //背景部分各值计算
		{
			w1+=Histogram[j];  //背景部分像素点总数
			u1+=j*Histogram[j]; //背景部分像素总灰度和
		}
		if(w1==0) //背景部分像素点数为0时退出
		{
			continue;
		}
		u1=u1/w1; //背景像素平均灰度
		w1=w1/totalNum; // 背景部分像素点数所占比例
		//***********背景各分量值计算**************************
 
		//***********前景各分量值计算**************************
		for(int k=i+1;k<255;k++)
		{
			w0+=Histogram[k];  //前景部分像素点总数
			u0+=k*Histogram[k]; //前景部分像素总灰度和
		}
		if(w0==0) //前景部分像素点数为0时退出
		{
			break;
		}
		u0=u0/w0; //前景像素平均灰度
		w0=w0/totalNum; // 前景部分像素点数所占比例
		//***********前景各分量值计算**************************
 
		//***********类间方差计算******************************
		double varValueI=w0*w1*(u1-u0)*(u1-u0); //当前类间方差计算
		if(varValue<varValueI)
		{
			varValue=varValueI;
			T=i;
		}
	}
	
    // 二值化
    double pixel_value;
    Mat img_return = img.clone();
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            pixel_value = img_return.ptr<uchar>(row)[col];
            // 记录大小像素
            if(pixel_value >= T)
            {
                img_return.ptr<uchar>(row)[col] = 255;
            }
            else
            {
                img_return.ptr<uchar>(row)[col] = 0;
            }
            // cout<<pixel_value;
        }
    }
    
    // 返回
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
        }
    }    
    return img_return;
}


// Kmeans
Mat Kmeans(Mat img, double delta)
{
    // 定义参数
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    Mat img_return = img.clone();
    double white = 255;
    double black = 0;

    // 图像处理
    // 求最终阈值
    int sum_low_pixel = 0;
    double sum_low_pixel_value = 0;
    int sum_high_pixel = 0;
    double sum_high_pixel_value = 0;
    double pixel_value;
    while(1)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width * channel; col++)
            {
                pixel_value = img_return.ptr<uchar>(row)[col];
                // 记录大小像素
                if(white - pixel_value <= pixel_value - black)
                {
                    sum_high_pixel = sum_high_pixel + 1;
                    sum_high_pixel_value = sum_high_pixel_value + pixel_value;
                }
                else
                {
                    sum_low_pixel = sum_low_pixel + 1;
                    sum_low_pixel_value = sum_low_pixel_value + pixel_value;
                }
                // cout<<pixel_value;
            }
        }

        double new_white = sum_high_pixel_value / sum_high_pixel;
        double new_black = sum_low_pixel_value / sum_low_pixel;

        if(abs(white - new_white) <= delta && abs(black - new_black) <= delta)
            break;
        else
        {
            white = new_white;
            black = new_black;
        }
    }

    // 二值化
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            pixel_value = img_return.ptr<uchar>(row)[col];
            // 记录大小像素
            if(white - pixel_value <= pixel_value - black)
            {
                img_return.ptr<uchar>(row)[col] = white;
            }
            else
            {
                img_return.ptr<uchar>(row)[col] = black;
            }
            // cout<<pixel_value;
        }
    }
    
    // 返回
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
        }
    }    
    return img_return;    
}


// 卷积
Mat conv2d(Mat img, Mat kernal, int padding)
{
    // 定义参数
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    int kernal_size = kernal.rows;
    double pixel_value;
    double kernal_value;
    double sum_pixel_value;
    Mat img_return = img.clone();

    // 卷积
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            sum_pixel_value = 0;
            for(int kernal_row = row - padding; kernal_row < row + kernal_size - padding; kernal_row++)
            {
                for(int kernal_col = col - padding; kernal_col < col + kernal_size - padding; kernal_col++)
                {
                    if(kernal_row < 0 || kernal_row >= height || kernal_col < 0 || kernal_col >= width)
                        pixel_value = 0;
                    else
                        pixel_value = img.at<uchar>(kernal_row, kernal_col);

                    kernal_value = kernal.ptr<float>(kernal_row - row + padding)[kernal_col - col + padding];
                    sum_pixel_value = sum_pixel_value + pixel_value * kernal_value;
                }
            }

            img_return.ptr<uchar>(row)[col] = sum_pixel_value / kernal_size / kernal_size;
        }
    }
    
    return img_return;
}


// roberts
Mat roberts(Mat img)
{
    // 构造卷积核
	Mat roberts_135 = (Mat_<float>(2, 2) << 1, 0, 0, -1);
	Mat roberts_45 = (Mat_<float>(2, 2) << 0, 1, -1, 0);
 
    Mat img_135;
    Mat img_45;
	img_135 = conv2d(img, roberts_135, 1);
    img_45 = conv2d(img, roberts_45, 1);
    
    // 平方
    Mat img_135_2;
    Mat img_45_2;
    pow(img_135, 2.0, img_135_2);
	pow(img_45, 2.0, img_45_2);
 
	Mat img_return = img_135_2 + img_45_2;
	
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    double pixel_value;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            pixel_value = img_return.ptr<uchar>(row)[col];
            pixel_value = sqrt(pixel_value);
        }
    }   

    // 返回
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
            // pixel_value = 255 - pixel_value;
        }
    }
    return img_return;
}
 

// sobel
Mat sobel(Mat img)
{
    // 构造卷积核
	Mat roberts_135 = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
	Mat roberts_45 = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
 
    Mat img_135;
    Mat img_45;
	img_135 = conv2d(img, roberts_135, 1);
    img_45 = conv2d(img, roberts_45, 1);
    
    // 平方
    Mat img_135_2;
    Mat img_45_2;
    pow(img_135, 2.0, img_135_2);
	pow(img_45, 2.0, img_45_2);
 
	Mat img_return = img_135_2 + img_45_2;
	
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    double pixel_value;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
        }
    }   

    // 返回
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            pixel_value = img_return.ptr<uchar>(row)[col];
            pixel_value = 255 - pixel_value;
        }
    }    
    return img_return;
}


// laplacian
Mat laplacian(Mat img)
{
    // 构造卷积核
	Mat roberts_135 = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
 
    Mat img_return;
	img_return = conv2d(img, roberts_135, 1);

    // 返回
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    double pixel_value; 
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width * channel; col++)
        {
            img_return.ptr<uchar>(row)[col] = 255 - img_return.ptr<uchar>(row)[col];
        }
    }    
    return img_return;
}


int main()
{
    // 获取图像
    Mat img = imread("image.png", 0);
    if(img.empty())
    {
        cout<<"no image";
        getchar();
        return -1;
    }
    
    imshow("img", img);


    // 全局阈值处理
    double delta = 1;
    Mat img_global_threshold = global_threshold(img, delta);
    imshow("img_global_threshold", img_global_threshold);


    // QTUS
    Mat img_qtus_threshold = qtus_threshold(img);
    imshow("img_qtus_threshold", img_qtus_threshold);


    // Kmeans
    delta = 1;
    Mat img_kmeans = Kmeans(img, delta);
    imshow("img_kmeans", img_kmeans);


    // roberts
    Mat img_roberts = roberts(img);
    imshow("img_roberts", img_roberts);


    // sobel
    Mat img_sobel = sobel(img);
    imshow("img_sobel", img_sobel);
    

    // laplacian
    Mat img_laplacian = laplacian(img);
    imshow("img_laplacian", img_laplacian);


    // 展示图片
    waitKey(0);
    return 0;
}