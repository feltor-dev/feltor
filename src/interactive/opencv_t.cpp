#include "opencv2/opencv.hpp"

int main(int, char**)
{
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
    {
        std::cerr << "Camera not found\n";
        return -1;
    }
    const unsigned Nx=320, Ny=240;
    cap.set(CV_CAP_PROP_FRAME_WIDTH,  Nx );//only takes certain values
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, Ny );

    cv::Mat edges;
    cv::Mat last, current, flow, vel(Ny, Nx, CV_32F);
    cv::Mat show;
    std::vector<cv::Mat> v;
    cap >> last;
    cv::cvtColor(last, last, CV_BGR2GRAY); //convert colors
    cv::namedWindow("Current",cv::WINDOW_NORMAL);
    cv::namedWindow("flow",cv::WINDOW_NORMAL);
    for(;;)
    {
        cap >> current; // get a new frame from camera
        cv::cvtColor(current, current, CV_BGR2GRAY); //convert colors
        cv::GaussianBlur(current, current, cv::Size(21,21), 0, 0); //Kernel size, sigma_x, sigma_y
        //maybe use filter first ?
        calcOpticalFlowFarneback(last, current, flow, 
                0.5, //double pyr_scale, 
                1, //int levels, 
                5,//int winsize, 
                3, //int iterations, 
                5, //int poly_n, 
                1.2, //double poly_sigma, 
                0
                );
        cv::split( flow, v);
        //erster index y, zweiter index x
        for( unsigned i=0; i<v[0].rows; i++)
            for( unsigned j=0; j<v[0].cols; j++)
                vel.at<float>( i,j) = sqrt( v[0].at<float>(i,j)*v[0].at<float>(i,j) + v[1].at<float>(i,j)*v[1].at<float>(i,j) );
        for( unsigned i=0; i<vel.rows; i++)
            for( unsigned j=0; j<vel.cols; j++)
                if( vel.at<float>(i,j) < 1) vel.at<float>(i,j) = 0;
        //or maybe use filter now to reduce noise?
        //scale velocity to 1 in order to account for distance from camera
        double min, max;
        cv::minMaxLoc( vel, &min, &max);
        if( max > 1) // if someone is there
            for( unsigned i=0; i<vel.rows; i++)
                for( unsigned j=0; j<vel.cols; j++)
                    vel.at<float>( i,j) /= max;
        std::cout << "Max " << max<<"\n";
        //vel.convertTo( show, CV_8UC1, 255);
        //cv::fastNlMeansDenoising( show, show, 3, 7, 21);
        

        //cv::Canny(edges, edges, 0, 30, 3); //edge detector
        cv::flip( vel, vel, +1);
        imshow("Current", current);
        imshow("flow", vel);
        //imshow("denoised", show);
        //swap fields
        cv::Mat temp = last;
        last = current;
        current = temp;
        if(cv::waitKey(30) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
