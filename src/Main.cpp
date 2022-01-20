#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
int main()
{
    std::cout << "PROJECT1" << std::endl;
    std::string projectPath = "E:/dev/vision_par_ordinateur/5A-OpenCV-Poker/poker/resources/antoine.png";
    // Read the image file
    Mat image = imread(projectPath);

    if (image.empty()) // Check for failure
    {
        cout << "Could not open or find the image" << endl;
        system("pause"); //wait for any key press
        return -1;
    }

    String windowName = "My HelloWorld Window"; //Name of the window

    namedWindow(windowName); // Create a window

    cv::putText(image, "Derp", cv::Point2f(100, 100),2,1,cv::Scalar(255,255,255));

    imshow(windowName, image); // Show our image inside the created window.

    waitKey(0); // Wait for any keystroke in the window

    destroyWindow(windowName); //destroy the created window

    return 0;
}