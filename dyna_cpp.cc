#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open video capture (0 for webcam or provide video file path)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video source." << endl;
        return -1;
    }

    Mat frame, fgMask, background;
    Ptr<BackgroundSubtractor> bgSubtractor = createBackgroundSubtractorMOG2(); // Create background subtractor

    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction
        bgSubtractor->apply(frame, fgMask);

        // Refine the mask using morphological operations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);

        // Extract the dynamic objects (foreground)
        Mat dynamicObjects;
        frame.copyTo(dynamicObjects, fgMask);

        // Extract the static background by removing dynamic objects
        Mat staticBackground;
        frame.copyTo(staticBackground);
        staticBackground.setTo(Scalar(0, 0, 0), fgMask);

        // Show the results
        imshow("Original Frame", frame);
        imshow("Foreground Mask", fgMask);
        imshow("Dynamic Objects", dynamicObjects);
        imshow("Static Background", staticBackground);

        // Break loop on key press
        if (waitKey(30) >= 0) break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
