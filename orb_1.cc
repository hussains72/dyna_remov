#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> // For ORB
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
    Ptr<BackgroundSubtractor> bgSubtractor = createBackgroundSubtractorKNN(); // Use KNN-based background subtractor
    Ptr<ORB> orb = ORB::create(); // Create ORB detector

    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction using KNN
        bgSubtractor->apply(frame, fgMask);

        // Refine the mask using morphological operations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);

        // Extract the dynamic objects (foreground)
        Mat dynamicObjects;
        frame.copyTo(dynamicObjects, fgMask);

        

        // ORB Feature Extraction on Dynamic Objects
        vector<KeyPoint> dynamicKeypoints;
        Mat dynamicDescriptors;
        orb->detectAndCompute(dynamicObjects, Mat(), dynamicKeypoints, dynamicDescriptors);

        // Draw keypoints on the dynamic objects
        Mat dynamicObjectsWithKeypoints;
        drawKeypoints(dynamicObjects, dynamicKeypoints, dynamicObjectsWithKeypoints, Scalar(0, 255, 0));


        // Extract the static background by removing dynamic objects
        Mat staticBackground;
        frame.copyTo(staticBackground);
        staticBackground.setTo(Scalar(0, 0, 0), fgMask);

  

        // ORB Feature Extraction on Static Background
        vector<KeyPoint> staticKeypoints;
        Mat staticDescriptors;
        orb->detectAndCompute(staticBackground, Mat(), staticKeypoints, staticDescriptors);

        // Draw keypoints on the static objects
        Mat staticObjectsWithKeypoints;
        drawKeypoints(staticBackground, staticKeypoints, staticObjectsWithKeypoints, Scalar(255, 0, 0));

        // Show the results
        imshow("Original Frame", frame);
        imshow("Foreground Mask (KNN)", fgMask);
        imshow("Dynamic Objects with Keypoints", dynamicObjectsWithKeypoints);
        imshow("Static Objects with Keypoints", staticObjectsWithKeypoints);

        // Break loop on key press
        if (waitKey(30) >= 0) break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
