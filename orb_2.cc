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

    Mat frame, fgMask;
    Ptr<BackgroundSubtractor> bgSubtractor = createBackgroundSubtractorKNN(); // Use KNN background subtractor
    Ptr<ORB> orb = ORB::create(); // Create ORB detector

    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()) break;

        // Apply background subtraction using KNN
        bgSubtractor->apply(frame, fgMask);

        // Refine the mask using morphological operations
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel); // Fill holes
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);  // Remove noise

        // Further improve mask by filtering small areas (connected components)
        Mat labels, stats, centroids;
        int numLabels = connectedComponentsWithStats(fgMask, labels, stats, centroids);
        for (int i = 1; i < numLabels; i++) { // Skip label 0 (background)
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < 500) { // Filter out small areas (adjust threshold as needed)
                fgMask.setTo(0, labels == i);
            }
        }

        // Extract the static background by removing dynamic objects
        Mat staticBackground = frame.clone();
        staticBackground.setTo(Scalar(0, 0, 0), fgMask);

        // ORB Feature Extraction
        vector<KeyPoint> keypoints;
        Mat descriptors;
        orb->detectAndCompute(frame, Mat(), keypoints, descriptors);

        // Filter out keypoints in the dynamic (foreground) areas
        vector<KeyPoint> filteredKeypoints;
        for (const auto& kp : keypoints) {
            if (fgMask.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x)) == 0) {
                filteredKeypoints.push_back(kp);
            }
        }

        // Draw keypoints on the static background
        Mat staticObjectsWithKeypoints;
        drawKeypoints(staticBackground, filteredKeypoints, staticObjectsWithKeypoints, Scalar(0, 255, 0));

        // Show the results
        imshow("Original Frame", frame);
        imshow("Refined Mask", fgMask);
        imshow("Static Objects with Keypoints", staticObjectsWithKeypoints);

        // Break loop on key press
        if (waitKey(30) >= 0) break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
