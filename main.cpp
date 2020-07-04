#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <iostream>
#include <fstream>
#include <sstream>

static void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        std::string error_message = "No valid input file was given, please check the given filename.";
    }
    std::string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char **argv)
{
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>\n";
        std::cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection.\n";
        std::cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database.\n";
        std::cout << "\t <device id> -- The webcam device id to grab frames from.\n";
        return 1;
    }

    std::string fn_csv = std::string(argv[2]);
    int deviceId = atoi(argv[3]);

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        std::cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << '\n';
        return 1;
    }

    int im_width = images[0].cols;
    int im_height = images[0].rows;

    cv::Ptr<cv::face::FisherFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
    model->train(images, labels);

    dlib::frontal_face_detector frontal_face_detector = dlib::get_frontal_face_detector();

    cv::VideoCapture cap(deviceId);

    if(!cap.isOpened()) {
        std::cerr << "Capture Device ID " << deviceId << "cannot be opened.\n";
        return -1;
    }

    for(;;)
    {
        cv::Mat cv_frame;
        cap >> cv_frame;

        cv::Mat original = cv_frame.clone();

        cv::Mat gray;
        cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);

        dlib::cv_image<dlib::bgr_pixel> d_frame(cv_frame);

        std::vector<dlib::rectangle> d_faces = frontal_face_detector(d_frame);
        std::vector<cv::Rect_<int>> cv_faces;

        for(std::size_t i = 0; i < d_faces.size(); ++i) {
            cv::Rect_<int> face_rect(d_faces[i].tl_corner().x(), d_faces[i].tl_corner().y(), d_faces[i].width(), d_faces[i].height());
            cv_faces.push_back(face_rect);
        }

        for(std::size_t i = 0; i < cv_faces.size(); ++i) {
            cv::Rect face_i = cv_faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            cv::Mat face = gray(face_i);

            cv::Mat face_resized;
            cv::resize(face, face_resized, cv::Size(im_width, im_height), 1.0, 1.0, cv::INTER_CUBIC);

            int prediction = model->predict(face_resized);

            cv::rectangle(original, face_i, CV_RGB(0, 255,0), 1);

            std::string box_text;
            if(prediction == 0) {
                box_text = "dima";
            }
            if(prediction == 1) {
                box_text = "edgar";
            }

            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);

            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        imshow("face_recognizer", original);

        char key = (char) cv::waitKey(20);

        if(key == 27) break;
    }

    return 0;
}
