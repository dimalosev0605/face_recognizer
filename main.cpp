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

    dlib::shape_predictor face_shape_predictor;
    dlib::deserialize(argv[1]) >> face_shape_predictor;

    cv::VideoCapture cap(deviceId);

    cv::namedWindow("face_recognizer", cv::WINDOW_NORMAL);

    if(!cap.isOpened()) {
        std::cerr << "Capture Device ID " << deviceId << "cannot be opened.\n";
        return -1;
    }

    for(;;)
    {
        cv::Mat cv_frame;
        cap >> cv_frame;
        dlib::cv_image<dlib::bgr_pixel> dlib_frame(cv_frame);

        cv::Mat original = cv_frame.clone();

        std::vector<dlib::rectangle> dlib_rects_around_faces = frontal_face_detector(dlib_frame);
        std::vector<cv::Rect_<int>> cv_rect_around_faces;

        for(std::size_t i = 0; i < dlib_rects_around_faces.size(); ++i) {
            cv::Rect_<int> cv_rect_around_face(dlib_rects_around_faces[i].tl_corner().x(),
                                               dlib_rects_around_faces[i].tl_corner().y(),
                                               dlib_rects_around_faces[i].width(),
                                               dlib_rects_around_faces[i].height());
            cv_rect_around_faces.push_back(cv_rect_around_face);
        }

        for(std::size_t i = 0; i < cv_rect_around_faces.size(); ++i) {
            cv::rectangle(original, cv_rect_around_faces[i], CV_RGB(0, 255, 0), 1);
        }


        //
        std::vector<dlib::full_object_detection> face_shapes;
        for(std::size_t i = 0; i < dlib_rects_around_faces.size(); ++i) {
            dlib::full_object_detection face_shape = face_shape_predictor(dlib_frame, dlib_rects_around_faces[i]);
            face_shapes.push_back(face_shape);
        }


        dlib::array<dlib::array2d<dlib::rgb_pixel>> face_chips;
        dlib::extract_image_chips(dlib_frame, dlib::get_face_chip_details(face_shapes), face_chips);

        std::vector<cv::Mat> cv_faces;
        for(std::size_t i = 0; i < face_chips.size(); ++i) {
            cv_faces.push_back(dlib::toMat(face_chips[i]));
        }

        for(std::size_t i = 0; i < cv_faces.size(); ++i) {
            cv::Mat gray;
            cv::cvtColor(cv_faces[i], gray, cv::COLOR_BGR2GRAY);
            int predict = model->predict(gray);

            std::string box_text;
            if(predict == 0) {
                box_text = "dima";
                cv::putText(original, box_text, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
            if(predict == 1) {
                box_text = "edgar";
                cv::putText(original, box_text, cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
            if(predict == 2) {
                box_text = "putin";
                cv::putText(original, box_text, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
        }
        //


        imshow("face_recognizer", original);

        char key = (char) cv::waitKey(20);

        if(key == 27) break;
    }

    return 0;
}
