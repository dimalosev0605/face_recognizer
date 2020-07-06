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
#include <dlib/image_io.h>

#include <iostream>
#include <fstream>
#include <sstream>

static void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if(!file) {
        std::string error_message = "No valid input file was given, please check the given filename.";
    }
    std::string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            auto img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            images.push_back(img);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char **argv)
{
    if (argc != 10) {
        std::cerr << "CLI error.\n"
                  << "Usage:\n"
                  << "[1] <path_to_face_landmarks.dat>\n"
                  << "[2] <path_to_csv_file>\n"
                  << "[3] <device_id>\n"
                  << "[4] <show uploaded images?> (int: 0 -> false, 1 -> true)\n"
                  << "[5] <model_threshold> (int: > 0, 0 -> don't set)\n"
                  << "[6] <full_face_shape_size> (int > 0). SAME AS IN PREPROCESSING!\n"
                  << "[7] <full_face_shape_padding> (double >= 0). SAME AS IN PREPROCESSING!\n"
                  << "[8] <show, what we trying to predict? (int: 0 -> false, 1 -> true)>\n"
                  << "[9] <show predicted_confidence? (int: 0 -> false, 1 -> true)>\n";
        return 1;
    }

    std::string fn_csv = std::string(argv[2]);
    int device_id = std::stoi(argv[3]);

    const int show_uploaded_images_temp = std::stoi(argv[4]);
    const bool show_uploaded_images = show_uploaded_images_temp == 0 ? false : true;

    const int model_threshold = std::stoi(argv[5]);

    const int full_face_shape_size = std::stoi(argv[6]);
    const double full_face_shape_padding = std::stod(argv[7]);

    const int show_trying_prediction_temp = std::stoi(argv[8]);
    const bool show_trying_prediction = show_trying_prediction_temp == 0 ? false : true;

    const int is_predicted_confidence_visible_temp = std::stoi(argv[9]);
    const bool is_predicted_confidence_visible = is_predicted_confidence_visible_temp == 0 ? false : true;

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        std::cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << '\n';
        return 1;
    }

    if(images.empty()) {
        std::cerr << "Images were not uploaded!\n";
        return 1;
    }

    const int img_width = images[0].cols;
    const int img_height = images[0].rows;

    if(show_uploaded_images) {
        for(std::size_t i = 0; i < images.size(); ++i) {
            cv::imshow(std::to_string(i), images[i]);
        }
    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
    if(model_threshold != 0) {
        model->setThreshold(model_threshold);
    }
    model->train(images, labels);

    dlib::frontal_face_detector frontal_face_detector = dlib::get_frontal_face_detector();

    dlib::shape_predictor face_shape_predictor;
    dlib::deserialize(argv[1]) >> face_shape_predictor;

    cv::VideoCapture cap(device_id);

    cv::namedWindow("face_recognizer", cv::WINDOW_NORMAL);

    if(!cap.isOpened()) {
        std::cerr << "Capture Device ID " << device_id << " cannot be opened.\n";
        return 1;
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

            dlib::rectangle rect_around_full_face = dlib_rects_around_faces[i];
            auto cv_rect_around_face = cv_rect_around_faces[i];

            dlib::matrix<dlib::bgr_pixel> bgr_full_face;

            dlib::full_object_detection full_face_shape = face_shape_predictor(dlib_frame, rect_around_full_face);
            dlib::extract_image_chip(dlib_frame,
                                     dlib::get_face_chip_details(full_face_shape, full_face_shape_size, full_face_shape_padding),
                                     bgr_full_face);

            std::vector<dlib::rectangle> rects_around_little_faces = frontal_face_detector(bgr_full_face);
            if(rects_around_little_faces.size() != 1) {
                std::cerr << "PIZDA!\n";
                continue;
            }

            dlib::rectangle rect_around_little_face = rects_around_little_faces[0];
            dlib::full_object_detection little_face_shape = face_shape_predictor(bgr_full_face, rect_around_little_face);

            std::vector<dlib::point> little_face_points;
            const auto number_of_points = little_face_shape.num_parts();
            for(std::size_t j = 0; j < number_of_points; ++j) {
                little_face_points.push_back(little_face_shape.part(j));
            }

            // main points:

            // near ears
            dlib::point point_0 = little_face_points[0];
            dlib::point point_1 = little_face_points[16];

            // under mouth
            dlib::point point_2 = little_face_points[5];
            dlib::point point_3 = little_face_points[11];

            // above the eyes
            dlib::point point_4 = little_face_points[19];
            dlib::point point_5 = little_face_points[24];

            // draw processed face
            dlib::point bl(little_face_points[4]);
            dlib::point br(little_face_points[12]);

            // max y?
            int max_y = std::max(bl.y(), br.y());
            bl.y() = max_y;
            br.y() = max_y;

            dlib::point tl(bl.x(), point_4.y());
            dlib::point tr(br.x(), point_5.y());

            // min y?
            int min_y = std::min(tl.y(), tr.y());
            tl.y() = min_y;
            tr.y() = min_y;

            dlib::rectangle dlib_processed_face_rect(tl, br);
            cv::Rect cv_processed_face_rect(dlib_processed_face_rect.left(), dlib_processed_face_rect.top(),
                                            dlib_processed_face_rect.width(), dlib_processed_face_rect.height());

            cv::Mat processed_face = dlib::toMat(bgr_full_face)(cv_processed_face_rect);

            cv::Mat resized_processed_face;
            cv::resize(processed_face, resized_processed_face, cv::Size(img_width, img_height), 0, 0, cv::INTER_CUBIC);

            dlib::matrix<unsigned char> gray_processed_face;
            dlib::assign_image(gray_processed_face, dlib::cv_image<dlib::bgr_pixel>(resized_processed_face));

            cv::Mat gray_cv_face = dlib::toMat(gray_processed_face);

            if(show_trying_prediction) {
                cv::imshow("Trying predict", gray_cv_face);
//                cv::destroyWindow("Trying predict");
            }

            int predicted_label = -1;
            double predicted_confidence = 0.0;

            model->predict(gray_cv_face, predicted_label, predicted_confidence);

            cv::rectangle(original, cv_rect_around_face, CV_RGB(0, 255,0), 1);

            std::string box_text;
            if(predicted_label == 0) {
                box_text = "dima";
                if(is_predicted_confidence_visible) {
                    box_text += ", " + std::to_string(predicted_confidence);
                }
            }
            if(predicted_label == 1) {
                box_text = "edgar";
                if(is_predicted_confidence_visible) {
                    box_text += ", " + std::to_string(predicted_confidence);
                }
            }

            int pos_x = std::max(cv_rect_around_face.tl().x - 10.0, 0.0);
            int pos_y = std::max(cv_rect_around_face.tl().y - 10.0, 0.0);

            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        imshow("face_recognizer", original);
        cv::waitKey(20);
    }

    return 0;
}
