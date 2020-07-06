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
    if (!file) {
        std::string error_message = "No valid input file was given, please check the given filename.";
    }
    std::string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            /*
            dlib::matrix<unsigned char> img;


            dlib::load_jpeg(img, path);

//            std::cout << "Show images:\n";
//            dlib::image_window* win = new dlib::image_window;;
//            win->set_image(img);

            cv::Mat cv_img = dlib::toMat(img);

//            cv::imshow(path, cv_img);

            images.push_back(cv_img);
            labels.push_back(atoi(classlabel.c_str()));
            */

            auto img = cv::imread(path, cv::IMREAD_GRAYSCALE);
//            cv::imshow(path, img);
            images.push_back(img);
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

    const int width = images[0].cols;
    const int height = images[0].rows;

    // show loaded images
//    std::cout << "Show loaded images:\n";
//    for(int i = 0; i < images.size(); ++i) {
//        cv::imshow(std::to_string(i), images[i]);
//    }

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
//    model->setThreshold(500);
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

            dlib::rectangle rect_around_full_face = dlib_rects_around_faces[i];
            auto cv_rect_around_face = cv_rect_around_faces[i];

            dlib::matrix<dlib::bgr_pixel> bgr_full_face;

            dlib::full_object_detection full_face_shape = face_shape_predictor(dlib_frame, rect_around_full_face);
            dlib::extract_image_chip(dlib_frame, dlib::get_face_chip_details(full_face_shape, 200, 0.5), bgr_full_face);

            std::vector<dlib::rectangle> rects_around_little_face = frontal_face_detector(bgr_full_face);
            if(rects_around_little_face.size() != 1) {
                std::cerr << "PIZDA!\n";
                continue;
    //            std::exit(-228);
            }

            dlib::rectangle rect_around_little_face = rects_around_little_face[0];
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

//            static int width = processed_face.cols;
//            static int height = processed_face.rows;

            cv::Mat resized_processed_face;
            cv::resize(processed_face, resized_processed_face, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

            dlib::matrix<unsigned char> gray_processed_face;
            dlib::assign_image(gray_processed_face, dlib::cv_image<dlib::bgr_pixel>(resized_processed_face));

            cv::Mat gray_cv_face = dlib::toMat(gray_processed_face);
            std::cout << "try predict:\n";
            cv::imshow("try predict", gray_cv_face);
            int predicted_label = -1;
            double predicted_confidence = 0.0;

            model->predict(gray_cv_face, predicted_label, predicted_confidence);

            cv::rectangle(original, cv_rect_around_face, CV_RGB(0, 255,0), 1);

            std::string box_text;
            if(predicted_label == 0) {
                box_text = "dima, " + std::to_string(predicted_confidence);
            }
            if(predicted_label == 1) {
                box_text = "edgar, " + std::to_string(predicted_confidence);
            }

            int pos_x = std::max(cv_rect_around_face.tl().x - 10.0, 0.0);
            int pos_y = std::max(cv_rect_around_face.tl().y - 10.0, 0.0);

            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            /*
            auto rect_around_face = cv_rect_around_faces[i];

            dlib::full_object_detection face_shape = face_shape_predictor(dlib_frame, dlib_rects_around_faces[i]);

            dlib::matrix<dlib::rgb_pixel> rgb_processed_face;
            dlib::extract_image_chip(dlib_frame, dlib::get_face_chip_details(face_shape, 100, 0), rgb_processed_face);

            dlib::matrix<unsigned char> gray_processed_face;
            dlib::assign_image(gray_processed_face, rgb_processed_face);
//            dlib::equalize_histogram(gray_processed_face);
//            std::cout << cv_rect_around_faces[i].width << " - " << cv_rect_around_faces[i].height << '\n';

            cv::Mat gray_cv_face = dlib::toMat(gray_processed_face);
            int predicted_label = -1;
            double predicted_confidence = 0.0;

            model->predict(gray_cv_face, predicted_label, predicted_confidence);

            cv::rectangle(original, rect_around_face, CV_RGB(0, 255,0), 1);

            std::string box_text;
            if(predicted_label == 0) {
                box_text = "dima, " + std::to_string(predicted_confidence);
            }
            if(predicted_label == 1) {
                box_text = "edgar, " + std::to_string(predicted_confidence);
            }

            int pos_x = std::max(rect_around_face.tl().x - 10.0, 0.0);
            int pos_y = std::max(rect_around_face.tl().y - 10.0, 0.0);

            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            */
        }


        //
        /*
        std::vector<dlib::full_object_detection> face_shapes;
        for(std::size_t i = 0; i < dlib_rects_around_faces.size(); ++i) {
            dlib::full_object_detection face_shape = face_shape_predictor(dlib_frame, dlib_rects_around_faces[i]);
            face_shapes.push_back(face_shape);
        }

        std::vector<dlib::matrix<dlib::rgb_pixel>> rgb_processed_faces;
        for(std::size_t i = 0; i < face_shapes.size(); ++i) {
            dlib::matrix<dlib::rgb_pixel> processed_face;
            dlib::extract_image_chip(dlib_frame, dlib::get_face_chip_details(face_shapes[i], 100, 0), processed_face);
            rgb_processed_faces.push_back(std::move(processed_face));
        }

        std::vector<dlib::matrix<unsigned char>> gray_processed_faces;
        for(std::size_t i = 0; i < rgb_processed_faces.size(); ++i) {
            dlib::matrix<unsigned char> gray_processed_face;
            dlib::assign_image(gray_processed_face, rgb_processed_faces[i]);
            gray_processed_faces.push_back(std::move(gray_processed_face));
        }

        for(std::size_t i = 0; i < gray_processed_faces.size(); ++i) {
            cv::Mat gray_cv_face = dlib::toMat(gray_processed_faces[i]);

            int predicted_label = -1;
            double predicted_confidence = 0.0;


            model->predict(gray_cv_face, predicted_label, predicted_confidence);

            std::string box_text;
            box_text = "predicted_label = " + std::to_string(predicted_label)
                    + ", predicted_confidence = " + std::to_string(predicted_confidence);
            cv::putText(original, box_text, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        */
        //

        imshow("face_recognizer", original);

        char key = (char) cv::waitKey(20);

        if(key == 27) break;
    }

    return 0;
}
