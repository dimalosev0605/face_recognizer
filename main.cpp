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

#include "boost/filesystem.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

// this function reads csv.txt file and fills data structures.
void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, std::map<int, std::string>& objs)
{
    std::ifstream file(filename, std::ifstream::in);
    if(!file.is_open()) {
        std::cerr << "Csv file was not open!\n";
        exit(1);
    }

    std::vector<std::string> lines;
    std::string line;
    while(getline(file, line)) {
        lines.push_back(line);
        line.clear();
    }

    for(std::size_t i = 0; i < lines.size(); ++i) {
        const std::string temp = lines[i];

        if(!temp.empty()) {
            int begin_obj_name = -1;
            int end_obj_name = -1;
            for(std::size_t j = temp.size() - 1; j >= 0; --j) {
                if(temp[j] == '/' && begin_obj_name == -1) {
                    begin_obj_name = j;
                    continue;
                }
                if(temp[j] == '/') {
                    end_obj_name = j;
                    break;
                }
            }

            std::string obj_name;
            for(int k = end_obj_name + 1; k < begin_obj_name; ++k) {
                obj_name.push_back(temp[k]);
            }

            std::size_t begin_obj_id = temp.size() - 1;
            const char separator = ';';

            for(; temp[begin_obj_id ] != separator; --begin_obj_id ) {}

            std::string obj_id;
            for(std::size_t k = begin_obj_id  + 1; k < temp.size(); ++k) {
                obj_id.push_back(temp[k]);
            }

            int id = std::stoi(obj_id);

            objs.insert({id, obj_name});

            std::string abs_img_path;
            for(std::size_t m = 0; m < begin_obj_id ; ++m) {
                abs_img_path.push_back(temp[m]);
            }

            auto img = cv::imread(abs_img_path, cv::IMREAD_GRAYSCALE);

            images.push_back(img);
            labels.push_back(id);
        }
    }
}

// this function create csv.txt file based on received data set. (File will created in directory with processed data set.)
std::string create_csv_file(const boost::filesystem::path& abs_path_to_data_set) {

    boost::filesystem::directory_iterator dir_iter(abs_path_to_data_set);
    boost::filesystem::directory_iterator dir_iter_end;

    std::vector<boost::filesystem::path> obj_dirs;
    for(; dir_iter != dir_iter_end; ++dir_iter) {
        if(boost::filesystem::is_directory(dir_iter->path())) {
            obj_dirs.push_back(dir_iter->path());
        }
    }

    const char separator = ';';
    int obj_id = 0;

    const std::string csv_file = abs_path_to_data_set.string() + '/' + "csv.txt";

    std::ofstream file(csv_file, std::ios_base::out);
    if(!file.is_open()) {
        std::cerr << "Csv file " << csv_file << " was not created!\n";
        exit(1);
    }

    for(std::size_t i = 0; i < obj_dirs.size(); ++i) {
        boost::filesystem::directory_iterator obj_dir_iter(obj_dirs[i]);
        boost::filesystem::directory_iterator obj_dir_iter_end;

        std::vector<std::string> obj_filenames;
        for(; obj_dir_iter != obj_dir_iter_end; ++obj_dir_iter) {
            if(boost::filesystem::is_regular_file(obj_dir_iter->path())) {
                obj_filenames.push_back(obj_dir_iter->path().string() + separator + std::to_string(obj_id));
            }
        }

        for(const auto& filename : obj_filenames) {
            file << filename << '\n';
        }

        ++obj_id;
    }

    return csv_file;
}

int main(int argc, const char **argv)
{
    if (argc != 10) {
        std::cerr << "Usage error.\n"
                  << "Usage:\n"
                  << "[1] <path_to_face_landmarks.dat>\n"
                  << "[2] <path_to_data_set>\n"
                  << "[3] <device_id>\n"
                  << "[4] <show uploaded images?> (int: 0 -> false, 1 -> true)\n"
                  << "[5] <model_threshold> (int: > 0, 0 -> don't set)\n"
                  << "[6] <full_face_shape_size> (int > 0)\n"
                  << "[7] <full_face_shape_padding> (double >= 0)\n"
                  << "[8] <show, what we trying to predict? (int: 0 -> false, 1 -> true)>\n"
                  << "[9] <show predicted_confidence? (int: 0 -> false, 1 -> true)>\n";
        return 1;
    }

    // parse command line arguments.

    const std::string path_to_face_landmarks = argv[1];
    const auto abs_path_to_data_set_dir = boost::filesystem::canonical(argv[2]);

    const auto abs_path_to_csv_file = create_csv_file(abs_path_to_data_set_dir);

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


    // create data structures for data.

    std::vector<cv::Mat> images;
    std::vector<int> labels;
    std::map<int, std::string> objs;

    read_csv(abs_path_to_csv_file, images, labels, objs);

    if(images.empty()) {
        std::cerr << "Images were not uploaded!\n";
        return 1;
    }
    if(images.size() == 1) {
        std::cerr << "This program needs at least two images to work. Add more images to your data set.\n";
        return 1;
    }

    // all images must have same size.

    const int img_width = images[0].cols;
    const int img_height = images[0].rows;

    if(show_uploaded_images) {
        for(std::size_t i = 0; i < images.size(); ++i) {
            cv::imshow(std::to_string(i), images[i]);
        }
    }

    // create face recognizer and train it.

    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
    if(model_threshold != 0) {
        model->setThreshold(model_threshold);
    }
    model->train(images, labels);


    // create frontal face detector and shape predictor.

    dlib::frontal_face_detector frontal_face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor face_shape_predictor;
    dlib::deserialize(path_to_face_landmarks) >> face_shape_predictor;


    // open video device.

    cv::VideoCapture cap(device_id);

    cv::namedWindow("face_recognizer", cv::WINDOW_NORMAL);

    if(!cap.isOpened()) {
        std::cerr << "capture device id " << device_id << " cannot be opened.\n";
        return 1;
    }

    // in this loop we do almost the same processing actions as in prepare_data_set program with images.

    for(;;)
    {
        // read frame from video device.

        cv::Mat cv_frame;
        cap >> cv_frame;
        dlib::cv_image<dlib::bgr_pixel> dlib_frame(cv_frame);


        // on this original frame we will draw rectangles around faces and predicted names.

        cv::Mat original = cv_frame.clone();


        // find faces on the current frame.

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

            // process every face. we will do same actions as in prepare_data_set program with images.

            dlib::rectangle rect_around_full_face = dlib_rects_around_faces[i];
            auto cv_rect_around_face = cv_rect_around_faces[i];

            dlib::matrix<dlib::bgr_pixel> bgr_full_face;

            dlib::full_object_detection full_face_shape = face_shape_predictor(dlib_frame, rect_around_full_face);
            dlib::extract_image_chip(dlib_frame,
                                     dlib::get_face_chip_details(full_face_shape, full_face_shape_size, full_face_shape_padding),
                                     bgr_full_face);

            std::vector<dlib::rectangle> rects_around_little_faces = frontal_face_detector(bgr_full_face);
            if(rects_around_little_faces.size() != 1) {
                continue;
            }

            dlib::rectangle rect_around_little_face = rects_around_little_faces[0];
            dlib::full_object_detection little_face_shape = face_shape_predictor(bgr_full_face, rect_around_little_face);

            std::vector<dlib::point> little_face_points;
            const auto number_of_points = little_face_shape.num_parts();
            for(std::size_t j = 0; j < number_of_points; ++j) {
                little_face_points.push_back(little_face_shape.part(j));
            }

            dlib::point point_0 = little_face_points[0];
            dlib::point point_1 = little_face_points[16];

            dlib::point point_2 = little_face_points[5];
            dlib::point point_3 = little_face_points[11];

            dlib::point point_4 = little_face_points[19];
            dlib::point point_5 = little_face_points[24];

            dlib::point bl(little_face_points[4]);
            dlib::point br(little_face_points[12]);

            int max_y = std::max(bl.y(), br.y());
            bl.y() = max_y;
            br.y() = max_y;

            dlib::point tl(bl.x(), point_4.y());
            dlib::point tr(br.x(), point_5.y());

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
            }


            // Predicting.

            int predicted_label = -1;
            double predicted_confidence = 0.0;
            model->predict(gray_cv_face, predicted_label, predicted_confidence);


            // draw rectangle around face we trying predict.

            cv::rectangle(original, cv_rect_around_face, CV_RGB(0, 255,0), 1);

            // find face in our data set.

            std::string box_text;
            auto iter = objs.find(predicted_label);
            if(iter != objs.end()) {
                box_text = iter->second;
                if(is_predicted_confidence_visible) {
                     box_text +=  ", " + std::to_string(predicted_confidence);
                }
            }
            else {
                box_text = "Unknown";
            }


            // write preson name.

            int pos_x = std::max(cv_rect_around_face.tl().x - 10.0, 0.0);
            int pos_y = std::max(cv_rect_around_face.tl().y - 10.0, 0.0);

            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        imshow("face_recognizer", original);
        cv::waitKey(20);

    }

    return 0;
}
