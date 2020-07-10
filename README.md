Face recognizer.

You need load "shape_predictor_68_face_landmarks.dat" for dlib::shape_predictor and pass it as first argument to program. (You can download it here: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

Pass directory with preprocessed data set as second parameter. (You must receive preprocessed data via this program: https://github.com/dimalosev0605/prepare_data_set).

Pass video device id as third parameter. (Usually it is 0).

Pass 1 as fourth parameter if you want to see loaded preprocessed images.

In fifth parameter you can set prediction model threshold. (You can read about it here: https://docs.opencv.org/4.3.0/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63).

Read about sixth and seventh parameters here: http://dlib.net/dlib/image_transforms/interpolation_abstract.h.html (100-300 is optimal for sixth and 0.5 is optimal for seventh).

Pass 1 as eight parameter if you want to see what we trying to predict.

Pass 1 as ninth parameter if you want to see predicted confidence near every face.







