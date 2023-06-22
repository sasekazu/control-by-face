#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "training_data_raw.h"
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;
CollectedData collected_data;

typedef matrix<double,136,1> sample_type;

int main() {
    try{
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
        
        image_window win;
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        typedef linear_kernel<sample_type> lin_kernel;
        multiclass_linear_decision_function<lin_kernel,double> df;
        deserialize("decision_function.dat") >> df;

        while(!win.is_closed())
        {
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }
            cv::flip(temp, temp, 1);
            cv_image<bgr_pixel> cimg(temp);
            std::vector<rectangle> faces = detector(cimg);
            win.clear_overlay();
            win.set_size(640, 480);
            win.set_background_color(0, 0, 0);
            win.set_image(cimg);
            if(faces.size() > 0) {
                collected_data.shape = pose_model(cimg, faces[0]);
                collected_data.left = faces[0].left();
                collected_data.top = faces[0].top();
                collected_data.width = faces[0].width();
                win.add_overlay(render_face_detections(collected_data.shape));
                cout << "Result: " << df(PartToMatrix(collected_data)) << endl;
            }
        }
    }
    catch(std::exception& e){
        std::cout << "Error at main(): " << e.what() << std::endl;
    }
}
