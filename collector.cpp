#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <thread>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

using namespace dlib;
using namespace std;
full_object_detection shape;
bool exit_flag = false;

struct TrainingDataRaw
{
    std::vector<int> labels;
    std::vector<std::vector<int>> data;

    template <class Archive>
    void serialize(Archive &archive) {
        archive(labels, data);
    }
};
TrainingDataRaw data_raw;


void AddData(TrainingDataRaw& data_raw, int label, const full_object_detection& shape){
    std::vector<int> parts;
    data_raw.labels.push_back(label);
    for(unsigned int i=0; i<shape.num_parts(); ++i) {
        parts.push_back(shape.part(i).x());
        parts.push_back(shape.part(i).y());
    }
    data_raw.data.push_back(parts);
}

void PrintData(const TrainingDataRaw& data_raw){
    for(unsigned int i=0; i<data_raw.labels.size(); ++i) {
        std::cout << "Data " << i << "----------------------" << std::endl;
        std::cout << "Label: " << data_raw.labels[i] << std::endl;
        std::cout << "Data: ";
        for(unsigned int j=0; j<data_raw.data[i].size(); ++j) {
            std::cout << data_raw.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void save(){
    std::ofstream ofs("out.cereal", std::ios::binary);
    cereal::BinaryOutputArchive oarchive(ofs); 
    oarchive(data_raw); 
}

void load(){
    std::ifstream ifs("out.cereal", std::ios::binary);
    cereal::BinaryInputArchive iarchive(ifs); 
    iarchive(data_raw); 
}

void capture() {
    while(!exit_flag){
        char key;
        std::cout << "Press label or ESC to exit: ";
        std::cin >> key;
        switch (key) {
            case '1':
                AddData(data_raw, 1, shape);
                break;
            case '2':
                AddData(data_raw, 2, shape);
                break;
            case '3':
                AddData(data_raw, 3, shape);
                break;
            case 'p':
                PrintData(data_raw);
                break;
            case 's':
                save();
                break;
            case 27: // Escape
                exit_flag = true;
                break;
        }
    }
}


int main()
{
    // load();

    std::thread t1(capture);
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        while(!win.is_closed() || exit_flag)
        {
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }
            cv_image<bgr_pixel> cimg(temp);
            std::vector<rectangle> faces = detector(cimg);
            win.clear_overlay();
            win.set_image(cimg);
            if(faces.size() > 0) {
                shape = pose_model(cimg, faces[0]);
                win.add_overlay(render_face_detections(shape));
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
