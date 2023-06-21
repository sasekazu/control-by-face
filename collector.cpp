#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <thread>
#include <nana/gui.hpp>
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/button.hpp>
#include "training_data_raw.h"

using namespace dlib;
using namespace std;
full_object_detection shape;
bool exit_flag = false;
TrainingDataRaw data_raw;

const int MIGI = 1;
const int HIDARI = 2;
const int MAE = 3;
const int USHIRO = 4;
const int UE = 5;
const int SHITA = 6;
const int TSUKAMU = 7;
const int TOMARU = 8;

void capture() {

    nana::form fm;
    nana::button btn_migi{fm};
    btn_migi.caption("migi");
    btn_migi.events().click([]{
        AddData(data_raw, MIGI, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_hidari{fm};
    btn_hidari.caption("hidari");
    btn_hidari.events().click([]{
        AddData(data_raw, HIDARI, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_mae{fm};
    btn_mae.caption("mae");
    btn_mae.events().click([]{
        AddData(data_raw, MAE, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ushiro{fm};
    btn_ushiro.caption("ushiro");
    btn_ushiro.events().click([]{
        AddData(data_raw, USHIRO, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ue{fm};
    btn_ue.caption("ue");
    btn_ue.events().click([]{
        AddData(data_raw, UE, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_shita{fm};
    btn_shita.caption("shita");
    btn_shita.events().click([]{
        AddData(data_raw, SHITA, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_tsukamu{fm};
    btn_tsukamu.caption("tsukamu");
    btn_tsukamu.events().click([]{
        AddData(data_raw, TSUKAMU, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_tomaru{fm};
    btn_tomaru.caption("tomaru");
    btn_tomaru.events().click([]{
        AddData(data_raw, TOMARU, shape);
        save("tmp.cereal", data_raw);
    });
    nana::button btn_print{fm};
    btn_print.caption("print");
    btn_print.events().click([]{
        PrintData(data_raw);
    });
    nana::button btn_save{fm};
    btn_save.caption("save");
    btn_save.events().click([]{
        save("out.cereal", data_raw);
    });


    nana::place layout(fm);
    layout.div("vert<migi><hidari><mae><ushiro><ue><shita><tsukamu><tomaru><print><save>");

    layout["migi"] << btn_migi;
    layout["hidari"] << btn_hidari;
    layout["mae"] << btn_mae;
    layout["ushiro"] << btn_ushiro;
    layout["ue"] << btn_ue;
    layout["shita"] << btn_shita;
    layout["tsukamu"] << btn_tsukamu;
    layout["tomaru"] << btn_tomaru;
    layout["print"] << btn_print;
    layout["save"] << btn_save;
    
    layout.collocate();
    fm.show();
    nana::exec();
    exit_flag = true;
}


int main()
{
    // load("out.cereal", data_raw);

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

        while(!win.is_closed() && !exit_flag)
        {
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }
            cv_image<bgr_pixel> cimg(temp);
            std::vector<rectangle> faces = detector(cimg);
            win.clear_overlay();
            win.set_size(640, 480);
            win.set_background_color(0, 0, 0);
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
        cout << "The bz2 file can be extracted the following command: " << endl;
        cout << "   bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
