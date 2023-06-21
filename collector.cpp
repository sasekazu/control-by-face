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
#include <locale>

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

    std::locale::global(std::locale(""));

    nana::form fm;
    nana::button btn_migi{fm};
    btn_migi.caption("1 みぎ");
    btn_migi.events().click([]{
        AddData(data_raw, MIGI, shape);
        cout << data_raw.data.size() << " Right" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_hidari{fm};
    btn_hidari.caption("2 ひだり");
    btn_hidari.events().click([]{
        AddData(data_raw, HIDARI, shape);
        cout << data_raw.data.size() << " Left" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_mae{fm};
    btn_mae.caption("3 まえ");
    btn_mae.events().click([]{
        AddData(data_raw, MAE, shape);
        cout << data_raw.data.size() << " Forward" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ushiro{fm};
    btn_ushiro.caption("4 うしろ");
    btn_ushiro.events().click([]{
        AddData(data_raw, USHIRO, shape);
        cout << data_raw.data.size() << " Backward" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ue{fm};
    btn_ue.caption("5 うえ");
    btn_ue.events().click([]{
        AddData(data_raw, UE, shape);
        cout << data_raw.data.size() << " Up" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_shita{fm};
    btn_shita.caption("6 した");
    btn_shita.events().click([]{
        AddData(data_raw, SHITA, shape);
        cout << data_raw.data.size() << " Down" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_tsukamu{fm};
    btn_tsukamu.caption("7 つかむ");
    btn_tsukamu.events().click([]{
        AddData(data_raw, TSUKAMU, shape);
        cout << data_raw.data.size() << " Grab" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_tomaru{fm};
    btn_tomaru.caption("8 とまる");
    btn_tomaru.events().click([]{
        AddData(data_raw, TOMARU, shape);
        cout << data_raw.data.size() << " Stop" << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_print{fm};
    btn_print.caption("表示");
    btn_print.events().click([]{
        PrintData(data_raw);
    });
    nana::button btn_save{fm};
    btn_save.caption("ほぞん");
    btn_save.events().click([]{
        save("out.cereal", data_raw);
        nana::msgbox msg{"保存しました"};
        msg<<"保存しました。\nファイル名：out.cereal\nデータ数："<<data_raw.data.size()<<"\n";
        msg.show();
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
    t1.join();
}
