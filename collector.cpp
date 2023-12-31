#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
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
bool exit_flag = false;
CollectedData collected_data;
TrainingDataRaw data_raw;

void capture() {

    std::locale::global(std::locale(""));

    nana::form fm;
    nana::button btn_tomaru{fm};
    btn_tomaru.caption("0 とまる");
    btn_tomaru.events().click([]{
        AddData(data_raw, TOMARU, collected_data);
        cout << data_raw.data.size() << " " << LABELS[TOMARU] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_migi{fm};
    btn_migi.caption("1 みぎ");
    btn_migi.events().click([]{
        AddData(data_raw, MIGI, collected_data);
        cout << data_raw.data.size() << " " << LABELS[MIGI] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_hidari{fm};
    btn_hidari.caption("2 ひだり");
    btn_hidari.events().click([]{
        AddData(data_raw, HIDARI, collected_data);
        cout << data_raw.data.size() << " " << LABELS[HIDARI] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_mae{fm};
    btn_mae.caption("3 てまえ");
    btn_mae.events().click([]{
        AddData(data_raw, TEMAE, collected_data);
        cout << data_raw.data.size() << " " << LABELS[TEMAE] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ushiro{fm};
    btn_ushiro.caption("4 おく");
    btn_ushiro.events().click([]{
        AddData(data_raw, OKU, collected_data);
        cout << data_raw.data.size() << " " << LABELS[OKU] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_ue{fm};
    btn_ue.caption("5 うえ");
    btn_ue.events().click([]{
        AddData(data_raw, UE, collected_data);
        cout << data_raw.data.size() << " " << LABELS[UE] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_shita{fm};
    btn_shita.caption("6 した");
    btn_shita.events().click([]{
        AddData(data_raw, SHITA, collected_data);
        cout << data_raw.data.size() << " " << LABELS[SHITA] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_tsukamu{fm};
    btn_tsukamu.caption("7 つかむ");
    btn_tsukamu.events().click([]{
        AddData(data_raw, TSUKAMU, collected_data);
        cout << data_raw.data.size() << " " << LABELS[TSUKAMU] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_hanasu{fm};
    btn_hanasu.caption("8 はなす");
    btn_hanasu.events().click([]{
        AddData(data_raw, HANASU, collected_data);
        cout << data_raw.data.size() << " " << LABELS[HANASU] << endl;
        save("tmp.cereal", data_raw);
    });
    nana::button btn_print{fm};
    btn_print.caption("ひょうじ");
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
    layout.div("vert<tomaru><migi><hidari><mae><ushiro><ue><shita><tsukamu><hanasu><<print><save>>");

    layout["tomaru"] << btn_tomaru;
    layout["migi"] << btn_migi;
    layout["hidari"] << btn_hidari;
    layout["mae"] << btn_mae;
    layout["ushiro"] << btn_ushiro;
    layout["ue"] << btn_ue;
    layout["shita"] << btn_shita;
    layout["tsukamu"] << btn_tsukamu;
    layout["hanasu"] << btn_hanasu;
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
