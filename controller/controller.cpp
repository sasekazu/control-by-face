/**
 * Software License Agreement (MIT License)
 * 
 * Copyright (c) 2022, UFACTORY, Inc.
 * 
 * All rights reserved.
 * 
 * @author Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>
 */

#include "xarm/wrapper/xarm_api.h"
#include "training_data_raw.h"
#include <nana/gui.hpp>
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/button.hpp>
#include <thread>
#include <locale>
#include <array>
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
typedef matrix<double, 136, 1> sample_type;
typedef std::array<fp32, 6> vec6;

float v = 10.0f;

vec6 cmd_vel[7];
vec6 send_vel;
int debug_cmd_id = 0;
bool start_flag = false;
bool exit_flag = false;
bool grab_flag = false;


void UpdateCommands() {
    cmd_vel[TOMARU] = vec6({ 0, 0, 0, 0, 0, 0 });
    cmd_vel[MIGI] = vec6({ 0, v, 0, 0, 0, 0 });
    cmd_vel[HIDARI] = vec6({ 0, -v, 0, 0, 0, 0 });
    cmd_vel[TEMAE] = vec6({ v, 0, 0, 0, 0, 0 });
    cmd_vel[OKU] = vec6({ -v, 0, 0, 0, 0, 0 });
    cmd_vel[UE] = vec6({ 0, 0, v, 0, 0, 0 });
    cmd_vel[SHITA] = vec6({ 0, 0, -v, 0, 0, 0 });
}

void GUI() {

    std::locale::global(std::locale(""));
    nana::form fm;
    nana::button btn_start{ fm };
    btn_start.caption(u8"スタート");
    btn_start.events().click([] {
        start_flag = true;
    });
    nana::button btn_stop{ fm };
    btn_stop.caption(u8"ストップ");
    btn_stop.events().click([] {
        start_flag = false;
    });
    nana::button btn_faster{ fm };
    btn_faster.caption(u8"はやく");
    btn_faster.events().click([] {
        v += 1.0;
        std::cout << "velocity: " << v << " [mm/s]" << std::endl;
        UpdateCommands();
        });
    nana::button btn_slower{ fm };
    btn_slower.caption(u8"おそく");
    btn_slower.events().click([] {
        v -= 1.0;
        std::cout << "velocity: " << v << " [mm/s]" << std::endl;
        UpdateCommands();
        });
    nana::button btn_exit{ fm };
    btn_exit.caption(u8"おわる");
    btn_exit.events().click([&] {
        exit_flag = true;
        fm.close();
        });
    nana::button btn_debug{ fm };
    btn_debug.caption(u8"デバッグ 顔を隠して押すとコマンド送り");
    btn_debug.events().click([] {
        ++debug_cmd_id;
        if (debug_cmd_id == 9) {
            debug_cmd_id = 0;
        }
        else if (debug_cmd_id == (int) TSUKAMU) {
            grab_flag = true;
        }
        else if (debug_cmd_id == (int) HANASU) {
            grab_flag = false;
        }
        else {
            send_vel = cmd_vel[debug_cmd_id];
        }
        cout << "DEBUG: " << LABELS[debug_cmd_id] << endl;
        });
    nana::place layout(fm);
    layout.div("vert<<start><stop>><<slower><faster>><debug><exit>");
    layout["start"] << btn_start;
    layout["stop"] << btn_stop;
    layout["slower"] << btn_slower;
    layout["faster"] << btn_faster;
    layout["debug"] << btn_debug;
    layout["exit"] << btn_exit;
    layout.collocate();
    fm.show();
    nana::exec();
    exit_flag = true;
    
}

void faceDetection() {

    UpdateCommands();

    try {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return;
        }

        image_window win;
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        typedef linear_kernel<sample_type> lin_kernel;
        multiclass_linear_decision_function<lin_kernel, double> df;
        deserialize("decision_function.dat") >> df;

        while (!(win.is_closed() || exit_flag))
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
            if (faces.size() > 0) {
                collected_data.shape = pose_model(cimg, faces[0]);
                collected_data.left = faces[0].left();
                collected_data.top = faces[0].top();
                collected_data.width = faces[0].width();
                win.add_overlay(render_face_detections(collected_data.shape));
                static double result_old;
                double result_new = df(PartToMatrix(collected_data));
                if (result_new != result_old) {
                    cout << "COMMAND: " << LABELS[(int)result_new] << " ";
                    cout << (start_flag ? "" : "STOPPED") << endl;
                }
                if (result_new == TSUKAMU) {
                    grab_flag = true;
                }
                else if (result_new == HANASU) {
                    grab_flag = false;
                }
                else {
                    send_vel = cmd_vel[(int)result_new];
                }
                result_old = result_new;
            }
        }
    }
    catch (std::exception& e) {
        std::cout << "Error at main(): " << e.what() << std::endl;
        exit_flag = true;
    }
}

int main(int argc, char **argv) {

    std::thread t1(GUI);
    std::thread t2(faceDetection);

    std::string port;
    if (argc < 2) {
      printf("Please enter IP address\n");
      printf("Here, the IP adress 192.168.1.152 is used\n");
      port = std::string("192.168.1.152");
    } 
    else {
        port = std::string(argv[1]);
    }

    XArmAPI* arm = new XArmAPI(port);
    sleep_milliseconds(500);
    if (arm->error_code != 0) arm->clean_error();
    if (arm->warn_code != 0) arm->clean_warn();
    arm->motion_enable(true);
    arm->set_mode(0);   // position control mode
    arm->set_state(0);
    sleep_milliseconds(500);

    int ret;
    arm->reset(true);
    //fp32 init_pos[6]{ 200, 0, 200, 180, 0, 0 };
    //arm->set_position(init_pos); // これをやると、この後の速度制御が動かなくなる

    arm->set_mode(5);   // velocity control mode
    arm->set_state(0);
    sleep_milliseconds(1000);

    vec6 vel0 = { 20, 0, 0, 0, 0, 0 };
    ret = arm->vc_set_cartesian_velocity(.data()); // mm/s?
    sleep_milliseconds(2500);   // 50 mm

    while (!exit_flag) {
        if (start_flag) {
            ret = arm->vc_set_cartesian_velocity(send_vel.data()); // mm/s?
            ret = arm->set_vacuum_gripper(grab_flag);
        }
        else {
            ret = arm->vc_set_cartesian_velocity(cmd_vel[TOMARU].data());
            ret = arm->set_vacuum_gripper(grab_flag);
        }
        sleep_milliseconds(500);
    }

    cout << "Control thread ended" << endl;

    t1.join();
    t2.join();

    return 0;
}