#include "ofApp.h"
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

ofVec3f velocity;
float v = 3.0f;
bool grab = false;


class Command {
public:
    Command(const ofVec3f& v = ofVec3f(0,0,0), bool g = false) 
        : v_(v), g_(g) {}
    void Set(const ofVec3f& v, bool g) { 
        v_ = v;
        g_ = g; 
    }
    void Apply(ofVec3f& v, bool& g) {
        v = v_;
        g = g_;
    }
private:
    ofVec3f v_;
    bool g_;
};
Command cmd[8];

void faceui() {
    cmd[TOMARU] .Set(ofVec3f( 0, 0, 0), false);
    cmd[MIGI]   .Set(ofVec3f( v, 0, 0), false);
    cmd[HIDARI] .Set(ofVec3f(-v, 0, 0), false);
    cmd[MAE]    .Set(ofVec3f( 0, 0, v), false);
    cmd[USHIRO] .Set(ofVec3f( 0, 0,-v), false);
    cmd[UE]     .Set(ofVec3f( 0, v, 0), false);
    cmd[SHITA]  .Set(ofVec3f( 0,-v, 0), false);
    cmd[TSUKAMU].Set(ofVec3f( 0, 0, 0), true );

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

        while (!win.is_closed())
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
                double result = df(PartToMatrix(collected_data));
                cout << "Result: " << LABELS[(int)result] << endl;
                cmd[(int)result].Apply(velocity, grab);
            }
        }
    }
    catch (std::exception& e) {
        std::cout << "Error at main(): " << e.what() << std::endl;
    }
}

//--------------------------------------------------------------
void ofApp::setup(){

    thread_face.reset(new thread(faceui));


    // 球の半径
    icoSphere.setRadius(5);
    // 球の解像度
    icoSphere.setResolution(4);
    // 球の位置
    icoSphere.setPosition(0, 20, 0);

    // ライトの設定
    ofSetSmoothLighting(true);
    // ライトの拡散光の色
    pointLight.setDiffuseColor(ofFloatColor(1, 1, 1));
    // ライトの反射光の色
    pointLight.setSpecularColor(ofFloatColor(1.f, 1.f, 1.f));
    // ライトの環境光の色
    pointLight.setAmbientColor(ofFloatColor(0.1f, 0.1f, 0.1f));
    // ライトの位置
    pointLight.setPosition(0, 200, 200);

    // 材料の輝き (max: 128)
    material.setShininess(120);
    // 材料の拡散色
    material.setDiffuseColor(ofFloatColor::red);

    // カメラの位置
    cam.setPosition(10, 40, 80);
    // カメラの視点
    cam.lookAt(glm::vec3(0, 20, 0), glm::vec3(0, 1, 0));
}

//--------------------------------------------------------------
void ofApp::update(){
    // 球の色
    if (grab) {
        material.setDiffuseColor(ofFloatColor::red);
    }
    else {
        material.setDiffuseColor(ofFloatColor::blue);
    }

    float dt = ofGetElapsedTimef() - time_prev;
    //cout << dt << endl;
    glm::vec3 pos = icoSphere.getPosition();
    pos.x += velocity.x * dt;
    pos.y += velocity.y * dt;
    pos.z += velocity.z * dt;
    icoSphere.setPosition(pos);
    time_prev += dt;
}

//--------------------------------------------------------------
void ofApp::draw(){
    // カメラの開始
	cam.begin();
    // デプステスト（奥行の重なりの計算）
    ofEnableDepthTest();
    // ライティング有効
    ofEnableLighting();
    // 点光源有効
    pointLight.enable();
    // グリッド描画
    ofDrawGrid(5, 10, true, false, true, true);
    // 材料有効
    material.begin();
    // 球の描画
    icoSphere.draw();
    // 材料無効
    material.end();
    // デプステスト無効
    ofDisableDepthTest();
    // カメラ終了
	cam.end();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'a') {
        velocity = ofVec3f(-v, 0, 0);
        grab = false;
    }
    if (key == 's') {
        velocity = ofVec3f(v, 0, 0);
        grab = false;
    }
    if (key == 'w') {
        velocity = ofVec3f(0, v, 0);
        grab = false;
    }
    if (key == 'z') {
        velocity = ofVec3f(0, -v, 0);
        grab = false;
    }
    if (key == 'u') {
        velocity = ofVec3f(0, 0, -v);
        grab = false;
    }
    if (key == 'j') {
        velocity = ofVec3f(0, 0, v);
        grab = false;
    }
    if (key == 'q') {
        velocity = ofVec3f(0, 0, 0);
        grab = false;
    }
    if (key == 'g') {
        velocity = ofVec3f(0, 0, 0);
        grab = true;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}