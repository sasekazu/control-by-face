#pragma once

#include "ofMain.h"
#include <thread>

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		ofEasyCam cam; // add mouse controls for camera movement
		ofLight pointLight;
		ofMaterial material;
		ofIcoSpherePrimitive icoSphere;
		float time_prev = 0.0f;

		unique_ptr<thread> thread_face;
};