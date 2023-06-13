#include "training_data_raw.h"
#include "matplotlibcpp.h"

using namespace std;
using namespace dlib;
namespace plt = matplotlibcpp;

int main() {
    TrainingDataRaw data_raw;
    load("out.cereal", data_raw);

    for(int i=0; i<data_raw.labels.size(); ++i) {
        std::vector<int> x = GetX(data_raw, i);
        std::vector<int> y = GetY(data_raw, i);
        plt::scatter(x, y);
        plt::show();
    }
}
