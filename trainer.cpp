#include "training_data_raw.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {
    TrainingDataRaw data_raw;
    load("out.cereal", data_raw);
    plt::plot({1,3,2,4});
    plt::show();
}
