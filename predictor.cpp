#include "training_data_raw.h"
#include "matplotlibcpp.h"
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;
namespace plt = matplotlibcpp;

typedef matrix<double,136,1> sample_type;

int main() {
    try{
        typedef linear_kernel<sample_type> lin_kernel;
        multiclass_linear_decision_function<lin_kernel,double> df;
        deserialize("decision_function.dat") >> df;
    }
    catch(std::exception& e){
        std::cout << "Error at main(): " << e.what() << std::endl;
    }
}
