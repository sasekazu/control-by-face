#include "training_data_raw.h"
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

typedef matrix<double,136,1> sample_type;

int main() {
    try{
        TrainingDataRaw data_raw;
        load("out.cereal", data_raw);
        std::vector<sample_type> samples = GetPartsMatrix(data_raw);
        std::vector<double> labels = ConvertToDouble(data_raw.labels);
        cout << "Num data: " << labels.size() << endl;

        typedef linear_kernel<sample_type> lin_kernel;
        typedef svm_multiclass_linear_trainer <lin_kernel, double> svm_mc_trainer;
        svm_mc_trainer trainer;        
        multiclass_linear_decision_function<lin_kernel,double> df = trainer.train(samples,labels);

        randomize_samples(samples, labels);
        trainer.set_c(1); // hyper parameter 
        cout << "cross validation: " << trainer.get_c() << "\n" <<
        cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;

        serialize("decision_function.dat") << df;
    }
    catch(std::exception& e){
        std::cout << "Error at main(): " << e.what() << std::endl;
    }
}
