#include "training_data_raw.h"
#include "matplotlibcpp.h"
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;
namespace plt = matplotlibcpp;

// typedef matrix<std::vector<int>> sample_type;
typedef matrix<double,136,1> sample_type;
// typedef matrix<double,2,1> sample_type;

int main() {
    try{
        TrainingDataRaw data_raw;
        load("out.cereal", data_raw);

        // for(int i=0; i<data_raw.labels.size(); ++i) {
        //     std::vector<int> x = GetX(data_raw, i);
        //     std::vector<int> y = GetY(data_raw, i);
        //     std::vector<int> labels(x.size());
        //     std::fill(labels.begin(), labels.end(), data_raw.labels[i]);
        //     plt::scatter_colored(x, y, labels);
        // }
        // plt::show();

        typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
        // typedef svm_multiclass_linear_trainer<any_trainer<sample_type> > ovo_trainer;
        ovo_trainer trainer;
        
        typedef radial_basis_kernel<sample_type> rbf_kernel;
        krr_trainer<rbf_kernel> rbf_trainer;
        rbf_trainer.set_kernel(rbf_kernel(1.0));
        trainer.set_trainer(rbf_trainer);

        std::vector<sample_type> samples = GetPartsMatrix(data_raw);
        std::vector<double> labels = ConvertToDouble(data_raw.labels);
        cout << "Num data: " << labels.size() << endl;
        randomize_samples(samples, labels);
        cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 10) << endl;
    }
    catch(std::exception& e){
        std::cout << "Error at main(): " << e.what() << std::endl;
    }
}
