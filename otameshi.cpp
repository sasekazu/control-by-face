#include <dlib/svm_threaded.h>

using namespace dlib;
using namespace std;


// Our data will be 3-dimensional data
typedef matrix<double,3,1> sample_type;

void generate_data (std::vector<sample_type>& samples, std::vector<string>& labels);

// ---------------------------

int main()
{
    std::vector<sample_type> samples;
    std::vector<string> labels;

    // First, get our labeled set of training data
    generate_data(samples, labels);

    // Define kernel
    typedef linear_kernel<sample_type> lin_kernel;

    // Define the SVM multiclass trainer
    typedef svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;
    svm_mc_trainer trainer;
    // Train and obtain the decision rule
    multiclass_linear_decision_function<lin_kernel,string> df = trainer.train(samples,labels);

    // Now lets do 5-fold cross-validation
    randomize_samples(samples, labels);
    cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;

    
    trainer.set_c(50); cout << "cross validation: " << trainer.get_c() << "\n" <<
    cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
    trainer.set_c(100); cout << "cross validation: " << trainer.get_c() << "\n" <<
    cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
    trainer.set_c(500); cout << "cross validation: " << trainer.get_c() << "\n" <<
    cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
    trainer.set_c(1000); cout << "cross validation: " << trainer.get_c() << "\n" <<
    cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;

}

// ---------------------------

void generate_data (std::vector<sample_type>& samples, std::vector<string>& labels)
{
    // We are going to generate 500 data points.
    // Points with an average value > 0.5 belong to the "high" class
    // Otherwise they belong to the "low" class

    int num = 500;

    // initialize random number generator
    dlib::rand rnd(cast_to_string(time(0)));

    sample_type m;
    for (int i = 0; i < num; i++)
    {
            m = randm(3,1, rnd);
            float average = (m(0) + m(1) + m(2)) / 3;
            string label;
            if (average > 0.5)
            {
                label = "high";
            }
            else
            {
                label = "low";
            }
            samples.push_back(m);
            labels.push_back(label);    
    }
}