#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/matrix.h>

struct TrainingDataRaw
{
    std::vector<int> labels;
    std::vector<std::vector<double>> data;

    template <class Archive>
    void serialize(Archive &archive) {
        archive(labels, data);
    }
};


struct CollectedData {
    dlib::full_object_detection shape;
    double top;
    double left;
    double width;
};

std::vector<double> PartToVector(const CollectedData& d){
    std::vector<double> parts;
    for(unsigned int i=0; i<d.shape.num_parts(); ++i) {
        parts.push_back((d.shape.part(i).x() - d.left)/d.width);
        parts.push_back((d.shape.part(i).y() - d.top)/d.width);
    }
    return parts;
}

dlib::matrix<double,136,1> PartToMatrix(const CollectedData& d){
    dlib::matrix<double,136,1> parts;
    for(unsigned int i=0; i<d.shape.num_parts(); ++i) {
        parts(i*2) = (d.shape.part(i).x() - d.left)/d.width;
        parts(i*2+1) = (d.shape.part(i).y() - d.top)/d.width;
    }
    return parts;
}

void AddData(TrainingDataRaw& data_raw, int label, const CollectedData& d){
    data_raw.labels.push_back(label);
    data_raw.data.push_back(PartToVector(d));
}

void PrintData(const TrainingDataRaw& data_raw){
    for(unsigned int i=0; i<data_raw.labels.size(); ++i) {
        std::cout << "Data " << i << "----------------------" << std::endl;
        std::cout << "Label: " << data_raw.labels[i] << std::endl;
        std::cout << "Data: ";
        for(unsigned int j=0; j<data_raw.data[i].size(); ++j) {
            std::cout << data_raw.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void save(const char* filename, const TrainingDataRaw& data_raw){
    std::ofstream ofs(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(ofs); 
    oarchive(data_raw); 
}

void load(const char* filename, TrainingDataRaw& data_raw){
    try{
        std::ifstream ifs(filename, std::ios::binary);
        cereal::BinaryInputArchive iarchive(ifs); 
        iarchive(data_raw); 
    }
    catch(std::exception& e){
        std::cout << "Error at load(): " << e.what() << std::endl;
    }
}

std::vector<dlib::point> GetParts(const TrainingDataRaw& data_raw, unsigned int index){
    std::vector<dlib::point> parts;
    for(unsigned int i=0; i<data_raw.data[index].size(); i+=2) {
        dlib::point part;
        part(0) = data_raw.data[index][i];
        part(1) = data_raw.data[index][i+1];
        parts.push_back(part);
    }
    return parts;
}

std::vector<int> GetX(const TrainingDataRaw& data_raw, unsigned int index){
    std::vector<int> x;
    for(unsigned int i=0; i<data_raw.data[index].size(); i+=2) {
        x.push_back(data_raw.data[index][i]);
    }
    return x;
}

std::vector<int> GetY(const TrainingDataRaw& data_raw, unsigned int index){
    std::vector<int> y;
    for(unsigned int i=0; i<data_raw.data[index].size(); i+=2) {
        y.push_back(data_raw.data[index][i+1]);
    }
    return y;
}

std::vector<dlib::matrix<double,136,1>> GetPartsMatrix(const TrainingDataRaw& data_raw){
    std::vector<dlib::matrix<double,136,1>> parts;
    for(unsigned int i=0; i<data_raw.data.size(); ++i) {
        dlib::matrix<double,136,1> part;
        for(unsigned int j=0; j<data_raw.data[i].size(); ++j) {
            part(j) = data_raw.data[i].at(j);
        }
        parts.push_back(part);
    }
    return parts;
}

std::vector<double> ConvertToDouble(const std::vector<int>& data){
    std::vector<double> data_double;
    for(unsigned int i=0; i<data.size(); ++i) {
        data_double.push_back(data[i]);
    }
    return data_double;
}