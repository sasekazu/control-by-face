#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <dlib/image_processing/frontal_face_detector.h>

struct TrainingDataRaw
{
    std::vector<int> labels;
    std::vector<std::vector<int>> data;

    template <class Archive>
    void serialize(Archive &archive) {
        archive(labels, data);
    }
};

void AddData(TrainingDataRaw& data_raw, int label, const dlib::full_object_detection& shape){
    std::vector<int> parts;
    data_raw.labels.push_back(label);
    for(unsigned int i=0; i<shape.num_parts(); ++i) {
        parts.push_back(shape.part(i).x());
        parts.push_back(shape.part(i).y());
    }
    data_raw.data.push_back(parts);
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
    std::ifstream ifs(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(ifs); 
    iarchive(data_raw); 
}
