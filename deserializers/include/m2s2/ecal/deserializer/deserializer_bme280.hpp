#ifndef __DESERIALIZER_BME280__
#define __DESERIALIZER_BME280__

#include "m2s2/ecal/deserializer/ideserializer.hpp"
#include "json.hpp"
#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <cstring>

namespace m2s2{ namespace ecal{ namespace deserializer{

class DeserializerBME280 : public m2s2::ecal::deserializer::Deserializer
{
    public:
    DeserializerBME280(
        std::string meas_path,
        std::string channel_name,
        std::string out_file_name
    );

    ~DeserializerBME280() = default;
    
    virtual struct BaseMsg deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );

    virtual void process_message(struct BaseMsg* msg);
    
    private:
    std::string out_file_name;
    struct EnviroData msg;
    std::ofstream o;

};

}}} // namespaces

#endif