#ifndef __DESERIALIZER_BOSON__
#define __DESERIALIZER_BOSON__

#include "m2s2/ecal/deserializer/ideserializer.hpp"

#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

namespace m2s2{ namespace ecal{ namespace deserializer{

class DeserializerBoson : public m2s2::ecal::deserializer::Deserializer
{
    public:
    DeserializerBoson(
        std::string meas_path,
        std::string channel_name,
        std::string out_path_raw,
        std::string out_path_rgb
    );

    ~DeserializerBoson() = default;
    
    virtual struct BaseMsg deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );

    virtual void process_message(struct BaseMsg* msg);
    
    private:
    struct ThermalRaw msg;
    std::string out_path_raw, out_path_rgb;
};

}}} // namespaces

#endif