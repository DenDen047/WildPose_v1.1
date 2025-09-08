#ifndef __DESERIALIZER_LIVOX_IMU__
#define __DESERIALIZER_LIVOX_IMU__

#include "m2s2/ecal/deserializer/ideserializer.hpp"
#include <json.hpp>
using json = nlohmann::json;

#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <cstring>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}


namespace m2s2{ namespace ecal{ namespace deserializer{

class DeserializerLivoxIMU : public m2s2::ecal::deserializer::Deserializer
{
public:
    DeserializerLivoxIMU(
        std::string meas_path,
        std::string channel_name,   // default: rt/livox/lidar
        std::string out_path
    );

    ~DeserializerLivoxIMU();

    virtual struct BaseMsg dry_run(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );
    virtual struct BaseMsg deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );

    virtual void process_message(struct BaseMsg* msg);

private:
    std::string out_path;
    struct IMU msg;
    std::ofstream output_json;
    // temporal variables
    uint8_t *frame_id_cstring;
    uint8_t *pointField_tmp;
    uint8_t *name_string;
    json json_array;
};

}}} // namespaces

#endif