#ifndef __DESERIALIZER_XIMEA__
#define __DESERIALIZER_XIMEA__

#include "m2s2/ecal/deserializer/ideserializer.hpp"

#include <m3api/xiApi.h>

#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}
#define XI_CARE(func) {XI_RETURN ret=func;char msg[200]="";if (XI_OK!=ret) {sprintf(msg,"Error on " #func " - expected XI_OK. Result:%d\n",ret);throw_line(msg);};}
#define SIZE_OF_CONTEXT_BUFFER (10*1024*1024) // 10MiB

namespace m2s2{ namespace ecal{ namespace deserializer{

class DeserializerXimea : public m2s2::ecal::deserializer::Deserializer
{
public:
    DeserializerXimea(
        std::string meas_path,
        std::string channel_name,
        std::string cam_context_path,
        std::string out_path_rgb,
        std::string out_path_raw
    );

    ~DeserializerXimea();

    virtual struct BaseMsg dry_run(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );
    virtual struct BaseMsg deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    );

    virtual void process_message(struct BaseMsg* msg);

    void read_bin_file(std::string src_path, char* dest, int size);

private:
    std::string cam_context_path, out_path_rgb, out_path_raw;
    HANDLE camh;
    char* cam_context;
    xiProcessingHandle_t proc;
    struct Image msg;
    uint8_t *frame_id_cstring;
    uint8_t *encoding_cstring;
};

}}} // namespaces

#endif