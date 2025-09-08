#ifndef __DESERIALIZER_LIVOX__
#define __DESERIALIZER_LIVOX__

#include "m2s2/ecal/deserializer/ideserializer.hpp"

#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <cstring>

//PCL Includes
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}

namespace m2s2{ namespace ecal{ namespace deserializer{

class DeserializerLivox : public m2s2::ecal::deserializer::Deserializer
{
public:
    DeserializerLivox(
        std::string meas_path,
        std::string channel_name,
        std::string out_path
    );

    ~DeserializerLivox();

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
    struct PointCloud2 msg;
    struct PointField pf_msg;
    // PCL STUFF
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::PCLHeader pcl_header;
    pcl::PCLPointField pcl_pf;
    std::vector<pcl::PCLPointField> pcl_pfs;
    // temporal variables
    uint8_t *frame_id_cstring;
    uint8_t *pointField_tmp;
    uint8_t *name_string;
};

}}} // namespaces

#endif