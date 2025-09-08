//LIDAR
#include <ecalhdf5/eh5_meas.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

//pcl stuff (needed to save to pcd, but had trouble installing (https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html) before I left for holiday. Will resume when I come back.)
//#include <pcl/PCLPointCloud2.h>
//#include <pcl/conversions.h>

//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}


int main(int argc, char** argv)
{
    std::string meas_path = "/home/orin/ecal_meas/2022-12-05_11-21-44.937_measurement";
    std::string channel_name = "rt/livox/lidar";
    
    eCAL::eh5::HDF5Meas livox_meas_(meas_path);

    // Check file status
    if (!livox_meas_.IsOk()){
        std::cout << "Problem with measurement file." << std::endl;
        return 0;
    }

    // --------------------
    // Metadata
    // --------------------

    // List Channels Available
    auto channels = livox_meas_.GetChannelNames();
    std::cout << "Channels Available: ";
    for (auto it = channels.begin(); it!=channels.end();it++)
    {
        std::cout<<std::endl<<*it;
    }
    std::cout<<std::endl;
    
    // Entry Info
    eCAL::eh5::EntryInfoSet entry_info_set;
    if (!livox_meas_.GetEntriesInfo(channel_name, entry_info_set)){
        std::cout << "Problem retrieving entries info" << std::endl;
        return 0;
    }
    std::cout << "Entries Info Retrieved Successfully!" << std::endl;
    
    for (auto it = entry_info_set.begin(); it!=entry_info_set.end();it++)
    {   
        size_t entry_size;
        if (!livox_meas_.GetEntryDataSize(it->ID, entry_size))
            return 0;
        uint8_t * data = new uint8_t[entry_size];
        if (!livox_meas_.GetEntryData(it->ID, data)){
            std::cout << "Problem getting entry data: " << it->ID << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
        //std::cout << "Retrieved Data: "<< std::endl;
        //std::cout << "ID: " << it->ID << std::endl;
    
        
        // Split data into correct format based on Lidar struct in msgs.hpp
        
        uint64_t size_of_frameid;

        std::string ID = "000000000000";

        uint32_t timestamp_sec;
        uint32_t timestamp_nanosec;

        uint32_t height;
        uint32_t width;

        long pf_arr_size;

        uint8_t is_bigendian;
        uint32_t point_step;
        uint32_t row_step;

        long data_size;

        uint8_t is_dense;


        int ptr = 0;
        
        // create PCL pointcloud2  
        //pcl::PCLPointCloud2 pcl_pc2;

        std::memcpy(&timestamp_sec, &data[ptr], sizeof(timestamp_sec));
        std::cout << "Timestamp secs: " <<  timestamp_sec << std::endl;

        ptr += sizeof(timestamp_sec);
        std::memcpy(&timestamp_nanosec, &data[ptr], sizeof(timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  timestamp_nanosec << std::endl;

        ptr += sizeof(timestamp_nanosec);
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        //std::cout << "Size of frame id: " <<  size_of_frameid << std::endl;
        
        ptr += sizeof(size_of_frameid);
        uint8_t *frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s = (const char*)&frame_id_cstring;
        ID = s;
        std::cout << "Frame ID: " <<  ID << std::endl;

        //PCL HEADER
        //pcl::PCLHeader pcl_header
        //pcl_header.stamp = static_cast<std::uint64_t>(timestamp_nanosec) / 1000ull;
        //pcl_header.seq = 0;
        //pcl_header.frame_id = ID;
        //pcl_pc2.header = pcl_header;

        ptr += size_of_frameid;
        std::memcpy(&height, &data[ptr], sizeof(height));
        std::cout << "Height: " <<  height << std::endl;

        ptr += sizeof(height);
        std::memcpy(&width, &data[ptr], sizeof(width));
        std::cout << "Width: " <<  width << std::endl;
        
        ptr += sizeof(width);
        std::memcpy(&pf_arr_size, &data[ptr], sizeof(pf_arr_size));
        //std::cout << "PointField Array Size: " <<  pf_arr_size << std::endl;

        ptr += sizeof(pf_arr_size);
        uint8_t *pointField_tmp = (uint8_t*)malloc((size_t)(121));
        std::memcpy(&pointField_tmp[0], &data[ptr], (size_t)(121));   
        
        //PCL METADATA
        //pcl_pc2.height = height;
        //pcl_pc2.width = width;

        //PointField array
        int pf_ptr = 0;
        std::cout << "Fields: " << std::endl;

        //PCL FIELDS
        //std::vector<pcl::PCLPointField> pcl_pfs
        
        for (int i=0; i<6; i++){
            //pcl::PCLPointField pcl_pf
            std::string name = "0000000000";
            uint64_t name_size;
            uint32_t offset;
            uint8_t datatype;
            uint32_t count; 

            std::memcpy(&name_size, &pointField_tmp[pf_ptr], sizeof(name_size));
            //std::cout << "Size of name: " << name_size  << std::endl;

            pf_ptr += sizeof(name_size);
            uint8_t *name_string = (uint8_t*)malloc((size_t)name_size);
            std::memcpy(&name_string, &pointField_tmp[pf_ptr], sizeof(name_string));
            std::string n = (const char*)&name_string;
            name = n;
            std::cout << "- Name: " <<  name << std::endl;

            pf_ptr += name_size;
            std::memcpy(&offset, &pointField_tmp[pf_ptr], sizeof(offset));
            std::cout << "  Offset: " <<  offset << std::endl;
 
            pf_ptr += sizeof(offset);
            std::memcpy(&datatype, &pointField_tmp[pf_ptr], sizeof(datatype));
            std::cout << "  Datatype: " <<  int(datatype) << std::endl;

            pf_ptr += sizeof(datatype);
            std::memcpy(&count, &pointField_tmp[pf_ptr], sizeof(count));
            std::cout << "  Count: " <<  count << std::endl;

            pf_ptr += sizeof(count);

            //pcl_pf.name = name;
            //pcl_pf.offset = offset;
            //pcl_pf.datatype = datatype;
            //pcl_pf.count = count;
            //pcl_pfs.push_back(pcl_pf);

        }
        
        //pcl_pc2.fields = pcl_pfs;

        ptr += size_t(121);
        std::memcpy(&is_bigendian, &data[ptr], sizeof(is_bigendian));
        std::cout << "Is Bigendian?: " <<  int(is_bigendian) <<  std::endl;

        ptr += sizeof(is_bigendian);
        std::memcpy(&point_step, &data[ptr], sizeof(point_step));
        std::cout << "Point Step: " <<  point_step << std::endl;

        ptr += sizeof(point_step);
        std::memcpy(&row_step, &data[ptr], sizeof(row_step));
        std::cout << "Row Step: " <<  row_step << std::endl;

        ptr += sizeof(row_step);
        std::memcpy(&data_size, &data[ptr], sizeof(data_size));
        //std::cout << "Data Size: " <<  data_size << std::endl;

        ptr += sizeof(data_size);
        //std::cout << "Ptr: " <<  ptr << std::endl;
        uint8_t *data_tmp = (uint8_t*)malloc((size_t)data_size);
        std::memcpy(&data_tmp[0], &data[ptr], (size_t)(data_size));

        ptr += data_size;
        std::memcpy(&is_dense, &data[ptr], sizeof(is_dense));
        std::cout << "Is Dense?: " <<  int(is_dense) << std::endl;

        // PCL DATA
        //pcl_pc2.is_bigendian = is_bigendian;
        //pcl_pc2.point_step = point_step;
        //pcl_pc2.row_step = row_step;
        //pcl_pc2.is_dense = is_dense;
        //pcl_pc2.data = data_tmp;

        //save to pcd
        //pcl_conversions::toPCL(pc2, pcl_pc2); not needed
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        //pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
        //std::string timestr = timestamp_sec + "_" + timestamp_nanosec;
        //std::string pcd_name = ("/home/orin/m2s2_ecal_deserializers/test_lidar/" + timestr + ".pcd");
        //pcl::io::savePCDFileASCII(pcd_name, *temp_cloud);
        
    }




    std::cout<<std::endl;
    std::cout << "Done Processing Measurement" << std::endl;
    return 0;
} 


/*// ENVIRO
#include <ecalhdf5/eh5_meas.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}

using json = nlohmann::json;
struct EnviroData {
    std::string ID = "123456";
    uint32_t timestamp_sec;
    uint32_t timestamp_nanosec;
    double temperature;
    double pressure;
    double humidity;
};

// Declarations

int main(int argc, char** argv)
{
    std::string meas_path = "/home/orin/ecal_meas/2022-11-15_13-10-27.417_measurement";
    std::string channel_name = "rt/radar_data";
    
    //open json file to write to 
    std::ofstream o("radar1.json");    
    
    eCAL::eh5::HDF5Meas enviro_meas_(meas_path);

    // Check file status
    if (!enviro_meas_.IsOk()){
        std::cout << "Problem with measurement file." << std::endl;
        return 0;
    }
    // --------------------
    // Metadata
    // --------------------

    // List Channels Available
    auto channels = enviro_meas_.GetChannelNames();
    std::cout << "Channels Available: ";
    for (auto it = channels.begin(); it!=channels.end();it++)
    {
        std::cout<<std::endl<<*it;
    }
    std::cout<<std::endl;
    
    // Entry Info
    eCAL::eh5::EntryInfoSet entry_info_set;
    if (!enviro_meas_.GetEntriesInfo(channel_name, entry_info_set)){
        std::cout << "Problem retrieving entries info" << std::endl;
        return 0;
    }
    std::cout << "Entries Info Retrieved Successfully!" << std::endl;
    
    for (auto it = entry_info_set.begin(); it!=entry_info_set.end();it++)
    {   
        size_t entry_size;
        if (!enviro_meas_.GetEntryDataSize(it->ID, entry_size))
            return 0;
        uint8_t * data = new uint8_t[entry_size];
        if (!enviro_meas_.GetEntryData(it->ID, data)){
            std::cout << "Problem getting entry data: " << it->ID << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Retrieved Data: "<< std::endl;
        std::cout << "ID: " << it->ID << std::endl;

        // Split data into correct format based on Image struct in msgs.hpp

        uint64_t size_of_frameid;

        std::string ID = "123456";

        uint32_t timestamp_sec;
        uint32_t timestamp_nanosec;
        uint32_t frame_size;
        
        int ptr = 0;
        
        std::memcpy(&timestamp_sec, &data[ptr], sizeof(timestamp_sec));
        std::cout << "Timestamp secs: " <<  timestamp_sec << std::endl;
        ptr += sizeof(timestamp_sec);

        std::memcpy(&timestamp_nanosec, &data[ptr], sizeof(timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  timestamp_nanosec << std::endl;
        ptr += sizeof(timestamp_nanosec);

        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        ptr += sizeof(size_of_frameid);

        uint8_t *frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s = (const char*)&frame_id_cstring;
        ID = s;
        std::cout << "Frame ID: " <<  ID << std::endl;
        ptr += size_of_frameid;

        std::memcpy(&frame_size, &data[ptr], sizeof(frame_size));
        std::cout << "Frame Size: " <<  frame_size << std::endl;
        ptr += sizeof(frame_size);

        std::vector<uint8_t> frame_data(data + ptr,data+ ptr + frame_size);
        std::cout << "DATA tmp size: " << sizeof(frame_data) << std::endl;

        //std::memcpy(&frame_data, &data[ptr], (size_t)(frame_size));
        //std::cout << "Frame Data: " << frame_data <<std::endl;
    
        // SAVE RADAR DATA IN JSON
        json j;
        j["timestamp_secs"] = timestamp_sec;
        j["timestamp_nanosecs"] = timestamp_nanosec;
        j["frame_id"] = ID;
        j["frame_size"] = frame_size;
        j["frame_data"] = frame_data;

        // write 
        std::ofstream o("radar1.json");    
        o << std::setw(4) << j << std::endl; 
        
    }

    std::cout<<std::endl;
    std::cout << "Done Processing Measurement" << std::endl;
    return 0;
} 

*/

/* // BOSON
#include <ecalhdf5/eh5_meas.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}

//using namespace cv;
// Declarations

int main(int argc, char** argv)
{
    std::string meas_path = "/home/aru/ecal_meas/2022-10-26_13-09-34.473_measurement";
    std::string channel_name = "rt/thermal_raw";
    int height = 512, width = 640;
    
    eCAL::eh5::HDF5Meas thermal_meas_(meas_path);

    // Check file status
    if (!thermal_meas_.IsOk()){
        std::cout << "Problem with measurement file." << std::endl;
        return 0;
    }

    // --------------------
    // Metadata
    // --------------------

    // List Channels Available
    auto channels = thermal_meas_.GetChannelNames();
    std::cout << "Channels Available: ";
    for (auto it = channels.begin(); it!=channels.end();it++)
    {
        std::cout<<std::endl<<*it;
    }
    std::cout<<std::endl;
    
    // Entry Info
    eCAL::eh5::EntryInfoSet entry_info_set;
    if (!thermal_meas_.GetEntriesInfo(channel_name, entry_info_set)){
        std::cout << "Problem retrieving entries info" << std::endl;
        return 0;
    }
    std::cout << "Entries Info Retrieved Successfully!" << std::endl;
    
    for (auto it = entry_info_set.begin(); it!=entry_info_set.end();it++)
    {   
        size_t entry_size;
        if (!thermal_meas_.GetEntryDataSize(it->ID, entry_size))
            return 0;
        uint8_t * data = new uint8_t[entry_size];
        if (!thermal_meas_.GetEntryData(it->ID, data)){
            std::cout << "Problem getting entry data: " << it->ID << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Retrieved Data: "<< std::endl;
        std::cout << "ID: " << it->ID << std::endl;
    
        
        // Split data into correct format based on Image struct in msgs.hpp

        uint64_t size_of_frameid;
        uint64_t encoding_bytes;

        std::string ID = "some";

        uint32_t timestamp_sec;
        uint32_t timestamp_nanosec;

        uint16_t height;
        uint16_t width;

        long data_size;

        int ptr = 0;
        
        std::memcpy(&timestamp_sec, &data[ptr], sizeof(timestamp_sec));
        std::cout << "Timestamp secs: " <<  timestamp_sec << std::endl;

        ptr += sizeof(timestamp_sec);
        std::memcpy(&timestamp_nanosec, &data[ptr], sizeof(timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  timestamp_nanosec << std::endl;

        ptr += sizeof(timestamp_nanosec);
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        //std::cout << "Size of frame id: " <<  size_of_frameid << std::endl;

        ptr += sizeof(size_of_frameid);
        uint8_t *frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s = (const char*)&frame_id_cstring;
        ID = s;
        std::cout << "Frame ID: " <<  ID << std::endl;
    
        ptr += size_of_frameid;
        std::memcpy(&height, &data[ptr], sizeof(height));
        std::cout << "Height: " <<  height << std::endl;

        ptr += sizeof(height);
        std::memcpy(&width, &data[ptr], sizeof(width));
        std::cout << "Width: " <<  width << std::endl;
        
        ptr += sizeof(width);
        std::memcpy(&data_size, &data[ptr], sizeof(data_size));
        std::cout << "Data Size: " <<  data_size << std::endl;

        ptr += sizeof(data_size);
        uint16_t *data_tmp = (uint16_t*)malloc((size_t)(data_size));
        std::memcpy(&data_tmp[0], &data[ptr], 2*(size_t)(data_size));    
        std::cout << "SIZE: " << (size_t)(data_size) << std::endl;

        //THERMAL IMAGE PROCESSING 
        //1. Convert data to cv matrix to store as 16 bit .png image
        cv::Mat raw_16 = cv::Mat(height, width, CV_16U, data_tmp);
        std::cout << "DATA STORED IN MAT " << std::endl;
        std::string timestr = std::to_string(timestamp_sec) + "_" + std::to_string(timestamp_nanosec);
        std::string img_name_raw = "/home/aru/repos/ximea_stuff/ecal_ximea/img/out/" + ID + "_" + timestr + ".png";
        cv::imwrite(img_name_raw, raw_16);
        std::cout << "Saved Image" << std::endl;

        //2. Normalise 16_bit data to store as grayscale image .jpeg
        cv::Mat raw_8 = cv::Mat(height, width, CV_8U, data_tmp);
        cv::normalize(raw_16, raw_16, 0, 65535, cv::NORM_MINMAX);
        raw_16.convertTo(raw_8, CV_8U, 1./256.);

        std::string img_name_rgb = "/home/aru/repos/ximea_stuff/ecal_ximea/img/rgb/" + ID + "_" + timestr + ".jpeg";
        cv::imwrite(img_name_rgb, raw_8);
        std::cout << "Saved False Colour Image" << std::endl;
        
    }

    std::cout<<std::endl;
    std::cout << "Done Processing Measurement" << std::endl;
    return 0;
} 
*/

/*
// XIMEA
    //std::string meas_path = "/home/orin/ecal_meas/2022-10-19_17-16-00.841_measurement";
    //std::string cam_context_path = "/home/aru/repos/ximea_stuff/ecal_ximea/img/cam_context.bin";
    //std::string channel_name = "rt/image_raw";
    //std::atring out_path = "/home/orin/m2s2_ecal_deserializers/test_ximea/";
#include <ecalhdf5/eh5_meas.h>

#include <m3api/xiApi.h>

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/core.hpp>

#define throw_line(message) {char err[500]="";sprintf(err,"Fatal error: %s in:" __FILE__ "line:%d\n",message,__LINE__);throw err;}
#define THROW_IF_ZERO(val) {if (!val) throw_line("Value "#val"is zero");}
#define XI_CARE(func) {XI_RETURN ret=func;char msg[200]="";if (XI_OK!=ret) {sprintf(msg,"Error on " #func " - expected XI_OK. Result:%d\n",ret);throw_line(msg);};}

// Declarations
void read_bin_file(std::string src_path, char* dest, int size);

int main(int argc, char** argv)
{
    std::string meas_path = "/home/orin/ecal_meas/2022-10-19_17-16-00.841_measurement";
    std::string cam_context_path = "/home/aru/repos/ximea_stuff/ecal_ximea/img/cam_context.bin";
    std::string channel_name = "rt/image_raw";
    int height = 1088, width = 2048;
    
    eCAL::eh5::HDF5Meas xi_meas_(meas_path);

    // Check file status
    if (!xi_meas_.IsOk()){
        std::cout << "Problem with measurement file." << std::endl;
        return 0;
    }

    // --------------------
    // Metadata
    // --------------------

    // List Channels Available
    auto channels = xi_meas_.GetChannelNames();
    std::cout << "Channels Available: ";
    for (auto it = channels.begin(); it!=channels.end();it++)
    {
        std::cout<<std::endl<<*it;
    }
    std::cout<<std::endl;

    // Entry Info
    eCAL::eh5::EntryInfoSet entry_info_set;
    if (!xi_meas_.GetEntriesInfo(channel_name, entry_info_set)){
        std::cout << "Problem retrieving entries info" << std::endl;
        return 0;
    }
    std::cout << "Entries Info Retrieved Successfully!" << std::endl;

    // Ximea Setup
    HANDLE camh = NULL;
    char* cam_context=NULL;
    #define SIZE_OF_CONTEXT_BUFFER (10*1024*1024) // 10MiB
    cam_context = (char*)malloc(SIZE_OF_CONTEXT_BUFFER);

    read_bin_file(cam_context_path, cam_context, SIZE_OF_CONTEXT_BUFFER);
    
    std::string s(cam_context);
    std::cout << "GOT CAM CONTEXT: " << std::endl << cam_context << std::endl;

    // Ximea offline processing
    std::cout << "Opening Ximea Offline Processing" << std::endl;
    xiProcessingHandle_t proc;
    XI_CARE(xiProcOpen(&proc));
    std::cout << "Ximea Offline Procesing Opened" << std::endl;
    int cam_context_len = (DWORD)strlen(cam_context);

    XI_CARE(xiProcSetParam(proc, XI_PRM_API_CONTEXT_LIST, cam_context, cam_context_len, xiTypeString));
    XI_IMG_FORMAT output_format = XI_RGB32;
    XI_CARE(xiProcSetParam(proc, XI_PRM_IMAGE_DATA_FORMAT, &output_format, sizeof(output_format), xiTypeInteger));
    
    for (auto it = entry_info_set.begin(); it!=entry_info_set.end();it++)
    {   
        size_t entry_size;
        if (!xi_meas_.GetEntryDataSize(it->ID, entry_size))
            return 0;
        uint8_t * data = new uint8_t[entry_size];
        if (!xi_meas_.GetEntryData(it->ID, data)){
            std::cout << "Problem getting entry data: " << it->ID << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Retrieved Data: "<< std::endl;
        std::cout << "ID: " << it->ID << std::endl;
        //std::cout << "Sender Timestamp: " << it->SndTimestamp << std::endl;
        //std::cout << "Reciever Timestamp: " << it->RcvTimestamp << std::endl;
        //std::cout << "size = " << entry_size << " | " << 
        //        "[" << int(data[0]) << ",...," << int(data[entry_size-1]) << "]" << std::endl;
        
        int actual_size = height * width;
        int size_diff = entry_size-actual_size;
        
        // Split data into correct format based on Image struct in msgs.hpp

        uint64_t size_of_frameid;
        uint64_t encoding_bytes;

        std::string ID = "some";

        uint32_t timestamp_sec;
        uint32_t timestamp_nanosec;

        uint32_t height;
        uint32_t width;

        std::string encoding;
        uint8_t is_bigendian;
        uint32_t step;

        long data_size;
        int ptr = 0;
        std::memcpy(&timestamp_sec, &data[ptr], sizeof(timestamp_sec));
        std::cout << "Timestamp secs: " <<  timestamp_sec << std::endl;

        ptr += sizeof(timestamp_sec);
        std::memcpy(&timestamp_nanosec, &data[ptr], sizeof(timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  timestamp_nanosec << std::endl;

        ptr += sizeof(timestamp_nanosec);
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        //std::cout << "Size of frame id: " <<  size_of_frameid << std::endl;

        ptr += sizeof(size_of_frameid);
        //std::cout << "PTR: " <<  ptr << std::endl;
        uint8_t *frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s((const char*)&frame_id_cstring, size_of_frameid);
        ID = s;
        std::cout << "Frame ID: " <<  ID << std::endl;
            
        ptr += size_of_frameid;
        std::memcpy(&height, &data[ptr], sizeof(height));
        std::cout << "Height: " <<  height << std::endl;

        ptr += sizeof(height);
        std::memcpy(&width, &data[ptr], sizeof(width));
        std::cout << "Width: " <<  width << std::endl;
        
        ptr += sizeof(width);
        std::memcpy(&encoding_bytes, &data[ptr], sizeof(encoding_bytes));
        //std::cout << "Size of encoding: " <<  encoding_bytes << std::endl;

        ptr += sizeof(encoding_bytes);        
        uint8_t *encoding_cstring = (uint8_t*)malloc(encoding_bytes);
        std::memcpy(&encoding_cstring, &data[ptr], sizeof(encoding_cstring));
        encoding = std::string((const char*)&encoding_cstring);
        std::cout << "Encoding: " <<  encoding << std::endl;

        ptr += encoding_bytes;
        std::memcpy(&is_bigendian, &data[ptr], sizeof(is_bigendian));
        std::cout << "Is Bigendian: " <<  is_bigendian << std::endl;

        ptr += sizeof(is_bigendian);
        std::memcpy(&step, &data[ptr], sizeof(step));
        std::cout << "Step: " <<  step << std::endl;

        ptr += sizeof(step);
        std::memcpy(&data_size, &data[ptr], sizeof(data_size));
        std::cout << "Data Size: " <<  data_size << std::endl;

        ptr += sizeof(data_size);
        std::cout << "Ptr: " <<  ptr << std::endl;

        uint8_t *data_tmp = (uint8_t*)malloc((size_t)data_size);
        std::memcpy(&data_tmp[0], &data[ptr], (size_t)(data_size));     

        // Save raw image
        cv::Mat img_mat = cv::Mat(height, width, CV_8UC1, data_tmp);
        std::string timestr = std::to_string(timestamp_sec) + "_" + std::to_string(timestamp_nanosec);
        std::string img_name = "/home/aru/repos/ximea_stuff/ecal_ximea/img/out/" + ID + "_" + timestr + ".tiff";
        cv::imwrite(img_name, img_mat);
        std::cout << "Saved Image" << std::endl;
        
        
        // Process image
        XI_IMG out_image;
        out_image.size = sizeof(XI_IMG);
        //std::cout << "Out image size: " << out_image.size << std::endl;

        out_image.bp = NULL;
        out_image.bp_size = 0;

        XI_CARE(xiProcPushImage(proc, data_tmp));
        XI_CARE(xiProcPullImage(proc, 0, &out_image));
        std::cout << "Ximea Processed Image" << std::endl;

        //std::cout << "image height: " << out_image.height << std::endl;
        //std::cout << "image width: " << out_image.width << std::endl;

        cv::Mat img_mat_rgb = cv::Mat(out_image.height, out_image.width, CV_8UC4, out_image.bp);
        std::cout << "copied image" << std::endl; 
        std::string img_name_rgb = "/home/aru/repos/ximea_stuff/ecal_ximea/img/rgb/" + ID + "_" + timestr + ".jpeg";
        cv::imwrite(img_name_rgb, img_mat_rgb);
        

        delete [] data_tmp;
        delete [] data;
        
        
    }

    //XI_CARE(xiProcClose(proc));

    std::cout<<std::endl;
    std::cout << "Done Processing Measurement" << std::endl;
    return 0;
}

void read_bin_file(std::string src_path, char* dest, int size){
    std::ifstream file_in(src_path, std::ios::in | std::ios::binary);

    file_in.read(dest, size);

    if (!file_in){
        std::cout << "Error Reading File" << std::endl;
    }

} */
