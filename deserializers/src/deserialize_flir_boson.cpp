#include "m2s2/ecal/deserializer/deserializer_boson.hpp"

int main(int argc, char** argv){

    //std::string meas_path = "/home/orin/ecal_meas/2022-10-26_13-09-34.473_measurement";
    //std::string channel_name = "rt/thermal_raw";
    //std::string out_path_raw = "/home/orin/m2s2_ecal_deserializers/test_boson/raw/";
    //std::string out_path_rgb = "/home/orin/m2s2_ecal_deserializers/test_boson/rgb/";


    std::string meas_path =     argv[1];
    std::string channel_name =  argv[2];
    std::string out_path_raw =  argv[3];
    std::string out_path_rgb =  argv[4];

    m2s2::ecal::deserializer::DeserializerBoson flir_boson_deserializer(meas_path, channel_name, out_path_raw, out_path_rgb);

    flir_boson_deserializer.process_all();

    return 0;
}

namespace m2s2{ namespace ecal{ namespace deserializer{

    DeserializerBoson::DeserializerBoson(std::string meas_path, std::string channel_name, std::string out_path_raw, std::string out_path_rgb) :
        Deserializer(meas_path, channel_name),
        out_path_raw(out_path_raw),
        out_path_rgb(out_path_rgb)
    {
        std::cout << "GOT BOSON MEASUREMENTS " << std::endl;
    }

    struct BaseMsg DeserializerBoson::deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {   

        struct ThermalRaw* msg = &this->msg;
        
        size_t entry_size;
        if (!meas_->GetEntryDataSize(ID, entry_size)){
            std::cout << "Error Getting Entry Data Size" << std::endl;
            return this->msg;
        }
        
        uint8_t *data = new uint8_t[entry_size];
        if (!meas_->GetEntryData(ID, data)){
            std::cout << "Problem getting entry data: " << ID << std::endl;
            return this->msg;
        }
        
        // Copy this->msg context to a ThermalRaw struct
        int ptr = 0;
        std::memcpy(&(this->msg.timestamp_sec), &data[ptr], sizeof((this->msg.timestamp_sec)));
        std::cout << "Timestamp secs: " <<  (this->msg.timestamp_sec) << std::endl;
        //std::cout << "PTR: " <<  ptr << std::endl;

        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  this->msg.timestamp_nanosec << std::endl;
        //std::cout << "PTR: " <<  ptr << std::endl;

        ptr += sizeof(this->msg.timestamp_nanosec);

        // Frame ID
        uint64_t size_of_frameid;
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        //std::cout << "PTR: " <<  ptr << std::endl;

        ptr += sizeof(size_of_frameid);
        uint8_t *frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s((const char*)&frame_id_cstring, size_of_frameid);
        this->msg.ID = s;
        std::cout << "Frame ID: " <<  this->msg.ID << std::endl;
        
        ptr += size_of_frameid;
        std::memcpy(&this->msg.height, &data[ptr], sizeof(this->msg.height));
        std::cout << "Height: " <<  this->msg.height << std::endl;

        ptr += sizeof(this->msg.height);
        std::memcpy(&this->msg.width, &data[ptr], sizeof(this->msg.width));
        std::cout << "Width: " <<  this->msg.width << std::endl;
        
        ptr += sizeof(this->msg.width);
        std::memcpy(&this->msg.data_size, &data[ptr], sizeof(this->msg.data_size));
        std::cout << "Data Size: " <<  this->msg.data_size << std::endl;

        ptr += sizeof(this->msg.data_size);
        this->msg.data = (uint16_t*)malloc((size_t)this->msg.data_size);
        std::memcpy(&this->msg.data[0], &data[ptr], 2*(size_t)(this->msg.data_size));    
      
        std::cout << "DONE DESERIALIZING" << std::endl;
        return this->msg;
    }

    void DeserializerBoson::process_message(struct BaseMsg* msg_){
        
        
        // Thermal Image Processing
        // 1. Convert data to cv matrix to store as 16 bit .png image
        std::cout << std::endl << "Processing Boson Image" << std::endl;
        cv::Mat raw_16 = cv::Mat(this->msg.height, this->msg.width, CV_16U, this->msg.data);
        std::string timestr = std::to_string(this->msg.timestamp_sec) + "_" + std::to_string(this->msg.timestamp_nanosec);
        std::string img_name_raw = this->out_path_raw + this->msg.ID + "_" + timestr + ".png";
        cv::imwrite(img_name_raw, raw_16);
        std::cout << "Saved Raw Image" << std::endl;

        // 2. Normalise 16_bit data to store as grayscale image .jpeg
        cv::Mat raw_8 = cv::Mat(this->msg.height, this->msg.width, CV_8U, this->msg.data);
        cv::normalize(raw_16, raw_16, 0, 65535, cv::NORM_MINMAX);
        raw_16.convertTo(raw_8, CV_8U, 1./256.);
        std::string img_name_rgb =  this->out_path_rgb + this->msg.ID + "_" + timestr + ".jpeg";
        cv::imwrite(img_name_rgb, raw_8);
        std::cout << "Saved False Colour Image" << std::endl;
        

    }

}}} // namespace m2s2, ecal, deserializer