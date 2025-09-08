#include "m2s2/ecal/deserializer/deserializer_bme280.hpp"
#include <fstream>
#include <json.hpp>


using json = nlohmann::json;

int main(int argc, char** argv){

    //std::string meas_path = "/home/orin/ecal_meas/2022-10-27_14-49-54.023_measurement";
    //std::string channel_name = "rt/enviro_data";
    //std::string out_file_name = "test_enviro.json";

    std::string meas_path = argv[1];
    std::string channel_name = argv[2];
    std::string out_file_name = argv[3];

    m2s2::ecal::deserializer::DeserializerBME280 bme280_deserializer(meas_path, channel_name, out_file_name);
    bme280_deserializer.process_all();

    return 0;
}

namespace m2s2{ namespace ecal{ namespace deserializer{

    DeserializerBME280::DeserializerBME280(std::string meas_path, std::string channel_name, std::string out_file_name) :
        Deserializer(meas_path, channel_name),
        out_file_name(out_file_name)
  
    {
        //open json file to write to 
        this->o.open(this->out_file_name);
        std::cout << "Created JSON file to write to" << std::endl;

    }   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    struct BaseMsg DeserializerBME280::deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {   

        struct EnviroData* msg = &this->msg;
        
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
        
        // Copy this->msg context to an EnviroData struct
        int ptr = 0;
        std::memcpy(&this->msg.timestamp_sec, &data[ptr], sizeof(this->msg.timestamp_sec));
        std::cout << "Timestamp secs: " <<  this->msg.timestamp_sec << std::endl;

        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        std::cout << "Timestamp nanosecs: " <<  this->msg.timestamp_nanosec << std::endl;

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
        std::memcpy(&this->msg.temperature, &data[ptr], sizeof(this->msg.temperature));
        std::cout << "Temperature: " <<  this->msg.temperature << std::endl;

        ptr += sizeof(this->msg.temperature);
        std::memcpy(&this->msg.pressure, &data[ptr], sizeof(this->msg.pressure));
        std::cout << "Pressure: " <<  this->msg.pressure << std::endl;

        ptr += sizeof(this->msg.pressure);
        std::memcpy(&this->msg.humidity, &data[ptr], sizeof(this->msg.humidity));
        std::cout << "Humidity: " <<  this->msg.humidity << std::endl;

        return this->msg;
    }

    void DeserializerBME280::process_message(struct BaseMsg* msg_){
        // Save Enviro data in a JSON file
        
        // Populate json object
        json j;
        j["frame_id"]           = this->msg.ID;
        j["timestamp_sec"]      = this->msg.timestamp_sec;
        j["timestamp_nanosec"]  = this->msg.timestamp_nanosec;
        j["temperature"]        = this->msg.temperature;
        j["pressure"]           = this->msg.pressure;
        j["humidity"]           = this->msg.humidity;

        // Write to json 
        this->o << std::setw(4) << j << std::endl;
        std::cout << "All measurements written to json file." << std::endl;

    }

}}} // namespace m2s2, ecal, deserializer