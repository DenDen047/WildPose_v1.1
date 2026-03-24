#include "m2s2/ecal/deserializer/deserializer_livox_imu.hpp"
#include <fstream>
#include <json.hpp>


using json = nlohmann::json;

int main(int argc, char** argv){

    // std::string meas_path = "/home/naoya/data/test/2022-11-30_15-17-43.373_measurement";
    // std::string channel_name = "rt/livox/lidar";
    // std::string out_path = "/home/naoya/data/test/2022-11-30_15-17-43.373_measurement/lidar/";

    std::string meas_path =         argv[1];
    std::string channel_name =      argv[2];
    std::string out_path =          argv[3];

    m2s2::ecal::deserializer::DeserializerLivoxIMU livox_imu_deserializer(meas_path, channel_name, out_path);

    livox_imu_deserializer.process_all();

    return 0;
}

namespace m2s2{ namespace ecal{ namespace deserializer{

    DeserializerLivoxIMU::DeserializerLivoxIMU(std::string meas_path, std::string channel_name, std::string out_path) :
        Deserializer(meas_path, channel_name),
        out_path(out_path)  // filename.json
    {
        this->output_json.open(this->out_path);
        this->json_array = json::array();
    }

    DeserializerLivoxIMU::~DeserializerLivoxIMU () {
        this->output_json << std::setw(4) << this->json_array << std::endl;
    }

    struct BaseMsg DeserializerLivoxIMU::dry_run(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {
        // init
        struct IMU* msg = &this->msg;

        // load the ROS2 topic data
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

        int ptr = 0;
        // --- Header ---
        // Timestamp
        std::memcpy(&this->msg.timestamp_sec, &data[ptr], sizeof(this->msg.timestamp_sec));
        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        ptr += sizeof(this->msg.timestamp_nanosec);

        // Frame ID
        uint64_t size_of_frameid;
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        ptr += sizeof(size_of_frameid);
        this->frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(this->frame_id_cstring, &data[ptr], (size_t)size_of_frameid);
        std::string s((const char*)this->frame_id_cstring, size_of_frameid);
        this->msg.ID = s;
        ptr += size_of_frameid;
        // std::cout << "Frame ID: " <<  this->msg.ID << std::endl;

        // --- Orientation ---
        std::memcpy(&this->msg.orientation, &data[ptr], sizeof(this->msg.orientation));
        ptr += sizeof(this->msg.orientation);
        std::cout << "Orientation_x: " <<  this->msg.orientation[0] << std::endl;
        std::cout << "Orientation_y: " <<  this->msg.orientation[1] << std::endl;
        std::cout << "Orientation_z: " <<  this->msg.orientation[2] << std::endl;
        std::cout << "Orientation_w: " <<  this->msg.orientation[3] << std::endl;


        delete [] data;

        return this->msg;
    }

    struct BaseMsg DeserializerLivoxIMU::deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {
        // init
        struct IMU* msg = &this->msg;

        // load the ROS2 topic data
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

        int ptr = 0;
        // --- Header ---
        // Timestamp
        std::memcpy(&this->msg.timestamp_sec, &data[ptr], sizeof(this->msg.timestamp_sec));
        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        ptr += sizeof(this->msg.timestamp_nanosec);

        // Frame ID
        uint64_t size_of_frameid;
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));
        ptr += sizeof(size_of_frameid);
        this->frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(this->frame_id_cstring, &data[ptr], (size_t)size_of_frameid);
        std::string s((const char*)this->frame_id_cstring, size_of_frameid);
        this->msg.ID = s;
        ptr += size_of_frameid;

        // --- Orientation ---
        std::memcpy(&this->msg.orientation, &data[ptr], sizeof(this->msg.orientation));
        ptr += sizeof(this->msg.orientation);
        std::memcpy(&this->msg.orientation_covariance, &data[ptr], sizeof(this->msg.orientation_covariance));
        ptr += sizeof(this->msg.orientation_covariance);
        // --- Angular Velocity ---
        std::memcpy(&this->msg.angular_velocity, &data[ptr], sizeof(this->msg.angular_velocity));
        ptr += sizeof(this->msg.angular_velocity);
        std::memcpy(&this->msg.angular_velocity_covariance, &data[ptr], sizeof(this->msg.angular_velocity_covariance));
        ptr += sizeof(this->msg.angular_velocity_covariance);
        // --- Linear Acceleration ---
        std::memcpy(&this->msg.linear_acceleration, &data[ptr], sizeof(this->msg.linear_acceleration));
        ptr += sizeof(this->msg.linear_acceleration);
        std::memcpy(&this->msg.linear_acceleration_covariance, &data[ptr], sizeof(this->msg.linear_acceleration_covariance));
        ptr += sizeof(this->msg.linear_acceleration_covariance);

        delete [] data;

        return this->msg;
    }

    void DeserializerLivoxIMU::process_message(struct BaseMsg* msg_){
        // Save Enviro data in a JSON file

        // Populate json object
        json j;
        // header
        j["frame_id"]           = this->msg.ID;
        j["timestamp_sec"]      = this->msg.timestamp_sec;
        j["timestamp_nanosec"]  = this->msg.timestamp_nanosec;
        // data
        j["orientation"] = this->msg.orientation;
        j["orientation_covariance"] = this->msg.orientation_covariance;
        j["angular_velocity"] = this->msg.angular_velocity;
        j["angular_velocity_covariance"] = this->msg.angular_velocity_covariance;
        j["linear_acceleration"] = this->msg.linear_acceleration;
        j["linear_acceleration_covariance"] = this->msg.linear_acceleration_covariance;

        // save to json_array
        this->json_array.push_back(j);
    }

}}} // namespace m2s2, ecal, deserializer