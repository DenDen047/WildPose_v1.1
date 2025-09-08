#ifndef __DESERIALIZER__
#define __DESERIALIZER__

#include <ecalhdf5/eh5_meas.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include "msgs.hpp"

namespace m2s2{ namespace ecal{ namespace deserializer{

class Deserializer
{
    public:
    inline explicit Deserializer(std::string meas_path, std::string channel_name){
        this->meas_ = new eCAL::eh5::HDF5Meas(meas_path);
        if (!this->meas_->IsOk()){
            std::cout << "Error: Problem with measurement file during deserialization." << std::endl;
        }
        this->meas_->GetEntriesInfo(channel_name, this->entry_info_set);
    }
    virtual ~Deserializer() = default;

    virtual struct BaseMsg dry_run(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    ) = 0;

    virtual struct BaseMsg deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID
    ) = 0;

    virtual void process_message(struct BaseMsg* msg) = 0;

    inline void process_all() {
        std::size_t count = 0;
        std::size_t total = this->entry_info_set.size();
        bool enable_dry_run = true;

        for (auto it=this->entry_info_set.begin(); it!=this->entry_info_set.end(); it++, count++)
        {
            if (enable_dry_run) {
                dry_run(this->meas_, it->ID);
                enable_dry_run = false;
            }
            // std::cout << "-------------------------------" << std::endl;
            struct BaseMsg msg = deserialize_message(this->meas_, it->ID);
            process_message(&msg);

            // progress
            float progress = static_cast<float>(count) / total;
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();
        }
        std::cout << std::endl;
    }

    inline std::string zeros_padding(uint32_t x, int length) {
        std::string old_string = std::to_string(x);
        return std::string(length - std::min(length, int(old_string.length())), '0') + old_string;
    }

    inline std::string get_timestamp_string(uint32_t sec, uint32_t nanosec) {
        std::string timestr = zeros_padding(sec, 10) + "_" + zeros_padding(nanosec, 9);

        return timestr;
    }

    // Measurement reading variables
    eCAL::eh5::HDF5Meas* meas_;
    eCAL::eh5::EntryInfoSet entry_info_set;

    private:

};

}}} // namespace m2s2 -> ecal -> deserializer

#endif