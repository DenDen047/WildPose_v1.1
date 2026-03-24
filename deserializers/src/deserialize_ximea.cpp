#include "m2s2/ecal/deserializer/deserializer_ximea.hpp"


int main(int argc, char** argv){
    //https://drive.google.com/drive/folders/1CwvUMmM0ryNN6WCilG9mI8PES1_DKQ4m?usp=sharing
    if(argc != 6 && argc != 7) {
        std::cerr << "Invalid number of arguments!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <meas_path> <channel_name> <cam_context_path> <out_path_rgb> <out_path_raw> [image_format]" << std::endl;
        std::cerr << "  image_format: 'webp' (default) or 'jpeg'" << std::endl;
        return 1;
    }

    // std::string meas_path = "/home/naoya/WildPose_v1.1/data/test/2022-11-30_15-17-43.373_measurement";
    // std::string channel_name = "rt/image_raw";
    // std::string cam_context_path = "/home/naoya/WildPose_v1.1/data/test/2022-11-30_15-17-43.373_measurement/cam_context.bin";
    // std::string out_path_rgb = "/home/naoya/WildPose_v1.1/data/test/2022-11-30_15-17-43.373_measurement/rgb/";
    // std::string out_path_raw = "/home/naoya/WildPose_v1.1/data/test/2022-11-30_15-17-43.373_measurement/raw/";

    std::string meas_path =         argv[1];
    std::string channel_name =      argv[2];
    std::string cam_context_path =  argv[3];
    std::string out_path_rgb =      argv[4];
    std::string out_path_raw =      argv[5];
    std::string image_format =      (argc == 7) ? argv[6] : "webp";  // Default: webp

    try {
        m2s2::ecal::deserializer::DeserializerXimea xi_deserializer(
            meas_path,
            channel_name,
            cam_context_path,
            out_path_rgb,
            out_path_raw,
            image_format
        );
        xi_deserializer.process_all();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

namespace m2s2{ namespace ecal{ namespace deserializer{

    DeserializerXimea::DeserializerXimea(std::string meas_path, std::string channel_name, std::string cam_context_path, std::string out_path_rgb, std::string out_path_raw, std::string image_format) :
        Deserializer(meas_path, channel_name),
        cam_context_path(cam_context_path),
        out_path_rgb(out_path_rgb),
        out_path_raw(out_path_raw),
        image_format(image_format),
        skip_raw_output(false)

    {
        // Check if raw output should be skipped
        if (this->out_path_raw == "none") {
            this->skip_raw_output = true;
            std::cout << "Raw image output: DISABLED (out_path_raw is 'none')" << std::endl;
        } else {
            std::cout << "Raw image output: ENABLED (output path: " << this->out_path_raw << ")" << std::endl;
        }

        // Validate and set image format
        if (this->image_format != "webp" && this->image_format != "jpeg") {
            std::cerr << "Warning: Invalid image format '" << this->image_format << "'. Using default 'webp'." << std::endl;
            this->image_format = "webp";
        }

        // Set file extension based on format
        this->file_extension = (this->image_format == "webp") ? ".webp" : ".jpeg";

        std::cout << "Image format: " << this->image_format << " (extension: " << this->file_extension << ")" << std::endl;

        this->camh = NULL;
        this->cam_context = NULL;
        this->cam_context = (char*)malloc(SIZE_OF_CONTEXT_BUFFER);
        if (this->cam_context == NULL) {
            throw std::runtime_error("Memory allocation failed!");
        }
        this->read_bin_file(this->cam_context_path, this->cam_context, SIZE_OF_CONTEXT_BUFFER);

        std::string s(cam_context);
        std::cout << "GOT CAM CONTEXT: " << std::endl << cam_context << std::endl;

        std::cout << "Opening Ximea Offline Processing" << std::endl;
        XI_CARE(xiProcOpen(&this->proc));
        std::cout << "Ximea Offline Procesing Opened" << std::endl;
        int cam_context_len = (DWORD)strlen(cam_context);

        XI_CARE(xiProcSetParam(this->proc, XI_PRM_API_CONTEXT_LIST, this->cam_context, cam_context_len, xiTypeString));
        XI_IMG_FORMAT output_format = XI_RGB32;
        XI_CARE(xiProcSetParam(this->proc, XI_PRM_IMAGE_DATA_FORMAT, &output_format, sizeof(output_format), xiTypeInteger));

    }

    DeserializerXimea::~DeserializerXimea(){
    }

    struct BaseMsg DeserializerXimea::dry_run(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {
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

        // Copy this->msg context to an Image struct

        //Timestamp
        int ptr = 0;
        std::memcpy(&(this->msg.timestamp_sec), &data[ptr], sizeof((this->msg.timestamp_sec)));
        // std::cout << "Timestamp secs: " <<  (this->msg.timestamp_sec) << std::endl;

        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        // std::cout << "Timestamp nanosecs: " <<  this->msg.timestamp_nanosec << std::endl;

        ptr += sizeof(this->msg.timestamp_nanosec);


        std::string timestr_raw = std::to_string(this->msg.timestamp_sec) + "_" + std::to_string(this->msg.timestamp_nanosec);
        std::string img_name_raw = this->out_path_raw + this->msg.ID + "_" + timestr_raw + ".tiff";
        std::string timestr = get_timestamp_string(this->msg.timestamp_sec, this->msg.timestamp_nanosec);
        std::string img_name_rgb = this->out_path_rgb + this->msg.ID + "_" + timestr + this->file_extension;

        // Frame ID
        uint64_t size_of_frameid;
        std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));

        ptr += sizeof(size_of_frameid);
        this->frame_id_cstring = (uint8_t*)malloc((size_t)size_of_frameid);
        std::memcpy(&this->frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
        std::string s((const char*)&frame_id_cstring, size_of_frameid);
        this->msg.ID = s;
        // std::cout << "Frame ID: " <<  this->msg.ID << std::endl;

        ptr += size_of_frameid;
        std::memcpy(&(this->msg.height), &data[ptr], sizeof((this->msg.height)));
        // std::cout << "Height: " <<  (this->msg.height) << std::endl;

        ptr += sizeof(this->msg.height);
        std::memcpy(&(this->msg.width), &data[ptr], sizeof((this->msg.width)));
        // std::cout << "Width: " <<  (this->msg.width) << std::endl;

        ptr += sizeof(this->msg.width);

        // Encoding
        uint64_t size_of_encoding;
        std::memcpy(&size_of_encoding, &data[ptr], sizeof(size_of_encoding));

        ptr += sizeof(size_of_encoding);
        this->encoding_cstring = (uint8_t*)malloc(size_of_encoding);
        std::memcpy(&this->encoding_cstring, &data[ptr], sizeof(encoding_cstring));
        this->msg.encoding = std::string((const char*)&encoding_cstring);
        // std::cout << "Encoding: " <<  this->msg.encoding << std::endl;

        ptr += size_of_encoding;
        std::memcpy(&(this->msg.is_bigendian), &data[ptr], sizeof((this->msg.is_bigendian)));
        // std::cout << "Is Bigendian?: " <<  int(this->msg.is_bigendian) << std::endl;

        ptr += sizeof(this->msg.is_bigendian);
        std::memcpy(&(this->msg.step), &data[ptr], sizeof((this->msg.step)));
        // std::cout << "Step: " <<  (this->msg.step) << std::endl;

        ptr += sizeof(this->msg.step);
        std::memcpy(&(this->msg.data_size), &data[ptr], sizeof((this->msg.data_size)));
        // std::cout << "Data size: " <<  (this->msg.data_size) << std::endl;

        ptr += sizeof(this->msg.data_size);
        this->msg.data = (uint8_t*)malloc((size_t)this->msg.data_size);
        std::memcpy(&this->msg.data[0], &data[ptr], (size_t)(this->msg.data_size));

        // std::cout << "DONE DESERIALIZING" << std::endl;

        delete [] data;

        return this->msg;
    }


    struct BaseMsg DeserializerXimea::deserialize_message(
        eCAL::eh5::HDF5Meas* meas_,
        long long ID)
    {
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

        // Copy this->msg context to an Image struct

        //Timestamp
        int ptr = 0;
        std::memcpy(&(this->msg.timestamp_sec), &data[ptr], sizeof((this->msg.timestamp_sec)));
        // std::cout << "Timestamp secs: " <<  (this->msg.timestamp_sec) << std::endl;

        ptr += sizeof(this->msg.timestamp_sec);
        std::memcpy(&this->msg.timestamp_nanosec, &data[ptr], sizeof(this->msg.timestamp_nanosec));
        // std::cout << "Timestamp nanosecs: " <<  this->msg.timestamp_nanosec << std::endl;

        ptr += sizeof(this->msg.timestamp_nanosec);


        std::string timestr_raw = std::to_string(this->msg.timestamp_sec) + "_" + std::to_string(this->msg.timestamp_nanosec);
        std::string img_name_raw = this->out_path_raw + this->msg.ID + "_" + timestr_raw + ".tiff";
        std::string timestr = get_timestamp_string(this->msg.timestamp_sec, this->msg.timestamp_nanosec);
        std::string img_name_rgb = this->out_path_rgb + this->msg.ID + "_" + timestr + this->file_extension;

        if (!std::filesystem::exists(img_name_rgb)) {
            // Frame ID
            uint64_t size_of_frameid;
            std::memcpy(&size_of_frameid, &data[ptr], sizeof(size_of_frameid));

            ptr += sizeof(size_of_frameid);
            std::memcpy(&this->frame_id_cstring, &data[ptr], sizeof(frame_id_cstring));
            std::string s((const char*)&frame_id_cstring, size_of_frameid);
            this->msg.ID = s;
            // std::cout << "Frame ID: " <<  this->msg.ID << std::endl;

            ptr += size_of_frameid;
            std::memcpy(&(this->msg.height), &data[ptr], sizeof((this->msg.height)));
            // std::cout << "Height: " <<  (this->msg.height) << std::endl;

            ptr += sizeof(this->msg.height);
            std::memcpy(&(this->msg.width), &data[ptr], sizeof((this->msg.width)));
            // std::cout << "Width: " <<  (this->msg.width) << std::endl;

            ptr += sizeof(this->msg.width);

            // Encoding
            uint64_t size_of_encoding;
            std::memcpy(&size_of_encoding, &data[ptr], sizeof(size_of_encoding));

            ptr += sizeof(size_of_encoding);
            std::memcpy(&this->encoding_cstring, &data[ptr], sizeof(encoding_cstring));
            this->msg.encoding = std::string((const char*)&encoding_cstring);
            // std::cout << "Encoding: " <<  this->msg.encoding << std::endl;

            ptr += size_of_encoding;
            std::memcpy(&(this->msg.is_bigendian), &data[ptr], sizeof((this->msg.is_bigendian)));
            // std::cout << "Is Bigendian?: " <<  int(this->msg.is_bigendian) << std::endl;

            ptr += sizeof(this->msg.is_bigendian);
            std::memcpy(&(this->msg.step), &data[ptr], sizeof((this->msg.step)));
            // std::cout << "Step: " <<  (this->msg.step) << std::endl;

            ptr += sizeof(this->msg.step);
            std::memcpy(&(this->msg.data_size), &data[ptr], sizeof((this->msg.data_size)));
            // std::cout << "Data size: " <<  (this->msg.data_size) << std::endl;

            ptr += sizeof(this->msg.data_size);
            std::memcpy(&this->msg.data[0], &data[ptr], (size_t)(this->msg.data_size));

            // std::cout << "DONE DESERIALIZING" << std::endl;
        }

        delete [] data;

        return this->msg;
    }

    void DeserializerXimea::process_message(struct BaseMsg* msg_){

        // Save raw image (skip if disabled)
        if (!this->skip_raw_output) {
            std::string timestr_raw = std::to_string(this->msg.timestamp_sec) + "_" + std::to_string(this->msg.timestamp_nanosec);
            std::string img_name_raw = this->out_path_raw + this->msg.ID + "_" + timestr_raw + ".tiff";
            if (!std::filesystem::exists(img_name_raw)) {
                cv::Mat img_mat_raw = cv::Mat(this->msg.height, this->msg.width, CV_8UC1, this->msg.data);
                cv::imwrite(img_name_raw, img_mat_raw);
                // std::cout << "Raw Image Saved" << std::endl;
            }
        }

        std::string timestr = get_timestamp_string(this->msg.timestamp_sec, this->msg.timestamp_nanosec);
        std::string img_name_rgb = this->out_path_rgb + this->msg.ID + "_" + timestr + this->file_extension;
        if (!std::filesystem::exists(img_name_rgb)) {
            // std::cout << std::endl << "Processing Ximea Image" << std::endl;
            struct Image* msg = (struct Image*)msg_;
            XI_IMG out_image;

            out_image.size = sizeof(XI_IMG);
            out_image.bp = NULL;
            out_image.bp_size = 0;

            XI_CARE(xiProcPushImage(this->proc, this->msg.data));
            XI_CARE(xiProcPullImage(this->proc, 0, &out_image));

            // std::cout << "image height: " << out_image.height << std::endl;
            // std::cout << "image width: " << out_image.width << std::endl;

            cv::Mat img_mat_bgra = cv::Mat(out_image.height, out_image.width, CV_8UC4, out_image.bp);

            // Convert BGRA to BGR (remove alpha channel) for proper encoding
            // XI_RGB32 format from Ximea is actually BGRA (OpenCV byte order)
            cv::Mat img_mat_bgr;
            cv::cvtColor(img_mat_bgra, img_mat_bgr, cv::COLOR_BGRA2BGR);

            // Save with appropriate format and compression
            std::vector<int> compression_params;
            if (this->image_format == "jpeg") {
                // JPEG compression (quality = 95 for high quality)
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(95);
            } else {
                // WebP lossless compression (quality = 100 for lossless)
                compression_params.push_back(cv::IMWRITE_WEBP_QUALITY);
                compression_params.push_back(100);
            }
            cv::imwrite(img_name_rgb, img_mat_bgr, compression_params);
            // std::cout << "Image Saved" << std::endl;
        }
    }

    void DeserializerXimea::read_bin_file(std::string src_path, char* dest, int size){
        std::ifstream file_in(src_path, std::ios::in | std::ios::binary);

        file_in.read(dest, size);

        if (!file_in){
            std::cout << "Error Reading File: " << src_path << std::endl;
        }
    }

}}} // namespace m2s2, ecal, deserializer