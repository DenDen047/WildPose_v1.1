namespace m2s2{ namespace ecal{ namespace deserializer{


    struct BaseMsg {
        std::string ID = "123456";

        uint32_t timestamp_sec;
        uint32_t timestamp_nanosec;
    };

    struct Image : BaseMsg {

        uint32_t height;
        uint32_t width;

        std::string encoding = "mono8";
        uint8_t is_bigendian;
        uint32_t step;

        uint64_t data_size;
        uint8_t* data;
    };

    struct ThermalRaw : BaseMsg {
        //change for new recording of boson messages to uint32_t
        uint16_t height;
        uint16_t width;

        uint64_t data_size;
        uint16_t* data;
    };

    struct AudioData : BaseMsg {

        uint64_t data_size;
        uint8_t* data;
    };

    struct EnviroData : BaseMsg {

        double temperature;
        double pressure;
        double humidity;

    };

    struct RadarFrame : BaseMsg {

        uint32_t frame_size;
        uint8_t* frame_data;
    };

    struct PointField {
       std::string name;
       uint32_t offset;
       uint8_t datatype;
       uint32_t count;

    };

    struct PointCloud2 : BaseMsg  {
        uint32_t height;
        uint32_t width;
        uint8_t is_bigendian;
        uint32_t point_step;
        uint32_t row_step;
        uint8_t* data;
        uint8_t is_dense;

    };

    struct IMU : BaseMsg  {
        double orientation[4];
        double angular_velocity[3];
        double linear_acceleration[3];
        double orientation_covariance[9];
        double angular_velocity_covariance[9];
        double linear_acceleration_covariance[9];
    };



}}} // namespace m2s2, ecal, deserializer