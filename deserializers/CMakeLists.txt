cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(XIMEA_LIBRARY m3api)
set(MSGS_INCLUDE_PATH usr/include)

project(m2s2_ecal_deserializers)

find_package(eCAL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
find_package(rclcpp REQUIRED)

set(PROJECT_NAME m2s2_ecal_deserializers)

# NOTE: build with ->  cmake --build . --target m2s2_ecal_deserializers
set(measurement_read_src src/ximea_measurement_read.cpp)
ecal_add_sample(${PROJECT_NAME} ${measurement_read_src})

# NOTE: build with ->  cmake --build . --target ximea
set(XIMEA_TARGET ximea)
ecal_add_sample(${XIMEA_TARGET} src/deserialize_ximea.cpp)

# NOTE: build with ->  cmake --build . --target bme280
set(BME280_TARGET bme280)
ecal_add_sample(${BME280_TARGET} src/deserialize_bme280.cpp)

# NOTE: build with ->  cmake --build . --target radar
set(RADAR_TARGET radar)
ecal_add_sample(${RADAR_TARGET} src/deserialize_radar.cpp)

# NOTE: build with ->  cmake --build . --target lidar
set(LIDAR_TARGET lidar)
ecal_add_sample(${LIDAR_TARGET} src/deserialize_livox.cpp)

# NOTE: build with ->  cmake --build . --target livox_imu
set(LIVOX_IMU_TARGET livox_imu)
ecal_add_sample(${LIVOX_IMU_TARGET} src/deserialize_livox_imu.cpp)

set(INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
    ${MSGS_INCLUDE_PATH}
    ${PCL_INCLUDE_DIRS}

)

set(LIBS
    eCAL::hdf5
    ${OpenCV_LIBS}
    ${XIMEA_LIBRARY}
    ${PCL_LIBRARIES}
)

set(DEFS
    ${PCL_DEFINITIONS})


target_compile_definitions(${PROJECT_NAME} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")
target_compile_definitions(${XIMEA_TARGET} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")
target_compile_definitions(${BME280_TARGET} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")
target_compile_definitions(${RADAR_TARGET} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")
target_compile_definitions(${LIDAR_TARGET} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")
target_compile_definitions(${LIVOX_IMU_TARGET} PRIVATE MEASUREMENT_PATH="${SAMPLE_PERSON_MEASUREMENT_PATH}")



target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
target_include_directories(${XIMEA_TARGET} PUBLIC ${INCLUDE_DIRS})
target_include_directories(${BME280_TARGET} PUBLIC ${INCLUDE_DIRS})
target_include_directories(${RADAR_TARGET} PUBLIC ${INCLUDE_DIRS})
target_include_directories(${LIDAR_TARGET} PUBLIC ${INCLUDE_DIRS})
target_include_directories(${LIVOX_IMU_TARGET} PUBLIC ${INCLUDE_DIRS})



target_link_libraries(${PROJECT_NAME} ${LIBS})
target_link_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
#target_compile_definitions(${PROJECT_NAME} ${DEFS})

target_link_libraries(${XIMEA_TARGET} ${LIBS})
target_link_directories(${XIMEA_TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${BME280_TARGET} ${LIBS})
target_link_directories(${BME280_TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${RADAR_TARGET} ${LIBS})
target_link_directories(${RADAR_TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${LIDAR_TARGET} ${LIBS})
target_link_directories(${LIDAR_TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PCL_LIBRARY_DIRS}
)

target_link_libraries(${LIVOX_IMU_TARGET} ${LIBS})
target_link_directories(${LIVOX_IMU_TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${PCL_LIBRARY_DIRS}
)


target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_compile_features(${XIMEA_TARGET} PRIVATE cxx_std_14)
target_compile_features(${BME280_TARGET} PRIVATE cxx_std_14)
target_compile_features(${RADAR_TARGET} PRIVATE cxx_std_14)
target_compile_features(${LIDAR_TARGET} PRIVATE cxx_std_14)
target_compile_features(${LIVOX_IMU_TARGET} PRIVATE cxx_std_14)




ecal_install_sample(${PROJECT_NAME})
ecal_install_sample(${XIMEA_TARGET})
ecal_install_sample(${BME280_TARGET})
ecal_install_sample(${RADAR_TARGET})
ecal_install_sample(${LIDAR_TARGET})
ecal_install_sample(${LIVOX_IMU_TARGET})





if(MSVC)
    set_property(TARGET ${PROJECT_NAME} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)

if(MSVC)
    set_property(TARGET ${XIMEA_TARGET} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)

if(MSVC)
    set_property(TARGET ${BME280_TARGET} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)

if(MSVC)
    set_property(TARGET ${RADAR_TARGET} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)

if(MSVC)
    set_property(TARGET ${LIDAR_TARGET} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)

if(MSVC)
    set_property(TARGET ${LIVOX_IMU_TARGET} PROPERTY LINK_FLAGS "/ignore:4098,4099")
endif(MSVC)
