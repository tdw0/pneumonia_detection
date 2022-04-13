cmake_minimum_required( VERSION 3.0 )

project( pneumonia-detector
         LANGUAGES CXX )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory( git-submodules/tiny-dnn )

add_executable( main
    src/main.cpp
)

set_target_properties( main
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

#target_compile_options( main
#    PUBLIC
#        -Wall
#        -Wextra
#)

target_link_libraries( main
    PUBLIC Threads::Threads
    PUBLIC tiny_dnn
)

if( USE_OPENCL )
        target_compile_definitions( main
                                    PRIVATE PD_USE_OPENCL )
endif()

if( USE_AVX )
        target_compile_definitions( main
                                    PRIVATE PD_USE_AVX )
endif()

# Copy scripts to build directory
configure_file( scripts/get-dataset.sh      get-dataset.sh )
configure_file( scripts/prep-dataset.sh     prep-dataset.sh )
configure_file( scripts/learn.sh            learn.sh )
configure_file( scripts/test.sh             test.sh )
configure_file( scripts/test-one.sh         test-one.sh )
configure_file( scripts/test-all.sh         test-all.sh )
