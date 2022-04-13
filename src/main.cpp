#include <chrono>
#include <filesystem>
#include <vector>
#include <iostream>
#include <regex>

#include "image.h"

#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

using namespace std;
namespace fs = std::filesystem;

float to_float( bool x ) {
    return x ? 1.f : 0.f;
}

float rand_float() {
    return rand() / ((float) RAND_MAX);
}

template <typename T, typename Allocator>
std::ostream& operator<<( std::ostream &oss, const std::vector<T, Allocator> &v ) {
    oss << "{ ";

    for( auto &x : v )
        oss << x << ", ";
    oss << "}";

    return oss;
}

// typedef vector<float> vec_t; 
// typedef vector<vec_t> tensor_t;
// typedef vector<vector<float>> tensor_t;
//   typedef vector<vector<vector<float>>> input_t;
// return some 3x3 test data!
// REMEMBER: column first or row first doesn't matter as long as we're consistent
// NOTE: pixel channels (in case of RGB) need to be sequential. The format is: R,G,B, R,G,B, R,G,B, etc...
vec_t test_data() {
    vec_t result;

    for( uint64_t i = 0; i < (512*512); i++ )
        result.push_back( (double)rand()/RAND_MAX );

    return result;
}

bool file_exists( const std::string &filename ) {
    ifstream f( filename.c_str() );
    return f.good();
}

network<sequential> init_net() {
    // create network instance
    network<sequential> net;

#ifdef PD_USE_OPENCL
    const tiny_dnn::core::backend_t bckend = tiny_dnn::core::backend_t::opencl;
#elif defined PD_USE_AVX
    const tiny_dnn::core::backend_t bckend = tiny_dnn::core::backend_t::avx;
#else
    const tiny_dnn::core::backend_t bckend = tiny_dnn::core::default_engine();
#endif

    // define network architecture
    net << convolutional_layer( 128, 128, 4, 1, 4, padding::same ) << tanh_layer()
        << convolutional_layer( 128, 128, 3, 4, 8, padding::same ) << tanh_layer()
        << average_pooling_layer( 128, 128, 8, 2 )                 << tanh_layer()

        << convolutional_layer( 64, 64, 8, 8, 8, padding::same ) << tanh_layer()
        << convolutional_layer( 64, 64, 4, 8, 8, padding::same ) << tanh_layer()
        << average_pooling_layer( 64, 64, 8, 2 )                 << tanh_layer()

        << convolutional_layer( 32, 32, 8, 8, 16, padding::same ) << tanh_layer()
        << convolutional_layer( 32, 32, 8, 16,16, padding::same ) << tanh_layer()
        << max_pooling_layer( 32, 32, 16, 2 )                     << tanh_layer()
        << fully_connected_layer( 16*16*16, 2*1024 ) << tanh_layer()
        << fully_connected_layer( 2*1024, 512 ) << tanh_layer()
        << fully_connected_layer( 512, 2 ) << tanh_layer();

//    uint64_t i = 512;
//    while( i /= 2 )
//        net << fully_connected_layer( i*2, i ) << tanh_layer();

    cout << "input  size: " << net.in_data_size() << endl;
    cout << "output size: " << net.out_data_size() << endl;

    // assert the input and output sizes
    if( (net.in_data_size()  != (128*128))
     or (net.out_data_size() != (2)) ) {
         cerr << "input/output not 128x128 -> 1!" << endl;
         throw;
    }

    // set first 6 layers prallel
    for(int i = 0; i < 16; i++ )
        net[i]->set_parallelize(true);

    return net;
}

int train( network<sequential> &net, fs::path program_directory, const std::vector<std::string> &arguments,
           uint64_t rounds = 2, uint64_t n_subregions = 5 ) {

    std::mt19937_64 rng( rand() );

    // get settings
    uint64_t batch_size = std::stoull( arguments.at(0) );
    uint64_t max_epoch  = 1; //std::stoull( arguments.at(1) );

    // determine test/train paths
    const auto test_directory = program_directory / "prepped_dataset" / "test";
    const auto train_directory= program_directory / "prepped_dataset" / "train";

    std::vector<fs::path> paths;

    // scan prepped dataset for files
    for( const auto &file : fs::recursive_directory_iterator(program_directory / "prepped_dataset" / "train") ) {
//        cout << file.path().generic_string() << endl;
        if( file.is_regular_file() )
            paths.push_back( file.path() );
    }

    // define our optimiser (placed here to prevent losing progress)
    adam opt;
    //opt.alpha = 0.00025;
    //opt.lambda = 0;

    // train the network over the entire dataset once 
    for( uint64_t epoch = 0; epoch < max_epoch; epoch++ ) {
        cout << "epoch loop" << endl;
        // shuffle the image paths
        std::shuffle( paths.begin(), paths.end(), rng );

        cout << "paths: " << paths.size() << endl;

        // process the dataset in batch_size'd chunks
        for( uint64_t start = 0; start < paths.size(); start += batch_size ) {
            // properly sized chunk
            uint64_t count = paths.size()-start;
            if( count > batch_size )
                count = batch_size;

            cout << "processing images: " << start << ", bs:" << batch_size << endl;


            // inputs and outputs
            vector<vec_t> images;
            vector<label_t> desired_output;

            // load the images
            for( uint64_t i = start; i < start+count; i++ ) {
                std:string f = paths.at(i).generic_string();
                //cout << "loading " << f << endl;
                gsimage_t image = gsimage_t::load( f );

                // check if the filename has a pneumonia descriptor in it
                bool has_pneumonia = regex_search( paths.at(i).filename().generic_string(), regex("VIRUS|BACTERIA") );

                try {
                    // get some random subregions and push them to the back of the images
                    const auto subregions = image.get_n_random_subregions( n_subregions, 128, 128 );
                    images.insert( images.end(), subregions.begin(), subregions.end() );
                    const auto ones = std::vector<label_t>( n_subregions, (has_pneumonia?1u:0u) );
                    desired_output.insert( desired_output.end(), ones.begin(), ones.end() );
                } catch( std::runtime_error &e ) {
                    // ignore this
                }
            }

            uint64_t rounds = (count/batch_size) + 1;
            clog << "images processed " << (start+batch_size) << "/" << paths.size() << "..." << endl;

            // log start
            auto t0 = std::chrono::high_resolution_clock::now();

            // create training set of size 5
            bool rc = net.train<mse>( opt, images, desired_output,
                                      images.size(), 1  // # of samples per parameter update, # of training epochs
    //                                  nop, nop,
    //                                  false, // reset_weights?
    //                                  4 // num_threads
                                    );

            if( rc == false )
                throw std::runtime_error("tiny_dnn didn't want to train: net.train() returned false!");

            // log end
            auto t1 = std::chrono::high_resolution_clock::now();

            // check diff
            std::chrono::nanoseconds diff( t1-t0 );
            clog << "took: " << diff.count()/(1000*1000*1000.0) << " seconds" << endl;

            clog << ((start+batch_size)/(double)paths.size()*100) << "% done" << endl;

    //        if( i == rounds-1 ) {
                //clog << "test result:" << endl;
                //auto result = net.test( images, desired_output );
                //result.print_summary( clog );
                //clog << "-----------------------" << endl;
                //result.print_detail( clog );
    //        }

            cout << "saving network" << endl;
            // save the trained network (inside the loop to prevent loss of progress
            net.save("network.net");
        }
    }

    cout << "returning..." << endl;

    return EXIT_SUCCESS;
}

int predict( network<sequential> &net, const std::vector<std::string> &filenames ) {
    uint64_t total_correct = 0, total_count = 0;
    uint64_t model_correct = 0, model_count = 0;

    // go over all files
    for( auto f : filenames ) {
        // determine if file label
        bool has_pneumonia = regex_search( f, regex("VIRUS|BACTERIA") );

        try {
            vec_t model_average = {0, 0};

            // load file
            gsimage_t image = gsimage_t::load(f);

            // loop through 10 random subregions to get a somewhat accurate picture
            for( uint64_t i = 0; i < 10; i++ ) {
                const auto subregion = image.get_random_subregion( 128, 128 );

                // set up the input (and desired output for analysis)
                const vector<vec_t>    ins = { subregion };
                const vector<vec_t>   outs = { { (has_pneumonia?0.f:1.f),
                                                 (has_pneumonia?1.f:0.f) } };

                // run the network
                const auto    o = net.predict( subregion );

                // sum the
                model_average.at(0) += o.at(0);
                model_average.at(1) += o.at(1);

                // calculate loss
                float l = abs(o.at(0)-outs.at(0).at(0))
                        + abs(o.at(1)-outs.at(0).at(1));

                // what does the network predict?
                bool prediction = o.at(0) < o.at(1);

                // was the prediction correct?
                if( has_pneumonia == prediction )
                    total_correct++;

                total_count++;

#ifdef DEBUG
                // print out debugging information
                clog << has_pneumonia << "==" << prediction << "? "
                     << "(prediction=" << p << ", loss=" << l << ",output=" << o << ")" << endl;
#endif
            }

            bool model_prediction = model_average.at(0) < model_average.at(1);

            if( has_pneumonia == model_prediction )
                model_correct++;

            model_count++;
        } catch( std::runtime_error &e ) {
            // ignore this
        }
    }

    cout << "model: " << ((double)model_correct)/model_count*100 << "% accurate "
         << "(" << model_correct << "/" << model_count << ")" << endl;


    cout << "total: " << ((double)total_correct)/total_count*100 << "% accurate "
         << "(" << total_correct << "/" << total_count << ")" << endl;

    return EXIT_SUCCESS;
}

int test( network<sequential> &net, const std::vector<std::string> arguments ) {
    if( arguments.size() != 2 ) {
        std::cerr << "Expected 2 arguments: filename and number of subregions to test" << std::endl;
        return EXIT_FAILURE;
    }

    gsimage_t image = gsimage_t::load( arguments.at(0) );

    uint64_t trials = std::stod( arguments.at(1) );

    // estimatation statistics
    uint64_t normal = 0,
             infected = 0;

    uint64_t count = 0;
    std::vector<double> averages(2);

    for( uint64_t i = 0; i < trials; i++ ) {
        // get a random subregion
        const auto subregion = image.get_random_subregion( 128, 128 );

        // run the network
        const auto    o = net.predict( subregion );

        // what does the network predict?
        bool prediction = o.at(0) < o.at(1);
        if( prediction )
            infected++;
        else
            normal++;

        // add averages to sums
        averages.at(0) += o.at(0);
        averages.at(1) += o.at(1);

        // keep track of count
        count++;
    }

    // divide averages
    averages.at(0) /= count;
    averages.at(1) /= count;

    cout << "# of normal:    " << normal << endl
         << "# of infected: "  << infected << endl;

    cout << "average output: " << averages << endl;

    return EXIT_SUCCESS;
}

int visualise( network<sequential> &net, const std::vector<std::string> arguments ) {
/*    if( arguments.size() != 1 ) {
        std::cerr << "Expected 1 argument: filename to visualize the data over, or 'none' to don't visualize the activation" << std::endl;
        return EXIT_FAILURE;
    }

    gsimage_t input_image = gsimage_t::load( arguments.at(0) );

    // visualize the network
    std::ofstream ofs("visualize/graph_net_example.txt");
    graph_visualizer viz(net, "graph");
    viz.generate(ofs);

    vec_t weight_img = net.at<conv>(0).weight_to_image();
    weight_img.write("visualise/kernel0.bmp");   */

    return EXIT_SUCCESS;
}

int main( int argc, char *argv[] ) {
    // determine directory the program is in
    if( argc <= 0 ) {
        std::cerr << "Could not determine path of program. Weird environment? (argc <= 0)" << std::endl;
        return EXIT_FAILURE;
    }

    const auto program_directory = fs::path(argv[0]).parent_path();

    // seed PRNG with time (good enough)
    srand( time(NULL) );

    // print a separator to denote a new run
    clog << "--run----------------------------" << endl;

    // load arguments into something usable
    std::vector<std::string> arguments;
    for( int i = 1; i < argc; i++ ) {
        arguments.push_back( argv[i] );
    }

    if( arguments.size() <= 1 ) {
        cerr << argv[0] << ": error, no command specified" << endl;
        return EXIT_FAILURE;
    }

    // init the network architecture
    network<sequential> net = init_net();

    // create the network only if it doesn't exist
    if( !file_exists("network.net") ) {
        clog << argv[0] << ": no network exists, saving..." << endl;
        net.init_weight();
        net.save("network.net");
    }

    // load the network
    clog << argv[0] << ": loading network" << endl;
    net.load("network.net");

    // extract command
    std::string command = arguments.at(0);
    arguments.erase( arguments.begin() );

    if( command == "train" )
        return train( net, program_directory, arguments );
    if( command == "predict" )
        return predict( net, arguments );
    if( command == "test" )
        return test( net, arguments );
    if( command == "visualise" )
        return visualise( net, arguments );

    cerr << argv[0] << ": Invalid command: " << command << endl;
    return EXIT_FAILURE;
}
