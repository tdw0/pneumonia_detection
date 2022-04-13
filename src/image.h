#ifndef IMAGE_H
#define IMAGE_H
#include <cassert>
#include <fstream>
#include <vector>
#include <tiny_dnn/tiny_dnn.h>

//! Custom std::getline replacement

/*std::basic_istream& getline( std::basic_istream &is,
                             std::string &str,
                             char delim = '\n',
                             size_t max_characters = 0 ) {
    str.erase();
}*/

//! Custom struct that holds a grayscale image
struct gsimage_t {
    const uint64_t width, height;
    const std::vector<float> data;

public:
    // no empty images here, boss
    gsimage_t() : width(0), height(0) {};
    gsimage_t( const uint64_t Width,
               const uint64_t Height,
               const std::vector<float> &Data ) : width(Width), height(Height), data(Data) {}

    //! Returns a subregion as a flat vector (row by row)
    tiny_dnn::vec_t get_subregion( uint64_t x_offset, uint64_t y_offset,
                                                uint64_t x_size,   uint64_t y_size ) const {
        // our resulting image
        tiny_dnn::vec_t result;

        // define index
        std::vector<float>::size_type idx = 0;

        // skip to appropriate starting point
        idx += x_offset + (y_offset*width);

        // start outputting pixels
        for( uint64_t y = 0; y < y_size; y++ ) {
            for( uint64_t x = 0; x < x_size; x++ ) {
                result.push_back( data.at(idx) );
                idx++;
            }

            // advance to start of next row
            // note: this code will advance beyond the image on the last loop
            idx += (width - x_size);
        }

        return result;
    }

    //! Function that loads a pgm image into a image_t (row by row!)
    static gsimage_t load( std::string filename ) {
        // open the stream, check if it's actually opened
        std::ifstream ifs( filename, std::ios_base::binary | std::ios_base::in );
        if( !ifs.is_open() )
            throw std::runtime_error("Unable to open file: " + filename);

        // check for magic /////////
        const std::string magic = "P5\n";
        std::string buf;
        for( int i = 0; i < 3; i++ ) {
            char c;
            ifs.read( &c, 1 );

            if( !ifs )
                throw std::runtime_error("Unable to extract magic from image: " + filename );

            buf.append(1, c);
        }

        if( buf != magic ) {
            std::cerr << buf << std::endl;
            std::cerr << magic << std::endl;

            throw std::runtime_error("Image is not pgm (P5) image!: " + filename );
        }

        uint64_t width, height, depth;
        ifs >> width;
        ifs >> height;
        ifs >> depth;

        if( !ifs )
            throw std::runtime_error("Unable to extract image sizes from pgm (P5) image!: " + filename);

        if( depth != 255 )
            throw std::runtime_error("Image is not 8-bit depth, we can't load this");

        std::vector<float> image_data;
        for( uint64_t i = 0; i < (width*height); i++ ) {
            char c[1];
            ifs.read( c, 1 );

            if( !ifs )
                throw std::runtime_error("Unable to extract image data, corrupt file??: " + filename);

            image_data.push_back( reinterpret_cast<uint8_t*>(c)[0] / 255.0f );
        }

        // finally return the image data
        return gsimage_t(width, height, image_data);
    }

    auto get_random_subregion( uint64_t x_size, uint64_t y_size ) {
        if( (x_size > width) or (y_size > height) ) {
            throw std::runtime_error("ERROR: get_random_subregion(): requested size larger than image!");
        }

        // this is not the most random
        uint64_t x_offset = rand()%(width  - x_size);
        uint64_t y_offset = rand()%(height - y_size);

        return get_subregion( x_offset, y_offset, x_size, y_size );
    }

    auto get_n_random_subregions( uint64_t n, uint64_t x_size, uint64_t y_size ) {
        std::vector<tiny_dnn::vec_t> result;

        for( uint64_t i = 0; i < n; i++ )
            result.push_back( get_random_subregion(x_size, y_size) );

        return result;
    }
};

#endif /* IMAGE_H */
