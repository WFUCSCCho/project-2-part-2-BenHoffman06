#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>
#include <sys/time.h>

#define BLUR_SIZE 16 // size of surrounding image is 2X this
#define IS_OPTIMIZED 0
#define OUTER_TILE_WIDTH 45

#include "bitmap_image.hpp"

using namespace std;

__global__ void blurKernel (uchar3 *in, uchar3 *out, int width, int height) {

 int col = blockIdx.x * blockDim.x + threadIdx.x;
 int row = blockIdx.y * blockDim.y + threadIdx.y;

 if (col < width && row < height) {
  int3 pixVal;
  pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
  int pixels = 0;

  // get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
  for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
   for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {

    int curRow = row + blurRow;
    int curCol = col + blurCol;

    // verify that we have a valid image pixel
    if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
     pixVal.x += in[curRow * width + curCol].x;
     pixVal.y += in[curRow * width + curCol].y;
     pixVal.z += in[curRow * width + curCol].z;
     pixels++; // keep track of number of pixels in the accumulated total
    }
   }
  }

  // write our new pixel value out
  out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
  out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
  out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
 }
}

// Called with dim3 dimBlock(16, 16, 1);
__global__ void blurKernelOptimized (uchar3 *in, uchar3 *out, int width, int height) {
    int INNER_TILE_WIDTH = OUTER_TILE_WIDTH - BLUR_SIZE * 2; // 13
    __shared__ uchar3 surrounding[OUTER_TILE_WIDTH][OUTER_TILE_WIDTH]; // [0][0] is top left of surrounding region, [OUTER_TILE_WIDTH-1][OUTER_TILE_WIDTH-1] is bottom right
    
    // COLLABORATIVE LOADING OF PIXELS INTO SHARED MEMORY
    // Calculate global cols and row bounds for pixels to be added to shared memory array
    int shared_topmost_row = blockIdx.y * INNER_TILE_WIDTH;
    int shared_leftmost_col = blockIdx.x * INNER_TILE_WIDTH;

    // Iterate thru valid pixels, splitting allocations between threads
    for (int i = threadIdx.y; i < OUTER_TILE_WIDTH; i+=blockDim.y) {
        for (int j = threadIdx.x; j < OUTER_TILE_WIDTH; j+=blockDim.x) {
            int pixel_row = shared_topmost_row + i;
            int pixel_col = shared_leftmost_col + j;
            surrounding[i][j] = in[pixel_row * width + pixel_col];
        }
    }
    __syncthreads();


    // ASSIGN THREAD TO UNIQUE PIXEL IN INNER TILE
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_row = global_row - shared_topmost_row; 
    int shared_col = global_col - shared_leftmost_col;

    bool shared_row_is_valid = 0 <= shared_row && shared_row < INNER_TILE_WIDTH;
    bool shared_col_is_valid = 0 <= shared_col && shared_col < INNER_TILE_WIDTH;

    // FOR THAT PIXEL, CALCULATE BLUR AND UPDATE ITS VALUE IN OUT
    if (global_col < width && global_row < height && shared_row_is_valid && shared_col_is_valid) { 
        int3 pixVal;
        pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
        int pixels = 0;

        // get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for(int blurRow = 0; blurRow < BLUR_SIZE*2 + 1; blurRow++) {
            for(int blurCol = 0; blurCol < BLUR_SIZE*2 + 1; blurCol++) {

                // verify that we have a valid image pixel
                if(shared_row + blurRow < height && shared_col + blurCol < width) {
                    uchar3 pixVal2 = surrounding[shared_row + blurRow][shared_col + blurCol];
                    pixVal.x += pixVal2.x;
                    pixVal.y += pixVal2.y;
                    pixVal.z += pixVal2.z;
                    pixels++; // keep track of number of pixels in the accumulated total
                }
            }
        }

        // write our new pixel value out
        out[global_row * width + global_col].x = (unsigned char)(pixVal.x / pixels);
        out[global_row * width + global_col].y = (unsigned char)(pixVal.y / pixels);
        out[global_row * width + global_col].z = (unsigned char)(pixVal.z / pixels);
    }
}

void print_timing_results(const char *filename, int size, float execution_time_in_seconds) {
    bool just_created = false;

    // Open file, checking for errors and adding header if creating now
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        just_created = true;
    } else fclose(fp);

    FILE *fptr = fopen(filename, "a");
    if (fptr == NULL) {
        printf("Can't open file with append mode.");
        exit(-1);
    }

    // Print results to opened file
    if (just_created) {
        fprintf(fptr, "image_size\ttime(s)\n");
    }
    fprintf(fptr, "%d\t%f\n", size, execution_time_in_seconds);
    fclose(fptr);
}

void run(dim3 dimGrid, dim3 dimBlock, uchar3 *d_in, uchar3 *d_out, int width, int height) { 
    if (IS_OPTIMIZED) {
        blurKernelOptimized<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    } else {
        blurKernel<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    }
    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
    struct timeval stop, start;

    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Converting " << argv[1] << " from color to grayscale..." << endl;

    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    // TODO: Fill in the correct blockSize and gridSize
    // currently only one block with one thread is being launched
    dim3 dimGrid(ceil((float)width / 16), ceil((float)height / 16), 1);
    dim3 dimBlock(16, 16, 1);

//    dim3 dimGrid(ceil(input_image.size()/1024), 1, 1);
//    dim3 dimBlock(1024, 1, 1);

    // Warm up kernel
    run(dimGrid, dimBlock, d_in, d_out, width, height);

    // Time kernel running
    gettimeofday(&start, NULL);
    run(dimGrid, dimBlock, d_in, d_out, width, height);
    gettimeofday(&stop, NULL);

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    // Print kernel runtime
    float execution_time_in_seconds = (stop.tv_sec - start.tv_sec) + (float) (stop.tv_usec - start.tv_usec)/1000000;
    printf("Execution time in seconds: %f \n", execution_time_in_seconds);
    
    const char *filename = IS_OPTIMIZED ? "optimized.csv" : "original.csv";
    print_timing_results(filename, width*height, execution_time_in_seconds);
    
    //Set updated pixels
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * height + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    if (IS_OPTIMIZED) {
        printf("Used optimized version.\n");
    } else {
        printf("Used unoptimized version.\n");
    }
    
    bmp.save_image("./blurred.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}