#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNELS 3

__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned
	char* Pin, int width, int height) {
	
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	if (Col < width && Row < height) {
		int greyOffset = Row * width + Col;
		int rgbOffset = greyOffset * CHANNELS;
		unsigned char R = Pin[rgbOffset]; // red value for pixel
		unsigned char G = Pin[rgbOffset + 1]; // green value for pixel
		unsigned char B = Pin[rgbOffset + 2]; // blue value for pixel
		Pout[rgbOffset] = 0.21f * R + 0.71f * G + 0.07f * B;
		Pout[rgbOffset+1] = 0.21f * R + 0.71f * G + 0.07f * B;
		Pout[rgbOffset+2] = 0.21f * R + 0.71f * G + 0.07f * B;
		

	}
}
int main(int arc,char ** argv) {
	int width, height, rgb;
	unsigned char* Pin = stbi_load(argv[1], &width, &height, &rgb, 3);
	unsigned char* ptrImageData = NULL;
	unsigned char* ptrImageDataOut = NULL;
	/*Reserva y asignacion de memoria de Host a Dispositivo*/
	cudaMalloc(&ptrImageDataOut, width * height * CHANNELS);
	cudaMalloc(&ptrImageData, width * height * CHANNELS);
	cudaMemcpy(ptrImageData, Pin, width * height * CHANNELS, cudaMemcpyHostToDevice);
	/*Invocacion de la funcion Kernel*/
	colorToGreyscaleConversion << <dim3((width / 16), (height / 16)), dim3(16, 16) >> > (ptrImageDataOut, ptrImageData, width, height);
	/*Copia de memoria de dispositivo a Host*/
	cudaMemcpy(Pin, ptrImageDataOut, width * height * CHANNELS, cudaMemcpyDeviceToHost);
	std::string NewImageFile = argv[1];
	NewImageFile = NewImageFile.substr(0, NewImageFile.find_last_of('.')) + "out.png";
	stbi_write_png(NewImageFile.c_str(), width, height, 3, Pin, 3 * width);
	stbi_image_free(Pin);
	/*Liberacion de memoria*/
	cudaFree(ptrImageData);
	cudaFree(ptrImageDataOut);
	return 0;
}