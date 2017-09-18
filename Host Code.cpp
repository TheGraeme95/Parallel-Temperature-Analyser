#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <CL/cl.hpp>
#include "Utils.h"
#include <chrono>


typedef float mytype;
using namespace std;
using namespace std::chrono;

cl::Event profEvent; // Event used to store profiling information.
size_t numInputElements, workGroupSize, initialInputSize; // Accessed by every kernel function so defined as global.

void print_help()
{
	cerr << "Application usage:" << endl;
	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

// A method to run the specified by argument kernel, while also displaying the profiling information
// every time it is called.

void callKernel(cl::Kernel& name, cl::CommandQueue& queue)
{
	cl::Event profEvent;
	queue.enqueueNDRangeKernel(name, cl::NullRange, cl::NDRange(numInputElements), cl::NDRange(workGroupSize), NULL, &profEvent);
	queue.finish();
	std::cout << "Execution time: "
		<< profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns" << "\n\t"
		<< GetFullProfilingInfo(profEvent, ProfilingResolution::PROF_US) << '\n' << std::endl;
}

/////////////////////////////////// SUM KERNEL CALL ///////////////////////////////////////
/////////////////////////////////// SUM KERNEL CALL ///////////////////////////////////////

float Sum(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype>& inputData)
{
	size_t input_size = inputData.size() * sizeof(mytype); // Size of input in bytes 
	vector<mytype> outputData(inputData.size()); // Defining vector to contain the output data.
	size_t output_size = outputData.size() * sizeof(mytype); // Size of output in bytes.
	size_t N = inputData.size() / workGroupSize; // The amount of partial sums returned after one iteration of reduction.


												 // Creating 2 buffers in which to pass and recieve information to and from kernel.
	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	// Write data to the input buffer. 
	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	// Fill the output buffer with 0's to allocate memory.
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);


	// Create the kernel for the sum function.
	// Passing the input buffer, local memory size and output buffer as arguments.
	cl::Kernel kernel_sum = cl::Kernel(program, "floatSum");
	kernel_sum.setArg(0, inBuffer);
	kernel_sum.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));//local memory size	
	kernel_sum.setArg(2, outBuffer);


	// The kernel is called once and then the output buffer is set to the input of the next
	// kernel call to further reduce.
	callKernel(kernel_sum, queue);
	kernel_sum.setArg(0, outBuffer);

	// It will continue to call the kernel until all the partial sums returned by reduction can be encompassed
	// within one work group.
	// Bigger work group sizes can potentially increase performance due to less kernel calls.
	while (N > workGroupSize)
	{
		callKernel(kernel_sum, queue);
		N = N / workGroupSize;
	}

	// Once the amount of partial sums in the output array is less then the work group size it will
	// then run once more to return a full reduced sum.
	callKernel(kernel_sum, queue);

	// The output buffer is read into the output vector.
	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);

	// The first value of the output array contains the sum of the whole input array which is
	// then returned.
	return (float)outputData[0];

}

//////////////////////////////// MINIMUM and MAXIMUM ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

float MinimumMaximum(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype>& inputData, int choice)
{

	size_t input_size = inputData.size() * sizeof(mytype);
	vector<mytype> outputData(inputData.size());
	size_t output_size = outputData.size() * sizeof(mytype);
	size_t N = inputData.size() / workGroupSize;

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);

	// Arguments are the same as the sum kernel except the choice argument is passed.
	// If choice == 1 then it will calculate the minimum, else it will calculate the max.
	cl::Kernel kernel_minmax = cl::Kernel(program, "MinMax");
	kernel_minmax.setArg(0, inBuffer);
	kernel_minmax.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));
	kernel_minmax.setArg(2, outBuffer);
	kernel_minmax.setArg(3, choice);

	callKernel(kernel_minmax, queue);
	kernel_minmax.setArg(0, outBuffer);

	while (N > workGroupSize)
	{
		callKernel(kernel_minmax, queue);
		N = N / workGroupSize;
	}
	// Keep reducing till only the min/max of all the workgroups is returned.
	callKernel(kernel_minmax, queue);

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);

	// Return the final min/max value.
	return (float)outputData[0];

}

//////////////////////////////// STANDARD DEVIATION ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

float StandardDeviation(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype>& inputData, float& mean)
{
	size_t input_size = inputData.size() * sizeof(mytype);//size in bytes
	vector<mytype> outputData(inputData.size());
	size_t output_size = outputData.size() * sizeof(mytype);
	size_t N = inputData.size() / workGroupSize;

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, output_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);

	// Calls the squared differences kernel in which the mean is passed in as an argument.
	// This is subtracted from each of the original input values, then squared and summed up
	// to return partial sum of sqaured differences.
	cl::Kernel kernel_std = cl::Kernel(program, "SquaredDifferences");
	kernel_std.setArg(0, inBuffer);
	kernel_std.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));
	kernel_std.setArg(2, outBuffer);
	kernel_std.setArg(3, mean);

	callKernel(kernel_std, queue);

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);

	// The partial sums returned from the SquaredDifferences kernel is then fed into the Sum function
	// to return one total sum.
	float sum = Sum(context, queue, program, outputData);

	// The total is then divided by the amount of elements N and then sqaured rooted to give
	// standard deviation.
	return sqrt((sum / (initialInputSize)));
}

///////////////////////////////////////////// BITONIC SORT ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<mytype> BitonicSort(cl::Context& context, cl::CommandQueue& queue, cl::Program& program, vector<mytype>& inputData)
{
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(); // Record sorting time
	size_t input_size = inputData.size() * sizeof(mytype);
	vector<mytype> outputData(inputData.size());
	size_t output_size = outputData.size() * sizeof(mytype);
	size_t N = inputData.size() / workGroupSize;

	cl::Buffer inBuffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer outBuffer(context, CL_MEM_READ_WRITE, input_size);

	queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, input_size, &inputData[0]);
	queue.enqueueFillBuffer(outBuffer, 0, 0, output_size);

	cl::Kernel kernel_sort = cl::Kernel(program, "sort");
	kernel_sort.setArg(0, inBuffer);
	kernel_sort.setArg(1, cl::Local(sizeof(mytype) * workGroupSize));
	kernel_sort.setArg(2, outBuffer);
	kernel_sort.setArg(3, 0); // the merge argument decide what the offset is going to be.
	
	int kernelCallCounter = 0;

	int merge = 1; // Start with the offsetted iD
	callKernel(kernel_sort, queue);
	kernelCallCounter += 1;
	kernel_sort.setArg(0, outBuffer); // Rerun the kernel continuously on the output buffer.

	bool isSorted = false;

	while (isSorted == false) // While the vector isn't sorted, it will continue to call the kernel repeatedly
	{
		kernel_sort.setArg(3, merge); // Alternate between the merge value being 1 or 0.
		merge = merge == 1 ? 0 : 1;		//Doing this allows the original element and the offset to be operated on.

		callKernel(kernel_sort, queue);
		kernelCallCounter += 1;
		queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]); // Read out the buffer every time
																					 // to check that it has been sorted.
		if (is_sorted(outputData.begin(), outputData.end()) == true) // If it has been sorted then stop running the kernel.
		{
			isSorted = true;
		}

	}

	queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, output_size, &outputData[0]);

	vector<mytype> outVec; // output vector to hold the median, upper quartile and lower quartile.

	outVec.push_back(outputData[round(0.5 * initialInputSize)]);
	outVec.push_back(outputData[round(0.75 * initialInputSize)]);
	outVec.push_back(outputData[round(0.25 * initialInputSize)]);

	printf("\nNumber of kernel calls to sort: %i\n", kernelCallCounter); // track the number of times the kernel is called.
	cout << "Total sort time (microseconds):" << endl; // Print the time taken to sort the vector.
	cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count() << endl;
	return outVec;

}


////////////////////////////////////////// MAIN ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//////////////////// READING DATA FROM FILE /////////////////////////////////

	ifstream inFileStream("temp_lincolnshire.txt"); // File path
	//////////////////// OPEN FILE ////////////////////////
	inFileStream.seekg(0, ios_base::end);
	size_t size = inFileStream.tellg();	 // Finds the size of the file
	inFileStream.seekg(0, ios_base::beg);
	char * data = new char[size]; // Creates an array of that size in which all data is read into as a block.
	inFileStream.read(&data[0], size);
	//////////////////// CLOSE FILE ///////////////////////
	inFileStream.close();

	//Parse the file by keeping track of the last space before \n
	long spacePos = 0;

	// Vector to store values
	vector<mytype> inputData;

	for (long i = 0; i < size; ++i) // Iterates through the stored block of data, to separate and extract last column
	{
		char c = data[i];
		if (c == ' ')
		{
			// Finds the last space which indicates the final column.
			spacePos = i + 1;
		}
		else if (c == '\n')
		{
			int leng = i - spacePos, index = 0;

			// Finds the word between the last space and before the new line. AKA last column data.
			char* word = new char[leng];
			word[leng] = '\0';

			// Stores that data between last space and next line.
			for (int j = spacePos; j < i; j++, index++)
			{
				word[index] = data[j];
			}

			// Convert string to float values which are pushed back into the storage vector.
			inputData.push_back((strtof(word, NULL)));
		}
	}

	///////////////////////////////// END OF FILE READING ////////////////////////////////////////

	try {
		// Defining context, queue and program.
		cl::Context context = GetContext(platform_id, device_id);
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		cl::Program::Sources sources;
		AddSources(sources, "myKernels.cl");
		cl::Program program(context, sources);

		try {
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation
		//host - input
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		initialInputSize = inputData.size(); // Store input size before padding to use for accurate mean.
		workGroupSize = 1024; // Higher workgroupSize can reduces # of kernel calls potentially reducing call overhead
		size_t padding = inputData.size() % workGroupSize;

		if (padding) // Padding the input vector with 0's to ensure that the workgroup size is a factor of the vector length
		{
			std::vector<mytype> A_ext(workGroupSize - padding, 0);
			inputData.insert(inputData.end(), A_ext.begin(), A_ext.end());
		}
		numInputElements = inputData.size(); // Number of input elements.

		///////////////////////////////// DISPLAYING RESULTS ///////////////////////////////////////

		// Each of the functions defined above return the values that are then printed here.

		float total = Sum(context, queue, program, inputData);
		float mean = total / initialInputSize;
		printf("\nTotal: %f\n", total);
		printf("Mean: %f\n", mean);

		printf("\n##############################################################################\n");
		printf("\n");
		float Minimum = MinimumMaximum(context, queue, program, inputData, 1);
		printf("\nMinimum: %f\n", Minimum);

		printf("\n##############################################################################\n");
		printf("\n");
		float Maximum = MinimumMaximum(context, queue, program, inputData, 0);
		printf("\nMaxiumum: %f\n", Maximum);

		printf("\n##############################################################################\n");
		printf("\n");
		float standardDev = StandardDeviation(context, queue, program, inputData, mean);
		printf("\nStandard Deviation: %f\n", standardDev);

		printf("\n##############################################################################\n");
		printf("\nPress key to sort..\n");
		getchar(); // Wait before starting the sort (Lots of console spam).

		vector<mytype> sorted = BitonicSort(context, queue, program, inputData); // Store array containing the median,
		float median = sorted.at(0);										// upper quartile and lower quartile.
		float upperQuartile = sorted.at(1);
		float lowerQuartile = sorted.at(2);
		printf("\n##############################################################################\n");
		printf("\nMedian: %f\n", median);
		printf("Upper Quartile: %f\n", upperQuartile);
		printf("Lower Quartile: %f\n", lowerQuartile);

		// Prints the total time the program has been running for in microseconds.
		printf("\nTotal Time Elapsed (microseconds)\n");
		cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count() << endl;


	}
	catch (cl::Error err)
	{
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
}