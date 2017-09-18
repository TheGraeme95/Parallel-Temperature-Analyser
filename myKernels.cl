 
 //////////////////////////////////////////// SUM KERNEL ////////////////////////////////////////////////
 
__kernel void floatSum ( __global const float* input, __local float* localMem, __global float* output)
 {
	int global_ID = get_global_id(0);
	uint local_ID = get_local_id(0);
	uint group_size = get_local_size(0);

  //Pass the whole input array into local memory
	localMem[local_ID] = input[global_ID];
	barrier(CLK_LOCAL_MEM_FENCE); // Waits for everything to catch up 

	  //Each workgroup is divided into 2 parts and 2 elements from the local_ID 
	  //and stride are added together.
	  //The elements in each workgroup summed through reduction.
	  //A partial sum for eachw workgroup is left.
	for (int i = 1; i < group_size; i *= 2) {
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_size))
		{			
			localMem[local_ID] += localMem[local_ID + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Wait for both parts of the workgroup to finish.
	}           

  // When the local ID reaches 0, the resulting sum for each workgroup will be written back to the output buffer.
  //The output buffer will contain all the resulting partial sums.
	if (local_ID == 0)
		output[get_group_id(0)] = localMem[0];	
  
 }



///////////////////////////////////// KERNEL FOR GETTING MIN AND MAX ////////////////////////////////////////////////

__kernel void MinMax(__global const float* input, __local float* localMem, __global float* output, int choice)
{
	int global_ID = get_global_id(0);
	int local_ID = get_local_id(0);
	int group_Size = get_local_size(0);


	//Pass the input into local memory.
	localMem[local_ID] = input[global_ID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < group_Size; i *= 2)
	{ 
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_Size))
		{

			//If the choice argument is 1 then it will return the miniumum value
			//else it will return the maximum.
			if (choice == 1)
				{
					//Iterate through workgroup and reduce into either the maximum or minimum.
					if (localMem[local_ID] > localMem[local_ID + i])
							localMem[local_ID] = localMem[local_ID + i];			
				}
			else
				{
					if (localMem[local_ID] < localMem[local_ID + i])
							localMem[local_ID] = localMem[local_ID + i];
				}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

	}
	//For each workgroup the minimum/maximum for each workgroup is returned to the output array.
	if (!local_ID)
		output[get_group_id(0)] = localMem[0];	

}


///////////////////////////// RETURNING THE SQUARED DIFFERENCES /////////////////////////////////////////////////////////

__kernel void SquaredDifferences(__global const float* input, __local float* localMem, __global float* output, float mean) 
{
	int global_ID = get_global_id(0);
	int local_ID = get_local_id(0);
	int group_Size = get_local_size(0);

	//When passing the data into local memory, the mean is subtracted and the result is squared for each element.
	localMem[local_ID] = ((input[global_ID] - mean) * (input[global_ID] - mean));

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < group_Size; i *= 2) 
	{
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_Size))
		{
			
			localMem[local_ID] += localMem[local_ID + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Writes out the partial sums of each workgroup, which will then be summed to give the sum of 
	// squared differnces with the mean
	if (!local_ID) 
	{
		output[get_group_id(0)] = localMem[0];	
	}
}

/////////////////////////////////// BITONIC SORT //////////////////////////////////////////////////////////

__kernel void sort(__global const float* input, __local float* localMem, __global float* output, int merge)
{
    int global_ID = get_global_id(0);
    int local_ID = get_local_id(0);    
    int groupSize = get_local_size(0);
	
    int offset = global_ID + ((groupSize/2) * merge); // offset the beginning of the array based on the merge value
														// and workgroup size.

	//load into local memory
    localMem[local_ID] = input[offset];
    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all work group items to finish before continuing.

    for (int i = 1; i < groupSize; i <<= 1) // Iterate through each element within each workgroup.
    {
        bool direction = ((local_ID & (i <<1)) != 0);

        for (int inc = i; inc > 0; inc >>= 1)	//Bitonic merge
        {										//Each comparator
            int j = local_ID ^ inc;
            float i_data = localMem[local_ID]; // define two different directions effectively being the start and end
            float j_data = localMem[j];			// of the vector. Pairs off the input values into bitonic sequence.

            bool smaller = (j_data < i_data) || ( j_data == i_data && j < local_ID); // check which bitonic is smaller
            bool swap = smaller ^ (j < local_ID) ^ direction; // and what the direction is

            barrier(CLK_LOCAL_MEM_FENCE); // Wait for each item in the workgroup to be checked and swapped before continuing.

            localMem[local_ID] = (swap) ? j_data : i_data; // Swaps the data based on the size comparison and in the right direction.
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

	// Sync and write back to global memory
    output[offset] = localMem[local_ID];    

}


////////////////////////////// MINIMUM KERNEL FOR INTEGERS USING ATOMIC MIN (NOT USED) /////////////////////

 __kernel void Minimum( __global const float* input, __local int* localMem, __global float* output)
 {
	uint local_id= get_local_id(0);
	uint group_size = get_local_size(0);

    //Pass the whole input array into local memory
	localMem[local_id] = input[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

    //Each workgroup is divided into 2 parts and 2 elements from the local_ID 
	//and stride are compared.
	for (int i = 1; i < group_size; i *= 2) {
		if (!(local_id % (i * 2)) && ((local_id + i) < group_size))		
		{
			if (localMem[local_id] > localMem[local_id + i])
					localMem[local_id] = localMem[local_id + i];		

		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}           

  // The resulting minimums of each workgroup are then compared and the atomic min function returns
  // the minumum of all the workgroups.
	if (local_id == 0)
		output[get_group_id(0)] = localMem[0];	
	 
 }



//////////////////////////// SUM KERNEL FOR INTEGERS USING ATOMIC ADD (NOT USED) /////////////////////////////

__kernel void Sum(__global const int* input, __local float* localMem, __global int* output) 
{
	int global_ID = get_global_id(0);
	int local_ID = get_local_id(0);
	int group_Size = get_local_size(0);

	//cache all N values from global memory to local memory
	localMem[local_ID] = input[global_ID];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < group_Size; i *= 2) 
	{
		if (!(local_ID % (i * 2)) && ((local_ID + i) < group_Size))
		{
			//printf("scratch[%d] += scratch[%d] (%d += %d)\n", lid, lid + i, scratch[lid], scratch[lid + i]);
			localMem[local_ID] += localMem[local_ID + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!local_ID) 
	{
		atomic_add(&output[0],localMem[local_ID]);
	}
}
