#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 50.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  /*dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);*/

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	glm::vec3 center(0.0f, 0.0f, 0.0f);
	int centerNeighbors = 0;

	// Rule 2: boids try to stay a distance d away from each other
	glm::vec3 separation(0.0f, 0.0f, 0.0f);

	// Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 cohesion(0.0f, 0.0f, 0.0f);
	int cohesionNeighbors = 0;

	glm::vec3 myPos = pos[iSelf];
	glm::vec3 myVel = vel[iSelf];
	glm::vec3 newVel(0.0f, 0.0f, 0.0f);


	for (int i = 0; i < N; i++) {
		if (i != iSelf) {
			glm::vec3 otherPos = pos[i];
			glm::vec3 otherVel = vel[i];
			
			float distance = glm::length(myPos - otherPos);

			if (distance < rule1Distance) {
				center += otherPos;
				centerNeighbors++;
			}
			if (distance < rule2Distance) {
				separation -= (otherPos - myPos);
			}
			if (distance < rule3Distance) {
				cohesion += otherVel;
				cohesionNeighbors++;
			}
		}
	}
	
	if (centerNeighbors > 0) {
		center /= centerNeighbors;
		newVel += (center - myPos) * rule1Scale;
	}
	if (cohesionNeighbors > 0) {
		cohesion /= cohesionNeighbors;
		newVel += cohesion * rule3Scale;
	}
	
	newVel += separation * rule2Scale;
	
	return newVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		// Compute a new velocity based on pos and vel1
		glm::vec3 newVelocity = computeVelocityChange(N, index, pos, vel1);
		
		// Clamp the speed 
		float speed = glm::length(newVelocity);
		if (speed > maxSpeed) {
			newVelocity = glm::normalize(newVelocity) * maxSpeed;
		}

		// Record the new velocity into vel2. not vel1 b/c we're using ping-pong buffers
		vel2[index] = newVelocity;
	}
}

/**
*	Updates vel2 into vel1
*/
__global__ void kernPingPongVelocity(int N, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		vel1[index] = vel2[index];
	}
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		// - Label each boid with the index of its grid cell.
		int x = (int)((pos[index].x - gridMin.x) * inverseCellWidth);
		int y = (int)((pos[index].y - gridMin.y) * inverseCellWidth);
		int z = (int)((pos[index].z - gridMin.z) * inverseCellWidth);
		int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);
		//printf("input is %f, %f, %f, invW is %f, xyz is %d, %d, %d, output is %d\n", pos[index].x, pos[index].y, pos[index].z, inverseCellWidth, x, y, z, gridIndex);
		gridIndices[index] = gridIndex;

		// - Set up a parallel array of integer indices as pointers to the actual
		//   boid data in pos and vel1/vel2
		indices[index] = index;
	}
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N) {
		int gridIndex = particleGridIndices[index];

		if (index == 0 || gridIndex != particleGridIndices[index - 1]) {
			gridCellStartIndices[gridIndex] = index;
			if (gridCellEndIndices[gridIndex] == -1) {
				gridCellEndIndices[gridIndex] = index;
			}
		}
		else if (gridIndex == particleGridIndices[index - 1]) {
			gridCellEndIndices[gridIndex] = imax(index, gridCellEndIndices[gridIndex]);
		}
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  const int *gridCellStartIndices, const int *gridCellEndIndices,
  const int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < N) {
		// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce the number of boids that need to be checked.
		// - Identify the grid cell that this particle is in
		glm::vec3 myPos = pos[index];
		glm::vec3 myVel = vel1[index];
		glm::vec3 newVel(0.0f, 0.0f, 0.0f);
		int x = (int)((myPos.x - gridMin.x) * inverseCellWidth);
		int y = (int)((myPos.y - gridMin.y) * inverseCellWidth);
		int z = (int)((myPos.z - gridMin.z) * inverseCellWidth);
		int myCellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

		// - Identify which cells (including this cell) may contain neighbors. This isn't always 9. 
		int cellsToCheck[9];
		for (int i = 0; i < 9; i++) {
			cellsToCheck[i] = -1;
		}

		cellsToCheck[0] = myCellIndex;
		int neighborSize = 1;
		if (x > 0) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x - 1, y, z, gridResolution);
		if (x < gridResolution - 1) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x + 1, y, z, gridResolution);
		if (y > 0) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x, y - 1, z, gridResolution);
		if (y < gridResolution - 1) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x, y + 1, z, gridResolution);
		if (z > 0) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x, y, z - 1, gridResolution);
		if (z < gridResolution - 1) cellsToCheck[neighborSize++] = gridIndex3Dto1D(x, y, z + 1, gridResolution);

		// set up variables
		glm::vec3 center(0.0f, 0.0f, 0.0f);     // rule 1
		int centerNeighbors = 0;
		glm::vec3 separation(0.0f, 0.0f, 0.0f); // rule 2
		glm::vec3 cohesion(0.0f, 0.0f, 0.0f);   // rule 3
		int cohesionNeighbors = 0;

		for (int i = 0; i < neighborSize; i++) {
			// - For each cell, read the start/end indices in the boid pointer array.
			int currentCell = cellsToCheck[i];
			if (currentCell == -1) continue;
			int startIndex = gridCellStartIndices[currentCell];
			int endIndex = gridCellEndIndices[currentCell];

			// - Access each boid in the cell and compute velocity change from the boids rules, if this boid is within the neighborhood distance.
			for (int j = startIndex; j <= endIndex; j++) {
				if (j == -1) continue;
				int otherIndex = particleArrayIndices[j];
				if (otherIndex != -1 && otherIndex != index ) {
					glm::vec3 otherPos = pos[otherIndex];
					glm::vec3 otherVel = vel1[otherIndex];

					float distance = glm::length(myPos - otherPos);

					if (distance < rule1Distance) {
						center += otherPos;
						centerNeighbors++;
					}
					if (distance < rule2Distance) {
						separation -= (otherPos - myPos);
					}
					if (distance < rule3Distance) {
						cohesion += otherVel;
						cohesionNeighbors++;
					}
				}

				if (centerNeighbors > 0) {
					center /= centerNeighbors;
					newVel += (center - myPos) * rule1Scale;
				}
				if (cohesionNeighbors > 0) {
					cohesion /= cohesionNeighbors;
					newVel += cohesion * rule3Scale;
				}

				newVel += separation * rule2Scale;
			}
		}

		// - Clamp the speed change before putting the new speed in vel2
		float speed = glm::length(newVel);
		if (speed > maxSpeed) {
			newVel = glm::normalize(newVel) * maxSpeed;
		}

		vel2[index] = newVel;
	}
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// calculate vel2
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	// TODO-1.2 ping-pong the velocity buffers
	kernPingPongVelocity << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernPingPongVelocity failed!");

}

void Boids::stepSimulationScatteredGrid(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	
	// TODO-2.1: Uniform Grid Neighbor search using Thrust sort in Parallel:
	// - label each particle with its array index as well as its grid index. Use 2x width grids.
	kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	/*std::unique_ptr<int[]>gridIndices{ new int[numObjects] };
	std::unique_ptr<int[]>arrayIndices{ new int[numObjects] };
	cudaMemcpy(gridIndices.get(), dev_particleGridIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	cudaMemcpy(arrayIndices.get(), dev_particleArrayIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");
	std::cout << "right before sort: " << std::endl;
	for (int i = 0; i < numObjects; i++) {
		std::cout << "  gridIndices: " << gridIndices[i];
		std::cout << "  arrayIndices: " << arrayIndices[i] << std::endl;
	}*/

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you are welcome to do a performance comparison.
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	/*cudaMemcpy(gridIndices.get(), dev_particleGridIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	cudaMemcpy(arrayIndices.get(), dev_particleArrayIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");
	std::cout << "after sort: " << std::endl;
	for (int i = 0; i < numObjects; i++) {
		std::cout << "  gridIndices: " << gridIndices[i];
		std::cout << "  arrayIndices: " << arrayIndices[i] << std::endl;
	}*/

	// - Naively unroll the loop for finding the start and end indices of each cell's data pointers in the array of boid indices
	kernResetIntBuffer << < fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << < fullBlocksPerGrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	kernIdentifyCellStartEnd << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	/*std::unique_ptr<int[]>starts{ new int[gridCellCount] };
	std::unique_ptr<int[]>ends{ new int[gridCellCount] };
	cudaMemcpy(starts.get(), dev_gridCellStartIndices, sizeof(int) * gridCellCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(ends.get(), dev_gridCellEndIndices, sizeof(int) * gridCellCount, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");
	std::cout << "after identify startEnd: " << std::endl;
	for (int i = 0; i < gridCellCount; i++) {
		std::cout << " cell " << i << "starts at " << starts[i] << " and ends at " << ends[i] << std::endl;
	}*/
	  
	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	  
	/*std::unique_ptr<int[]>speeds{ new int[numObjects] };
	cudaMemcpy(speeds.get(), dev_vel2, sizeof(int) * numObjects, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");
	std::cout << "right after updateVel: " << std::endl;
	for (int i = 0; i < numObjects; i++) {
		std::cout << "  vel2: " << speeds[i] << std::endl;
	}*/


	// - Update positions
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	
	// - Ping-pong buffers as needed
	kernPingPongVelocity << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernPingPongVelocity failed!");
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  glm::vec3 *dev_testPos;
  int *dev_testGrid;
  int *dev_testArray;
  int N = 4;
  float width = 1;
  float invWidth = 1.0f / width;
  int resolution = 3;
  int cellCount = resolution * resolution * resolution;

  std::unique_ptr<glm::vec3[]>testPos{ new glm::vec3[N] };
  std::unique_ptr<int[]>testGrid{ new int[cellCount] };
  std::unique_ptr<int[]>testArray{ new int[cellCount] };
   

  testPos[0] = glm::vec3(0, 0.2f, 1.2f);
  testPos[1] = glm::vec3(0, 1.1f, 2.2f);
  testPos[2] = glm::vec3(2.5f, 0.1f, 0);
  testPos[3] = glm::vec3(2.5f, 2.5f, 2.5f);


  cudaMalloc((void**)&dev_testPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_testGrid, cellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  cudaMalloc((void**)&dev_testArray, cellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before compute indeces " << std::endl;
  for (int i = 0; i < cellCount; i++) {
	  testGrid[i] = 0;
	  testArray[i] = 0;

    std::cout << "  gridIndex: " << testGrid[i];
    std::cout << "    arrayIndex: " << testArray[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_testPos, testPos.get(), sizeof(glm::vec3) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_testGrid, testGrid.get(), sizeof(int) * cellCount, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_testArray, testArray.get(), sizeof(int) * cellCount, cudaMemcpyHostToDevice);

  kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (N, resolution,
	  glm::vec3(0,0,0), invWidth, dev_testPos, dev_testArray, dev_testGrid);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(testGrid.get(), dev_testGrid, sizeof(int) * cellCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(testArray.get(), dev_testArray, sizeof(int) * cellCount, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "\nafter compute indeces " << std::endl;
  for (int i = 0; i < cellCount; i++) {
	  std::cout << "  gridIndex: " << testGrid[i];
	  std::cout << "    arrayIndex: " << testArray[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_testPos);
  cudaFree(dev_testGrid);
  cudaFree(dev_testArray);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
