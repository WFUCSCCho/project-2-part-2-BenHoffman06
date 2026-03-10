**Project 2 (Part 2): In the Interest of Time**

*Due: 3/5/2026 @ 12pm (noon)*

![FAJ7fXgWUAIQKMd](https://github.com/user-attachments/assets/525429f8-4361-48cf-9cdc-c922eb2f30b5)


**Topics**
  * NVIDIA Threads Architecture
  * CUDA Events
  * File I/O
  * Parallel Algorithm Analysis

**Background**

During class, we discussed how to implement an image blurring subroutine.  We also discussed several metrics for measuring parallel algorithm performance.

**Part 2: Image Blur**

You are given a working program that blurred an image. Apply the program to several bmp images of different sizes and graph how its performance varies with system size. Modify the program using shared memory to improve its performance and compare how its performance varies with system size as compared to the original program.

Write a brief description of the strategy you employed to improve the performance of the blur program. You will not be graded on how much your strategy improved the performance or even whether performance improved. All I am looking for here is that you gave it some thought and you were able to reason out a strategy.

**Functional Requirements**

* Command Line Argument: The program must use a command line argument that corresponds to filename.

* Error Handling: You must check for:
  * The appropriate number of command line arguments.
  * Appropriate error messages must be issued for failures, followed by a graceful exit.

**Performance Analyses**
To analyze the relative performances of the CPU vs. GPU implementations of your image blurring algorithms, you will compute the time it takes to complete an image blurring function/kernel call.  

Hint: In the execution of your GPU implementation, for accurate timing information, you must call the kernel twice because the GPU initialization takes time that is unrelated to the algorithm.


**Project Submission**

Part 2 of your project will be submitted via GitHub and the link to your repository will be submitted on Canvas.

* Compilation: Any program that does not compile or lacks correctly constructed Makefiles will not be graded.
* Comments and Style: Significant code blocks must be commented. Use recognizable variable names and avoid acronyms. Poorly commented or difficult-to-follow code may result in a lack of assistance.

**Pledged Work Policy** 
You may only seek aid from the course professor or explicitly specified assistants. Tutors are not permitted. You may discuss only basic C/C++ syntax with others. Any other discussions of the project are strictly prohibited.Your code and implementation must be the product of your own work.

**Analysis** 

I wanted to use a tiling approach similar to the matrix multiplication algorithm. After all, to process one pixel's blur you need to read 33x33 = 1089 elements, but to process 4 (next to each other in a 2x2 tile) you only need 34x34-33x33 = 67 more. These savings only grow the bigger tiles you use, so I tried to get them about as big as I could (calculating how many uchar3s could fit into shared memory, with some wiggle room.) That gives me the size of the surrounding tile I use to read from, but then this can only serve the tiles in the middle. So, these surrounding tiles must overlap a bit as they surround all the inner tile regeions, which fill the grid. This means some efficiency savings are lost, but such is life. 

This approach ends up working quite well. For small images, it's harder to tell a difference, but for larger images you can really tell -- at large image sizes my algorithm runs ~2x faster. 

![Analysis image](image.png)