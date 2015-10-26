#include "applyfft.h"

ApplyFFT::ApplyFFT()
{
    clSetup_ = new openClSetup();
    err_ = 0;
    numPlatoforms_= 0;
    numDevices_ = 0;

    platformID_ = clSetup_->_getCurrentPlatform(&numPlatoforms_, &err_);
    deviceID_ = clSetup_->_getCurrentDevicesID(platformID_[0], &numDevices_, &err_);
    context_ = clSetup_->_getCurrentContext(numDevices_, deviceID_, &err_);
    commandQueue_ = clSetup_->_getCurrentCommandQueue(context_, deviceID_[0], &err_);
}

float* ApplyFFT::perform1DFFT(clfftPrecision precision, clfftLayout layout,
                            clfftDirection direction, size_t sizeOfData, float input[])
{
    size_t N = sizeOfData ;
    float *ret = (float*) malloc (2 * N * sizeof(input));
    /************************************************************************************************/
    /*                               Setup clFFT                                                    */

    clfftStatus status;
    clfftSetupData setupData;
    status = clfftInitSetupData(&setupData);
    status = clfftSetup(&setupData);

    /************************************************************************************************/
    /*                               planhande clFFT                                                */

    clfftPlanHandle planHandle;
    size_t length[1] = {N};
    int print_iter = 0;

    /************************************************************************************************/
    /*                               create memory                                                  */

    inputBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                        2 * N * sizeof(input), NULL, &err_);
    err_ = clEnqueueWriteBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
            2 * N * sizeof(input), input, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               create plane                                                   */

    status = clfftCreateDefaultPlan(&planHandle, context_, CLFFT_1D, length);
    status = clfftSetPlanPrecision(planHandle, precision);
    status = clfftSetLayout(planHandle, layout , layout );
    status = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /************************************************************************************************/
    /*                               bake plane                                                     */

    status = clfftBakePlan(planHandle, 1, &commandQueue_, NULL, NULL);
    status = clfftEnqueueTransform(planHandle, direction, 1, &commandQueue_, 0,
                                   NULL, NULL, &inputBuffer_, NULL, NULL);
    clFinish(commandQueue_);

    clEnqueueReadBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
                        2 * N * sizeof(ret), ret, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               Free system                                                    */

    clfftDestroyPlan(&planHandle);
    clfftTeardown();
    free(input);

    return ret;
}


float* ApplyFFT::perform2DFFT(clfftPrecision precision, clfftLayout layout,
                            clfftDirection direction, size_t N1, size_t N2, float input[])
{
    float *ret = (float*) malloc (2 * N1 * N2 * sizeof(input));
    /************************************************************************************************/
    /*                               Setup clFFT                                                    */

    clfftStatus status;
    clfftSetupData setupData;
    status = clfftInitSetupData(&setupData);
    status = clfftSetup(&setupData);

    /************************************************************************************************/
    /*                               planhande clFFT                                                */

    clfftPlanHandle planHandle;
    size_t length[2] = {N1, N2};
    int print_iter = 0;

    /************************************************************************************************/
    /*                               create memory                                                  */

    inputBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                        2 * N1 * N2 * sizeof(input), NULL, &err_);
    err_ = clEnqueueWriteBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
            2 * N1 * N2 * sizeof(input), input, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               create plane                                                   */

    status = clfftCreateDefaultPlan(&planHandle, context_, CLFFT_2D, length);
    status = clfftSetPlanPrecision(planHandle, precision);
    status = clfftSetLayout(planHandle, layout , layout );
    status = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /************************************************************************************************/
    /*                               bake plane                                                     */

    status = clfftBakePlan(planHandle, 1, &commandQueue_, NULL, NULL);
    status = clfftEnqueueTransform(planHandle, direction, 1, &commandQueue_, 0,
                                   NULL, NULL, &inputBuffer_, NULL, NULL);
    clFinish(commandQueue_);

    clEnqueueReadBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
                        2 * N1 * N2 * sizeof(ret), ret, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               Free system                                                    */

    clfftDestroyPlan(&planHandle);
    clfftTeardown();
    free(input);

    return ret;
}

float* ApplyFFT::perform3DFFT(clfftPrecision precision, clfftLayout layout,
                            clfftDirection direction, size_t N1, size_t N2, size_t N3,float input[])
{
    float *ret = (float*) malloc (2 * N1 * N2 * N3 * sizeof(input));
    /************************************************************************************************/
    /*                               Setup clFFT                                                    */

    clfftStatus status;
    clfftSetupData setupData;
    status = clfftInitSetupData(&setupData);
    status = clfftSetup(&setupData);

    /************************************************************************************************/
    /*                               planhande clFFT                                                */

    clfftPlanHandle planHandle;
    size_t length[3] = {N1, N2, N3};
    int print_iter = 0;

    /************************************************************************************************/
    /*                               create memory                                                  */

    inputBuffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                        2 * N1 * N2 * N3 * sizeof(input), NULL, &err_);
    err_ = clEnqueueWriteBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
            2 * N1 * N2 * N3 * sizeof(input), input, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               create plane                                                   */

    status = clfftCreateDefaultPlan(&planHandle, context_, CLFFT_3D, length);
    status = clfftSetPlanPrecision(planHandle, precision);
    status = clfftSetLayout(planHandle, layout , layout );
    status = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /************************************************************************************************/
    /*                               bake plane                                                     */

    status = clfftBakePlan(planHandle, 1, &commandQueue_, NULL, NULL);
    status = clfftEnqueueTransform(planHandle, direction, 1, &commandQueue_, 0,
                                   NULL, NULL, &inputBuffer_, NULL, NULL);
    clFinish(commandQueue_);

    clEnqueueReadBuffer(commandQueue_, inputBuffer_, CL_TRUE, 0,
                        2 * N1 * N2 * N3 * sizeof(ret), ret, 0, NULL, NULL);

    /************************************************************************************************/
    /*                               Free system                                                    */

    clfftDestroyPlan(&planHandle);
    clfftTeardown();
    free(input);

    return ret;
}

ApplyFFT::~ApplyFFT()
{
    clReleaseMemObject(inputBuffer_);
    clReleaseContext(context_);
    clReleaseDevice(deviceID_[0]);
    clReleaseCommandQueue(commandQueue_);
}
