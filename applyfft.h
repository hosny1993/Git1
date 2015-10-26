#ifndef APPLYFFT_H
#define APPLYFFT_H
#include <clFFT.h>
#include <openclsetup.h>


class ApplyFFT
{

private:

    openClSetup *clSetup_ ;

    cl_int err_;
    cl_uint numPlatoforms_;
    cl_uint numDevices_;

    cl_platform_id* platformID_;
    cl_device_id* deviceID_;
    cl_context context_;
    cl_command_queue commandQueue_;
    cl_mem inputBuffer_;

public:
    ApplyFFT();
    ~ApplyFFT();

    float* perform1DFFT(clfftPrecision precision,clfftLayout layout,
                      clfftDirection direction, size_t sizeOfData, float input[]);

    float* perform2DFFT(clfftPrecision precision, clfftLayout layout,
                                clfftDirection direction, size_t N1, size_t N2, float input[]);
    float* perform3DFFT(clfftPrecision precision, clfftLayout layout,
                                clfftDirection direction, size_t N1, size_t N2, size_t N3,float input[]);
};

#endif // APPLYFFT_H
