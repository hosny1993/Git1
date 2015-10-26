#include <iostream>
#include "applyfft.h"
#include <openclsetup.h>

using namespace std;

int main()
{
    ApplyFFT *fft = new ApplyFFT();

    size_t N1 = 8, N0 = 8, N2 = 4;

    float *dataVector = (float*) malloc (sizeof(float*) * N1 );
    int print_iter = 0;
     /***************************************    1D    ************************************************/
    printf("1D_FFT Input :  \n");
        while(print_iter < N1)
        {
            float x = print_iter;
            float y = print_iter*3;
            dataVector[2*print_iter  ] = x;
            dataVector[2*print_iter+1] = y;
            printf("(%f, %f) ", x, y);
            print_iter++;
        }
        printf("\n");


    dataVector = fft->perform1DFFT(CLFFT_SINGLE, CLFFT_COMPLEX_INTERLEAVED,
                                    CLFFT_FORWARD, N1, dataVector);

    printf("1D_FFT Result :  \n");
    print_iter = 0;
        while(print_iter < N1)
        {
            printf("(%f, %f) ", dataVector[2*print_iter], dataVector[2*print_iter+1]);
                    print_iter++;
        }
    printf("\n");
    dataVector = NULL;
    /***************************************    2D    ************************************************/

    printf("2D_FFT Input :  \n");
    size_t buffer_size  = N0 * N1 * 2 * sizeof(float*);
    dataVector = (float*)malloc(buffer_size);

    int i, j, k;
        i = j = 0;
        for (i=0; i < N0; ++i) {
            for (j=0; j < N1; ++j) {
                float x = 0.5f;
                float y = 0.5f;
                unsigned idx = 2*(j+i*N0);
                dataVector[idx] = x;
                dataVector[idx+1] = y;
                printf("(%f, %f) ", x, y);
            }
            printf("\n");
        }

    dataVector = fft->perform2DFFT(CLFFT_SINGLE, CLFFT_COMPLEX_INTERLEAVED,
                                     CLFFT_FORWARD, N0, N1, dataVector);
    printf("2D_FFT Result :  \n");

    i = j = 0;
        for (i=0; i<N0; ++i)
        {
            for (j=0; j<N1; ++j)
            {
                unsigned idx = 2*(j+i*N0);
                printf("(%f, %f) ", dataVector[idx], dataVector[idx+1]);
            }
            printf("\n");
        }
    printf("\n");

     /***************************************    3D    ************************************************/

    N0 = 4; N1 = 4; N2 = 4;
    printf("3D_FFT Input :  \n");
    buffer_size  = N0 * N1 * N2 * 2 * sizeof(float*);
    dataVector = (float*)malloc(buffer_size);

        i = j = k = 0;
        for (i=0; i<N0; ++i) {
            for (j=0; j<N1; ++j) {
                for (k=0; k<N2; ++k) {
                    float x = 0.0f;
                    float y = 0.0f;
                    if (i==0 && j==0 && k==0) {
                        x = y = 0.5f;
                    }
                    unsigned idx = 2*(k+j*N1+i*N0*N1);
                    dataVector[idx] = x;
                    dataVector[idx+1] = y;
                    printf("(%f, %f) ", dataVector[idx], dataVector[idx+1]);
                }
                printf("\n");
            }
            printf("\n");
        }

    dataVector = fft->perform3DFFT(CLFFT_SINGLE, CLFFT_COMPLEX_INTERLEAVED,
                                     CLFFT_FORWARD, N0, N1, N2, dataVector);

    printf("3D_FFT Result :  \n");
        i = j = k = 0;
        for (i=0; i<N0; ++i) {
            for (j=0; j<N1; ++j) {
                for (k=0; k<N2; ++k) {
                    unsigned idx = 2*(k+j*N1+i*N0*N1);
                    printf("(%f, %f) ", dataVector[idx], dataVector[idx+1]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");


    return 0;
}

