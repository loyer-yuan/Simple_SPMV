#include "Matrix.h"


int main()
{
    xsparse::SPMatrixCOO<float> spMat;
    if (!spMat.CreateRamdomly(100, 10, 0.01f, true))
    {
        std::cerr << "Failed to create SPMatrixCOO!" << std::endl;
        return -1;
    }
    spMat.PrintCOO();
    spMat.PrintAll();


    return 0;
}
