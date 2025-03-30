#ifndef XSPARSE_MATRIX_HPP
#define XSPARSE_MATRIX_HPP

#define DATA_MIN 0.0f
#define DATA_MAX 10.0f

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "Math.hpp"

using namespace std;

namespace xsparse {

struct EMatType
{
    enum struct EDMatFormat : uint8_t
    {
        DMatFormatColMajor = 0,
        DMatFormatRowMajor,

        DMatFormatCount
    };  // enum EDMatFormat

    enum struct ESPMatFormat : uint8_t
    {
        SPMatFormatCOO = 0,
        SPMatFormatELL,
        SPMatFormatCSR,
        SPMatFormatDIA,

        SPMatFormatCount
    };  // enum ESPMatFormat
};  // struct EMatType

typedef EMatType::EDMatFormat DMatF;
typedef EMatType::ESPMatFormat SPMatF;

//
// Base Matrix
//

/**
 * * @brief Base matrix
 */
template <typename DType>
class BaseMatrix
{
public:
    uint32_t m;  // Number of rows
    uint32_t n;  // Number of rows and columns
    unique_ptr<DType[]> data;  // Pointer to the data

    bool isCreated = false;  // Flag to check if the matrix is created

public:
    BaseMatrix() noexcept : m(0), n(0), data(nullptr), isCreated(false)
    {
    }

    virtual ~BaseMatrix() noexcept = default;

    virtual bool CreateRamdomly(
        const uint32_t m, const uint32_t n, const float prob,
        const bool isRuntimeRandom)
    {
        if (this->IsCreated())
        {
            cerr << "BaseMatrix already created!" << endl;
            return false;
        }

        this->m = m;
        this->n = n;
        this->data = make_unique<DType[]>(m * n);

        GenerateData(this->data.get(), m * n, prob, isRuntimeRandom);
        this->isCreated = true;
        return true;
    }

    void PrintAll() const
    {
        if (!this->IsCreated())
        {
            cerr << "Matrix not created!" << endl;
            return;
        }

        uint32_t rows = GetRows();
        uint32_t cols = GetCols();
        cout << "Matrix(" << rows << ", " << cols << "):" << endl;
        cout << "Data:" << endl;
        for (uint32_t i = 0; i < rows; ++i)
        {
            for (uint32_t j = 0; j < cols; ++j)
            {
                cout << std::setw(6) << std::setprecision(4) << std::fixed;
                cout << (*this)(i, j) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    [[nodiscard]] virtual DType operator()(const uint32_t i, const uint32_t j) const
    {
        cerr << "Not implemented yet!" << endl;
        return 0;
    }

    bool operator==(const BaseMatrix &other) const
    {
        if (!this->IsCreated() || !other.IsCreated())
        {
            cerr << "Matrix not created! Cannot compare!" << endl;
            return false;
        }

        if (this->m != other.m || this->n != other.n)
        {
            return false;
        }

        for (uint32_t i = 0; i < GetRows(); ++i)
        {
            for (uint32_t j = 0; j < GetCols(); ++j)
            {
                if (AllClose((*this)(i, j), other(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    virtual bool Destroy()
    {
        if (!isCreated)
        {
            cerr << "BaseMatrix not created!" << endl;
            return false;
        }
        data.reset();  // Release the memory
        m = 0;
        n = 0;
        isCreated = false;  // Reset the created flag
        return true;
    }

    [[nodiscard]] inline uint32_t GetRows() const
    {
        return m;
    }

    [[nodiscard]] inline uint32_t GetCols() const
    {
        return n;
    }

    [[nodiscard]] inline uint32_t GetSize() const
    {
        return m * n;
    }

    inline bool IsCreated() const
    {
        return isCreated && m > 0 && n > 0 && data.get() != nullptr;
    }

};  // class BaseMatrix

//
// Matrix Information Interface
//

template <typename MatType, MatType Format>
struct MatInfo;

//
// Dense Matrix
//

template <DMatF Format>
struct MatInfo<DMatF, Format>
{
    uint32_t lda = 0;  // Leading dimension of the matrix

    static constexpr const char *GetFormatName()
    {
        if constexpr (Format == DMatF::DMatFormatRowMajor)
            return "Row Major";
        else if constexpr (Format == DMatF::DMatFormatColMajor)
            return "Column Major";
        else
            return "Unknown Format";
    }
};

template <typename DType, DMatF Format = DMatF::DMatFormatRowMajor>
class DMatrix : public BaseMatrix<DType>
{
    typedef MatInfo<DMatF, Format> MInfoType;  // Matrix information type
public:
    MInfoType mInfo;  // Matrix information

public:
    DMatrix() noexcept : BaseMatrix<DType>(), mInfo{0}
    {
    }

    ~DMatrix() noexcept override = default;

    bool CreateRamdomly(
        const uint32_t m, const uint32_t n, const float prob,
        const bool isRuntimeRandom) override
    {
        if (this->IsCreated())
        {
            cerr << "DMatrix already created!" << endl;
            return false;
        }
        if (!BaseMatrix<DType>::CreateRamdomly(m, n, prob, isRuntimeRandom))
        {
            cerr << "Failed to create DMatrix!" << endl;
            return false;
        }

        if constexpr (Format == DMatF::DMatFormatRowMajor)
            mInfo.lda = n;
        else if constexpr (Format == DMatF::DMatFormatColMajor)
            mInfo.lda = m;

        return true;
    }

    // TODO: Implement COO format creation
    bool CreateFromCOO()
    {
        if (this->IsCreated())
        {
            cerr << "DMatrix already created!" << endl;
            return false;
        }

        cerr << "CreateFromCOO not implemented!" << endl;
        return false;
    }

    [[nodiscard]] inline DType operator()(
        const uint32_t i, const uint32_t j) const override
    {
        if (!this->IsCreated()) [[unlikely]]
        {
            cerr << "Matrix not created!" << endl;
            return 0;
        }

        if constexpr (Format == DMatF::DMatFormatRowMajor)
            return this->data[i * mInfo.lda + j];
        else if constexpr (Format == DMatF::DMatFormatColMajor)
            return this->data[j * mInfo.lda + i];
        else
        {
            cerr << "Unknown format!" << endl;
            return 0;
        }
    }

};  // class DMatrix

//
// Sparse Matrix - COO
//

template <>
struct MatInfo<SPMatF, SPMatF::SPMatFormatCOO>
{
    uint32_t nnz = 0;  // Number of non-zero elements

    std::unique_ptr<uint32_t[]> rowIdx = nullptr;  // Row indices
    std::unique_ptr<uint32_t[]> colIdx = nullptr;  // Column indices

    static constexpr const char *GetFormatName()
    {
        return "COO";
    }
};

template <typename DType>
class SPMatrixCOO : public BaseMatrix<DType>
{
    typedef MatInfo<SPMatF, SPMatF::SPMatFormatCOO>
        MInfoType;  // Matrix information type

public:
    MInfoType mInfo;  // Matrix information

public:
    SPMatrixCOO() noexcept : BaseMatrix<DType>(), mInfo{0}
    {
    }

    ~SPMatrixCOO() noexcept override = default;

    bool CreateRamdomly(
        const uint32_t m, const uint32_t n, const float prob,
        const bool isRuntimeRandom) override
    {
        if (this->IsCreated())
        {
            cerr << "SPMatrixCOO already created!" << endl;
            return false;
        }

        this->m = m;
        this->n = n;
        this->mInfo.nnz = static_cast<uint32_t>(m * n * prob);
        this->mInfo.rowIdx = make_unique<uint32_t[]>(this->mInfo.nnz);
        this->mInfo.colIdx = make_unique<uint32_t[]>(this->mInfo.nnz);
        this->data = make_unique<DType[]>(this->mInfo.nnz);

        if (GenerateData(
                this->data.get(), this->mInfo.nnz, 1, isRuntimeRandom, DATA_MIN,
                DATA_MAX) != this->mInfo.nnz)
        {
            cerr << "Error: Generated data is not equal to nnz!" << endl;
            return false;
        }
        GenerateIndices2D(
            this->mInfo.rowIdx.get(), this->mInfo.colIdx.get(), m, n, this->mInfo.nnz,
            isRuntimeRandom);
        this->isCreated = true;

        this->isSorted = (is_sorted(
            this->mInfo.rowIdx.get(), this->mInfo.rowIdx.get() + this->mInfo.nnz));
        assertm(
            this->isSorted,
            "Error: Generated indices are not sorted! "
            "Please check the data generation process.");
        return true;
    }

    bool CreateFromCOOFile(const string &filename)
    {
        cerr << "CreateFromCOOFile not implemented!" << endl;
        return false;
    }

    [[nodiscard]] DType operator()(const uint32_t i, const uint32_t j) const override
    {
        assertm(
            i < this->m && j < this->n && i >= 0 && j >= 0,
            "Error: Index out of bounds! Please check the indices.");
        if (!this->IsCreated()) [[unlikely]]
        {
            cerr << "Matrix not created!" << endl;
            return 0;
        }

        if (isSorted) [[likely]]
        {
            const uint32_t *rowStart = this->mInfo.rowIdx.get();
            const uint32_t *rowEnd = this->mInfo.rowIdx.get() + this->mInfo.nnz;

            const uint32_t *rowLow = lower_bound(rowStart, rowEnd, i);
            const uint32_t *rowUp = upper_bound(rowStart, rowEnd, i);

            if (rowLow >= rowEnd)
                return (DType)0.0f;

            const uint32_t rowIdxStartOffset = static_cast<uint32_t>(rowLow - rowStart);
            const uint32_t rowIdxEndOffset = static_cast<uint32_t>(rowUp - rowStart);

            const uint32_t *colStart = this->mInfo.colIdx.get() + rowIdxStartOffset;
            const uint32_t *colEnd = this->mInfo.colIdx.get() + rowIdxEndOffset;

            const uint32_t *colLow = lower_bound(colStart, colEnd, j);

            if (colLow == colEnd || *colLow != j)
                return (DType)0.0f;

            const uint32_t *colUp = upper_bound(colStart, colEnd, j);

            if (colUp - colLow > 1)
            {
                cerr << "Error: Have duplicate indices!" << endl;
                return 0;
            }

            return this->data[colLow - this->mInfo.colIdx.get()];
        }
        else [[unlikely]]
        {
            cerr << "Error: Matrix not sorted!" << endl;
            return 0;
        }
    }

    bool Destroy() override
    {
        if (!this->IsCreated())
        {
            cerr << "SPMatrixCOO not created!" << endl;
            return false;
        }
        this->data.reset();  // Release the memory
        this->mInfo.rowIdx.reset();
        this->mInfo.colIdx.reset();
        this->m = 0;
        this->n = 0;
        this->mInfo.nnz = 0;
        this->isCreated = false;  // Reset the created flag
        this->isSorted = false;  // Reset the sorted flag
        return true;
    }

    bool IsSorted() const
    {
        return isSorted;
    }

    void PrintCOO() const
    {
        if (!this->IsCreated())
        {
            cerr << "Matrix not created!" << endl;
            return;
        }

        uint32_t rows = this->GetRows();
        uint32_t cols = this->GetCols();
        cout << "Matrix(" << rows << ", " << cols << ") NNZ: " << this->mInfo.nnz
             << endl;
        cout << "Data:" << endl;
        cout << "RowIdx ColIdx Value" << endl;
        cout << "---------------------" << endl;
        for (uint32_t i = 0; i < this->mInfo.nnz; ++i)
        {
            cout << std::setw(6) << std::setprecision(4) << std::fixed;
            cout << this->mInfo.rowIdx[i] << ' ' << this->mInfo.colIdx[i] << ' '
                 << this->data[i] << endl;
        }
        cout << endl;
    }

private:
    bool isSorted = false;  // Flag to check if the matrix is row-major sorted

};  // class SPMatrixCOO

}  // namespace xsparse

#endif  // XSPARSE_MATRIX_HPP
