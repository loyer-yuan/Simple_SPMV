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
#include "mmio/mmio.h"

#define IdxType       uint32_t
#define ELLZeroIdxVal 2294967295U

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
    IdxType m;  // Number of rows
    IdxType n;  // Number of rows and columns
    unique_ptr<DType[]> data;  // Pointer to the data

    bool isCreated = false;  // Flag to check if the matrix is created

public:
    BaseMatrix() noexcept : m(0), n(0), data(nullptr), isCreated(false)
    {
    }

    virtual ~BaseMatrix() noexcept = default;

    virtual bool CreateRamdomly(
        const IdxType m, const IdxType n, const float prob, const bool isRuntimeRandom)
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

    void PrintMat() const
    {
        if (!this->IsCreated())
        {
            cerr << "Matrix not created!" << endl;
            return;
        }

        IdxType rows = GetRows();
        IdxType cols = GetCols();
        cout << "Matrix(" << rows << ", " << cols << "):" << endl;
        cout << "Data:" << endl;
        for (IdxType i = 0; i < rows; ++i)
        {
            for (IdxType j = 0; j < cols; ++j)
            {
                cout << std::setw(6) << std::setprecision(4) << std::fixed;
                cout << (*this)(i, j) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    [[nodiscard]] virtual DType operator()(const IdxType i, const IdxType j) const = 0;

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

        for (IdxType i = 0; i < GetRows(); ++i)
        {
            for (IdxType j = 0; j < GetCols(); ++j)
            {
                if (AllClose((*this)(i, j), other(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    inline virtual bool IsCreated() const
    {
        return isCreated && m > 0 && n > 0 && data.get() != nullptr;
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

    [[nodiscard]] inline IdxType GetRows() const
    {
        return m;
    }

    [[nodiscard]] inline IdxType GetCols() const
    {
        return n;
    }

    [[nodiscard]] inline IdxType GetSize() const
    {
        return m * n;
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
    IdxType lda = 0;  // Leading dimension of the matrix

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
    DMatrix() noexcept : BaseMatrix<DType>(), mInfo{}
    {
    }

    ~DMatrix() noexcept override = default;

    bool CreateRamdomly(
        const IdxType m, const IdxType n, const float prob,
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
        const IdxType i, const IdxType j) const override
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
    IdxType nnz = 0;  // Number of non-zero elements

    std::unique_ptr<IdxType[]> rowIdx = nullptr;  // Row indices
    std::unique_ptr<IdxType[]> colIdx = nullptr;  // Column indices

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
    SPMatrixCOO() noexcept : BaseMatrix<DType>(), mInfo{}
    {
    }

    ~SPMatrixCOO() noexcept override = default;

    bool CreateRamdomly(
        const IdxType m, const IdxType n, const float prob,
        const bool isRuntimeRandom) override
    {
        if (this->IsCreated())
        {
            cerr << "SPMatrixCOO already created!" << endl;
            return false;
        }

        this->m = m;
        this->n = n;
        this->mInfo.nnz = static_cast<IdxType>(m * n * prob);
        this->mInfo.rowIdx = make_unique<IdxType[]>(this->mInfo.nnz);
        this->mInfo.colIdx = make_unique<IdxType[]>(this->mInfo.nnz);
        this->data = make_unique<DType[]>(this->mInfo.nnz);

        if (GenerateData<DType>(
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

    bool CreateFromFile(const string &filename)
    {
        MM_typecode matcode;
        FILE *f = fopen(filename.c_str(), "r");
        IdxType lM, lN, lNNZ;

        if (f == nullptr)
        {
            cerr << "Error: Could not open file!" << endl;
            return false;
        }
        if (mm_read_banner(f, &matcode) != 0)
        {
            cerr << "Error: Could not process Matrix Market banner!" << endl;
            fclose(f);
            return false;
        }

        // clang-format off
        if (!mm_is_valid(matcode) &&
            !mm_is_matrix(matcode) && !mm_is_coordinate(matcode) &&
            !(mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) &&
            !(mm_is_general(matcode) || mm_is_symmetric(matcode)))
        // clang-format on
        {
            cerr << "Error: Unsupported Matrix Market type!" << endl;
            fclose(f);
            return false;
        }

        typedef struct
        {
            IdxType rowIdx;
            IdxType colIdx;
            DType value;
        } Element;

        {
            int m, n, nnz;
            if (mm_read_mtx_crd_size(f, &m, &n, &nnz) != 0)
            {
                cerr << "Error: Could not parse matrix size!" << endl;
                fclose(f);
                return false;
            }
            lM = static_cast<IdxType>(m);
            lN = static_cast<IdxType>(n);
            lNNZ = static_cast<IdxType>(nnz);
            if (lM <= 0 || lN <= 0 || lNNZ <= 0)
            {
                cerr << "Error: Invalid matrix size!" << endl;
                fclose(f);
                return false;
            }
        }
        vector<Element> elements(lNNZ);

        cout << "Reading COO file..." << endl;
        IdxType lrowIdx, lcolIdx;
        if (mm_is_pattern(matcode))
        {
            for (IdxType i = 0; i < lNNZ; ++i)
            {
                fscanf(f, "%u %u\n", &lrowIdx, &lcolIdx);
                elements[i].rowIdx = lrowIdx - 1;  // Convert to 0-based index
                elements[i].colIdx = lcolIdx - 1;  // Convert to 0-based index
            }
        }
        else if (mm_is_integer(matcode))
        {
            int val = 0;
            for (IdxType i = 0; i < lNNZ; ++i)
            {
                fscanf(f, "%u %u %d\n", &lrowIdx, &lcolIdx, &val);
                elements[i].rowIdx = lrowIdx - 1;  // Convert to 0-based index
                elements[i].colIdx = lcolIdx - 1;  // Convert to 0-based index
                elements[i].value = static_cast<DType>(val);
            }
        }
        else if (mm_is_real(matcode))
        {
            double val = 0.0;
            for (IdxType i = 0; i < lNNZ; ++i)
            {
                fscanf(f, "%u %u %lg\n", &lrowIdx, &lcolIdx, &val);
                elements[i].rowIdx = lrowIdx - 1;  // Convert to 0-based index
                elements[i].colIdx = lcolIdx - 1;  // Convert to 0-based index
                elements[i].value = static_cast<DType>(val);
            }
        }
        else [[unlikely]]
        {
            cerr << "Error: Unsupported data type!" << endl;
            fclose(f);
            return false;
        }
        fclose(f);

        cout << "Sorting the data..." << endl;
        // Sort the elements based on row and column indices
        std::sort(
            elements.begin(), elements.end(),
            [](const Element &a, const Element &b)
            {
                if (a.rowIdx != b.rowIdx)
                    return a.rowIdx < b.rowIdx;
                return a.colIdx < b.colIdx;
            });

        cout << "Copy the data..." << endl;
        this->m = lM;
        this->n = lN;
        this->mInfo.nnz = lNNZ;
        this->mInfo.rowIdx = make_unique<IdxType[]>(lNNZ);
        this->mInfo.colIdx = make_unique<IdxType[]>(lNNZ);
        this->data = make_unique<DType[]>(lNNZ);
        if (mm_is_pattern(matcode))
        {
            for (IdxType i = 0; i < lNNZ; ++i)
            {
                this->mInfo.rowIdx[i] = elements[i].rowIdx;
                this->mInfo.colIdx[i] = elements[i].colIdx;
                this->data[i] = 1.01f;  // Set value to 1 for pattern matrix
            }
        }
        else
        {
            for (IdxType i = 0; i < lNNZ; ++i)
            {
                this->mInfo.rowIdx[i] = elements[i].rowIdx;
                this->mInfo.colIdx[i] = elements[i].colIdx;
                this->data[i] = elements[i].value;
            }
        }

        this->isCreated = true;
        this->isSorted = (is_sorted(
            this->mInfo.rowIdx.get(), this->mInfo.rowIdx.get() + this->mInfo.nnz));
        assertm(
            this->isSorted,
            "Error: Generated indices are not sorted! "
            "Please check the data generation process.");
        cout << "Matrix created from file(" << filename << ") successfully!" << endl;
        return true;
    }

    [[nodiscard]] DType operator()(const IdxType i, const IdxType j) const override
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
            const IdxType *rowStart = this->mInfo.rowIdx.get();
            const IdxType *rowEnd = this->mInfo.rowIdx.get() + this->mInfo.nnz;

            const IdxType *rowLow = lower_bound(rowStart, rowEnd, i);
            const IdxType *rowUp = upper_bound(rowStart, rowEnd, i);

            if (rowLow >= rowEnd)
                return (DType)0.0f;

            const IdxType rowIdxStartOffset = static_cast<IdxType>(rowLow - rowStart);
            const IdxType rowIdxEndOffset = static_cast<IdxType>(rowUp - rowStart);

            const IdxType *colStart = this->mInfo.colIdx.get() + rowIdxStartOffset;
            const IdxType *colEnd = this->mInfo.colIdx.get() + rowIdxEndOffset;

            const IdxType *colLow = lower_bound(colStart, colEnd, j);

            if (colLow == colEnd || *colLow != j)
                return (DType)0.0f;

            const IdxType *colUp = upper_bound(colStart, colEnd, j);

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

    bool IsCreated() const override
    {
        return this->isCreated && this->m > 0 && this->n > 0 &&
               this->data.get() != nullptr && this->mInfo.rowIdx.get() != nullptr &&
               this->mInfo.colIdx.get() != nullptr;
    }

    [[nodiscard]] inline IdxType GetNNZ() const
    {
        return mInfo.nnz;
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

        IdxType rows = this->GetRows();
        IdxType cols = this->GetCols();
        cout << "Matrix(" << rows << ", " << cols << ") NNZ: " << this->mInfo.nnz
             << endl;
        cout << "Data:" << endl;
        cout << "RowIdx ColIdx Value" << endl;
        cout << "---------------------" << endl;
        for (IdxType i = 0; i < this->mInfo.nnz; ++i)
        {
            cout << std::setw(6) << std::setprecision(4) << std::fixed;
            cout << this->mInfo.rowIdx[i] << ' ' << this->mInfo.colIdx[i] << ' '
                 << this->data[i] << endl;
        }
        cout << endl;
    }

    [[nodiscard]] inline size_t GetEffectiveSizeInBytes() const
    {
        return this->mInfo.nnz * sizeof(DType);
    }

    [[nodiscard]] inline size_t GetAllSizeInBytes() const
    {
        return this->mInfo.nnz * (sizeof(DType) + sizeof(IdxType) * 2);
    }

    [[nodiscard]] inline size_t GetTheoreticalFlops() const
    {
        return this->mInfo.nnz * 2;  // 2 flops for each non-zero element
    }

private:
    bool isSorted = false;  // Flag to check if the matrix is row-major sorted

};  // class SPMatrixCOO

//
// Sparse Matrix - CSR
//

template <>
struct MatInfo<SPMatF, SPMatF::SPMatFormatCSR>
{
    IdxType nnz = 0;  // Number of non-zero elements

    std::unique_ptr<IdxType[]> rowPtr = nullptr;  // Row pointers
    std::unique_ptr<IdxType[]> colIdx = nullptr;  // Column indices

    static constexpr const char *GetFormatName()
    {
        return "CSR";
    }
};

template <typename DType>
class SPMatrixCSR : public BaseMatrix<DType>
{
    typedef MatInfo<SPMatF, SPMatF::SPMatFormatCSR>
        MInfoType;  // Matrix information type
public:
    MInfoType mInfo;  // Matrix information

public:
    SPMatrixCSR() noexcept : BaseMatrix<DType>(), mInfo{}
    {
    }

    ~SPMatrixCSR() noexcept override = default;

    bool CreateRamdomly(
        const IdxType m, const IdxType n, const float prob,
        const bool isRuntimeRandom) override
    {
        cerr << "CSR:CreateRamdomly not implemented!" << endl;
        return false;
    }

    bool CreateFromCOO(const SPMatrixCOO<DType> &matCoo)
    {
        if (this->IsCreated())
        {
            cerr << "SPMatrixCSR already created!" << endl;
            return false;
        }
        if (!matCoo.IsCreated() || !matCoo.IsSorted())
        {
            if (!matCoo.IsCreated())
                cerr << "SPMatrixCOO not created!" << endl;
            if (!matCoo.IsSorted())
                cerr << "SPMatrixCOO not sorted!" << endl;
            return false;
        }

        this->m = matCoo.GetRows();
        this->n = matCoo.GetCols();
        this->mInfo.nnz = matCoo.mInfo.nnz;

        this->mInfo.rowPtr = make_unique<IdxType[]>(this->m + 1);
        this->mInfo.colIdx = make_unique<IdxType[]>(this->mInfo.nnz);
        this->data = make_unique<DType[]>(this->mInfo.nnz);

        IdxType rowIdxCoo = 0;
        this->mInfo.rowPtr[0] = 0;
        for (IdxType i = 0; i < this->m; ++i)
        {
            while (rowIdxCoo < this->mInfo.nnz && matCoo.mInfo.rowIdx[rowIdxCoo] == i)
            {
                ++rowIdxCoo;
            }
            this->mInfo.rowPtr[i + 1] = rowIdxCoo;
        }
        memcpy(
            this->mInfo.colIdx.get(), matCoo.mInfo.colIdx.get(),
            this->mInfo.nnz * sizeof(IdxType));
        memcpy(this->data.get(), matCoo.data.get(), this->mInfo.nnz * sizeof(DType));
        this->isCreated = true;

        return true;
    }

    [[nodiscard]] DType operator()(const IdxType i, const IdxType j) const override
    {
        assertm(
            i < this->m && j < this->n && i >= 0 && j >= 0,
            "Error: Index out of bounds! Please check the indices.");
        if (!this->IsCreated()) [[unlikely]]
        {
            cerr << "Matrix not created!" << endl;
            return 0;
        }

        const IdxType rowIdxStartOffset = this->mInfo.rowPtr[i];
        const IdxType rowIdxEndOffset = this->mInfo.rowPtr[i + 1];

        if (rowIdxStartOffset == rowIdxEndOffset)
            return (DType)0.0f;

        const IdxType *colStart = this->mInfo.colIdx.get() + rowIdxStartOffset;
        const IdxType *colEnd = this->mInfo.colIdx.get() + rowIdxEndOffset;

        const IdxType *colLow = lower_bound(colStart, colEnd, j);
        if (colLow == colEnd || *colLow != j)
            return (DType)0.0f;

        const IdxType *colUp = upper_bound(colStart, colEnd, j);
        if (colUp - colLow > 1)
        {
            cerr << "Error: Have duplicate indices!" << endl;
            return 0;
        }
        return this->data[colLow - this->mInfo.colIdx.get()];
    }

    bool IsCreated() const override
    {
        return this->isCreated && this->m > 0 && this->n > 0 &&
               this->data.get() != nullptr && this->mInfo.rowPtr.get() != nullptr &&
               this->mInfo.colIdx.get() != nullptr;
    }

    bool Destroy() override
    {
        if (!this->IsCreated())
        {
            cerr << "SPMatrixCSR not created!" << endl;
            return false;
        }
        this->data.reset();  // Release the memory
        this->mInfo.rowPtr.reset();
        this->mInfo.colIdx.reset();
        this->m = 0;
        this->n = 0;
        this->mInfo.nnz = 0;
        this->isCreated = false;  // Reset the created flag
        return true;
    }

    void PrintCSR() const
    {
        if (!this->IsCreated())
        {
            cerr << "Matrix not created!" << endl;
            return;
        }

        IdxType rows = this->GetRows();
        IdxType cols = this->GetCols();
        cout << "Matrix(" << rows << ", " << cols << ") NNZ: " << this->mInfo.nnz
             << endl;
        cout << "RowIdx:" << endl;
        for (IdxType i = 0; i < rows + 1; ++i)
        {
            cout << std::setw(6) << std::setprecision(4) << std::fixed;
            cout << this->mInfo.rowPtr[i] << " ";
        }
        cout << endl;
        cout << "ColIdx Value" << endl;
        cout << "---------------------" << endl;
        for (IdxType i = 0; i < this->mInfo.nnz; ++i)
        {
            cout << std::setw(6) << std::setprecision(4) << std::fixed;
            cout << this->mInfo.colIdx[i] << ' ' << this->data[i] << endl;
        }
        cout << endl;
    }

    [[nodiscard]] inline IdxType GetNNZ() const
    {
        return mInfo.nnz;
    }

    [[nodiscard]] inline size_t GetEffectiveSizeInBytes() const
    {
        return this->mInfo.nnz * sizeof(DType);
    }

    [[nodiscard]] inline size_t GetAllSizeInBytes() const
    {
        return this->mInfo.nnz * (sizeof(DType) + sizeof(IdxType)) +
               (this->m + 1) * sizeof(IdxType);
    }

    [[nodiscard]] inline size_t GetTheoreticalFlops() const
    {
        return this->mInfo.nnz * 2;  // 2 flops for each non-zero element
    }
};  // class SPMatrixCSR

//
// Sparse Matrix - ELL
//

template <>
struct MatInfo<SPMatF, SPMatF::SPMatFormatELL>
{
    IdxType nnz = 0;  // Number of non-zero elements
    IdxType numRows = 0;  // Number of rows
    IdxType maxCol = 0;  // Maximum number of non-zero elements in a row

    std::unique_ptr<IdxType[]> idxMat = nullptr;  // Column indices

    static constexpr const char *GetFormatName()
    {
        return "ELL";
    }
};

template <typename DType>
class SPMatrixELL : public BaseMatrix<DType>
{
    typedef MatInfo<SPMatF, SPMatF::SPMatFormatELL>
        MInfoType;  // Matrix information type
public:
    MInfoType mInfo;  // Matrix information

public:
    SPMatrixELL() noexcept : BaseMatrix<DType>(), mInfo{}
    {
    }

    ~SPMatrixELL() noexcept override = default;

    bool CreateRamdomly(
        const IdxType m, const IdxType n, const float prob,
        const bool isRuntimeRandom) override
    {
        cerr << "ELL:CreateRamdomly not implemented!" << endl;
        return false;
    }

    bool CreateFromCOO(const SPMatrixCOO<DType> &matCoo)
    {
        if (this->IsCreated())
        {
            cerr << "SPMatrixELL already created!" << endl;
            return false;
        }
        if (!matCoo.IsCreated() || !matCoo.IsSorted())
        {
            if (!matCoo.IsCreated())
                cerr << "SPMatrixCOO not created!" << endl;
            if (!matCoo.IsSorted())
                cerr << "SPMatrixCOO not sorted!" << endl;
            return false;
        }

        this->m = matCoo.GetRows();
        this->n = matCoo.GetCols();
        this->mInfo.nnz = matCoo.mInfo.nnz;

        this->mInfo.numRows = matCoo.GetRows();

        // Count the maximum number of non-zero elements in a row
        IdxType mMaxCol = 0;
        {
            IdxType curMaxCol = 0;
            IdxType curRowIdx = matCoo.mInfo.rowIdx[0];
            for (IdxType i = 0; i < this->mInfo.nnz; ++i)
            {
                if (matCoo.mInfo.rowIdx[i] == curRowIdx)
                {
                    ++curMaxCol;
                }
                else
                {
                    mMaxCol = max(mMaxCol, curMaxCol);
                    curRowIdx = matCoo.mInfo.rowIdx[i];
                    curMaxCol = 1;
                }
            }
            mMaxCol = max(mMaxCol, curMaxCol);
        }

        this->mInfo.maxCol = mMaxCol;
        this->mInfo.idxMat = make_unique<IdxType[]>(mMaxCol * this->mInfo.numRows);
        this->data = make_unique<DType[]>(mMaxCol * this->mInfo.numRows);

        {
            // Assign the first value of each row as ELLZeroIdxVal, representing no value
            for (IdxType i = 0; i < this->mInfo.numRows; ++i)
                this->mInfo.idxMat[i * mMaxCol] = ELLZeroIdxVal;

            IdxType curRowIdx = matCoo.mInfo.rowIdx[0];
            IdxType *iPtr = this->mInfo.idxMat.get() + curRowIdx * mMaxCol;
            DType *dPtr = this->data.get() + curRowIdx * mMaxCol;

            IdxType cooRowIdx = 0u;
            IdxType cooColIdx = 0u;
            IdxType colCount = 0u;
            for (IdxType cooIdx = 0; cooIdx < this->mInfo.nnz; ++cooIdx)
            {
                cooRowIdx = matCoo.mInfo.rowIdx[cooIdx];
                cooColIdx = matCoo.mInfo.colIdx[cooIdx];
                if (curRowIdx == cooRowIdx) [[likely]]
                {
                    *iPtr = cooColIdx;
                    *dPtr = matCoo.data[cooIdx];
                    iPtr++;
                    dPtr++;
                    colCount += 1;
                }
                else [[unlikely]]
                {
                    // Transfer the current row to another row
                    assertm(
                        colCount <= mMaxCol, "Error: Column count exceeds maximum!");
                    if (colCount < mMaxCol) [[likely]]
                    {
                        *iPtr = ELLZeroIdxVal;
                    }
                    iPtr = this->mInfo.idxMat.get() + cooRowIdx * mMaxCol;
                    dPtr = this->data.get() + cooRowIdx * mMaxCol;
                    curRowIdx = cooRowIdx;

                    // Assignment
                    *iPtr = cooColIdx;
                    *dPtr = matCoo.data[cooIdx];
                    iPtr++;
                    dPtr++;
                    colCount = 1;
                }
            }
        }

        this->isCreated = true;
        return true;
    }

    [[nodiscard]] DType operator()(const IdxType i, const IdxType j) const override
    {
        assertm(
            i < this->m && j < this->n && i >= 0 && j >= 0,
            "Error: Index out of bounds! Please check the indices.");
        if (!this->IsCreated()) [[unlikely]]
        {
            cerr << "Matrix not created!" << endl;
            return 0;
        }

        IdxType *rowPtr = this->mInfo.idxMat.get() + i * this->mInfo.maxCol;
        for (IdxType icol = 0; icol < this->mInfo.maxCol; ++icol)
        {
            if (rowPtr[icol] == j) [[likely]]
            {
                return this->data[i * this->mInfo.maxCol + icol];
            }
            else if (rowPtr[icol] == ELLZeroIdxVal) [[unlikely]]
            {
                break;  // No more non-zero elements in this row
            }
        }

        return 0;
    }

    bool IsCreated() const override
    {
        return this->isCreated && this->m > 0 && this->n > 0 &&
               this->data.get() != nullptr && this->mInfo.idxMat.get() != nullptr &&
               this->mInfo.nnz > 0 && this->mInfo.numRows > 0 && this->mInfo.maxCol > 0;
    }

    bool Destroy() override
    {
        if (!this->IsCreated())
        {
            cerr << "SPMatrixELL not created!" << endl;
            return false;
        }
        this->data.reset();  // Release the memory
        this->mInfo.idxMat.reset();
        this->m = 0;
        this->n = 0;
        this->mInfo.nnz = 0;
        this->mInfo.numRows = 0;
        this->mInfo.maxCol = 0;
        this->isCreated = false;  // Reset the created flag
        return true;
    }

    void PrintELL() const
    {
        if (!this->IsCreated())
        {
            cerr << "Matrix not created!" << endl;
            return;
        }

        IdxType rows = this->GetRows();
        IdxType cols = this->GetCols();
        cout << "Matrix(" << rows << ", " << cols << ") NNZ: " << this->mInfo.nnz
             << endl;
        cout << "IdxMat:" << endl;
        for (IdxType i = 0; i < rows; ++i)
        {
            for (IdxType j = 0; j < this->mInfo.maxCol; ++j)
            {
                cout << std::setw(6) << std::setprecision(4) << std::fixed;
                cout << this->mInfo.idxMat[i * this->mInfo.maxCol + j] << " ";
            }
            cout << endl;
        }
        cout << "DataMat:" << endl;
        for (IdxType i = 0; i < rows; ++i)
        {
            for (IdxType j = 0; j < this->mInfo.maxCol; ++j)
            {
                cout << std::setw(6) << std::setprecision(4) << std::fixed;
                cout << this->data[i * this->mInfo.maxCol + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    [[nodiscard]] inline IdxType GetNNZ() const
    {
        return mInfo.nnz;
    }

    [[nodiscard]] inline IdxType GetMaxCol() const
    {
        return mInfo.maxCol;
    }

    [[nodiscard]] inline size_t GetEffectiveSizeInBytes() const
    {
        return this->mInfo.nnz * sizeof(DType);
    }

    [[nodiscard]] inline size_t GetAllSizeInBytes() const
    {
        return this->mInfo.numRows * this->mInfo.maxCol *
               (sizeof(DType) + sizeof(IdxType));
    }

    [[nodiscard]] inline size_t GetTheoreticalFlops() const
    {
        return this->mInfo.nnz * 2;  // 2 flops for each non-zero element
    }
};  // class SPMatrixELL

}  // namespace xsparse

#endif  // XSPARSE_MATRIX_HPP
