// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
// 

#if !defined(KRATOS_SPARSE_MATRIX_MULTIPLICATION_UTILITY_H_INCLUDED )
#define  KRATOS_SPARSE_MATRIX_MULTIPLICATION_UTILITY_H_INCLUDED

// System includes
#include <vector>
#include <math.h>
#include <algorithm>
#include <bits/stl_numeric.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// External includes
#include "amgcl/value_type/interface.hpp"

// Project includes
#include "includes/define.h"

namespace Kratos
{
///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{
    
///@}
///@name  Enum's
///@{
    
///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{
    
/** 
 * @class SparseMatrixMultiplicationUtility 
 * @ingroup ContactStructuralMechanicsApplication
 * @brief An utility to multiply sparse matrix in Ublas
 * @details Taken and adapted for ublas from external_libraries/amgcl/detail/spgemm.hpp by Denis Demidov <dennis.demidov@gmail.com>
 * @todo Remove as soon as we do not depend of Ublas anymore...
 * @author Vicente Mataix Ferrandiz
 */
class SparseMatrixMultiplicationUtility
{
public:
    ///@name Type Definitions
    ///@{
    
    // d
    /// Pointer definition of TreeContactSearch
    KRATOS_CLASS_POINTER_DEFINITION( SparseMatrixMultiplicationUtility );
      
    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor
    SparseMatrixMultiplicationUtility(){};
    
    /// Desctructor
    virtual ~SparseMatrixMultiplicationUtility()= default;;
    
    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{
    
    /// Metafunction that returns value type of a matrix or a vector type.
    template <class T, class Enable = void>
    struct value_type {
        typedef typename T::value_type type;
    };
    
    /**
     * @brief The first is an OpenMP-enabled modification of classic algorithm from Saad
     * @details It is used whenever number of OpenMP cores is 4 or less. Saad, Yousef. Iterative methods for sparse linear systems. Siam, 2003.
     * @param A The first matrix to multiply
     * @param B The second matrix to multiply
     * @param C The resulting matrix
     */
    template <class AMatrix, class BMatrix, class CMatrix>
    static void MatrixMultiplicationSaad(
        const AMatrix& A, 
        const BMatrix& B, 
        CMatrix& C
        )
    {
        typedef typename value_type<CMatrix>::type Val;
        typedef std::ptrdiff_t Idx;

        // Auxiliar sizes
        const std::size_t nrows = A.size1();
        const std::size_t ncols = B.size2();
        
        // We check the size
        if (C.size1() != nrows || C.size2() != ncols)
            C.resize(nrows, ncols, false);
        
        // Get access to A, B and C data
        const std::size_t* index1_a = A.index1_data().begin();
        const std::size_t* index2_a = A.index2_data().begin();
        const double* values_a = A.value_data().begin();
        const std::size_t* index1_b = B.index1_data().begin();
        const std::size_t* index2_b = B.index2_data().begin();
        const double* values_b = B.value_data().begin();
        std::ptrdiff_t* c_ptr = new std::ptrdiff_t[nrows + 1];
        
        c_ptr[0] = 0;
        
        #pragma omp parallel
        {
            std::vector<std::ptrdiff_t> marker(ncols, -1);

            #pragma omp for
            for(Idx ia = 0; ia < static_cast<Idx>(nrows); ++ia) {
                Idx row_begin_a = index1_a[ia];
                Idx row_end_a   = index1_a[ia+1];
            
                Idx C_cols = 0;
                for(Idx ja = row_begin_a; ja < row_end_a; ++ja) {
                    Idx ca = index2_a[ja];
                    Idx row_begin_b = index1_b[ca];
                    Idx row_end_b   = index1_b[ca+1];
                    
                    for(Idx jb = row_begin_b; jb < row_end_b; ++jb) {
                        Idx cb = index2_b[jb];
                        if (marker[cb] != ia) {
                            marker[cb]  = ia;
                            ++C_cols;
                        }
                    }
                }
                c_ptr[ia + 1] = C_cols;
            }
        }
                
        // We initialize the sparse matrix
        std::partial_sum(c_ptr, c_ptr + nrows + 1, c_ptr);
        const std::size_t nonzero_values = c_ptr[nrows];
        Idx* aux_index1_c = new Idx[nonzero_values];
        Idx* aux_index2_c = new Idx[nonzero_values];
        Val* aux_val_c = new Val[nonzero_values];
        
        #pragma omp parallel
        {
            std::vector<std::ptrdiff_t> marker(ncols, -1);

            #pragma omp for
            for(Idx ia = 0; ia < static_cast<Idx>(nrows); ++ia) {
                Idx row_begin_a = index1_a[ia];
                Idx row_end_a   = index1_a[ia+1];
                
                Idx row_beg = c_ptr[ia];
                Idx row_end = row_beg;

                for(Idx ja = row_begin_a; ja < row_end_a; ++ja) {
                    const Idx ca = index2_a[ja];
                    const Val va = values_a[ja];
                    
                    Idx row_begin_b = index1_b[ca];
                    Idx row_end_b   = index1_b[ca+1];

                    for(Idx jb = row_begin_b; jb < row_end_b; ++jb) {
                        const Idx cb = index2_b[jb];
                        const Val vb = values_b[jb];

                        if (marker[cb] < row_beg) {
                            marker[cb] = row_end;
                            aux_index1_c[row_end] = ia;
                            aux_index2_c[row_end] = cb;
                            aux_val_c[row_end] = va * vb;
                            ++row_end;
                        } else {
                            aux_val_c[marker[cb]] += va * vb;
                        }
                    }
                }
                
                SortRow(aux_index2_c + row_beg, aux_val_c + row_beg, row_end - row_beg);
            }
        }
        
        // We finally push back
        for (std::size_t i = 0; i < nonzero_values; i++) {
            C.push_back(aux_index1_c[i], aux_index2_c[i], aux_val_c[i]);
        }
    }

    /**
     * @brief Row-merge algorithm from Rupp et al. 
     * @details The algorithm  requires less memory and shows much better scalability than classic one. It is used when number of OpenMP cores is more than 4.
     * @param A The first matrix to multiply
     * @param B The second matrix to multiply
     * @param C The resulting matrix
     */
    template <class AMatrix, class BMatrix, class CMatrix>
    static void MatrixMultiplicationRMerge(
        const AMatrix &A, 
        const BMatrix &B, 
        CMatrix &C
        ) 
    {
        typedef typename value_type<CMatrix>::type Val;
        typedef std::size_t Idx;

        // Auxiliar sizes
        const std::size_t nrows = A.size1();
        const std::size_t ncols = B.size2();
        
        // Get access to A and B data
        const std::size_t* index1_a = A.index1_data().begin();
        const std::size_t* index2_a = A.index2_data().begin();
        const double* values_a = A.value_data().begin();
        const std::size_t* index1_b = B.index1_data().begin();
        const std::size_t* index2_b = B.index2_data().begin();
        const double* values_b = B.value_data().begin();
        
        Idx max_row_width = 0;

        #pragma omp parallel
        {
            Idx my_max = 0;

            #pragma omp for
            for(int i = 0; i < static_cast<Idx>(nrows); ++i) {
                Idx row_beg = index1_a[i];
                Idx row_end = index1_a[i+1];
                
                Idx row_width = 0;
                for(Idx j = row_beg; j < row_end; ++j) {
                    Idx a_col = index2_a[j];
                    row_width += index1_b[a_col + 1] - index1_b[a_col];
                }
                my_max = std::max(my_max, row_width);
            }

            #pragma omp critical
            max_row_width = std::max(max_row_width, my_max);
        }

    #ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
    #else
        const int nthreads = 1;
    #endif

        std::vector< std::vector<Idx> > tmp_col(nthreads);
        std::vector< std::vector<Val> > tmp_val(nthreads);

        for(int i = 0; i < nthreads; ++i) {
            tmp_col[i].resize(3 * max_row_width);
            tmp_val[i].resize(2 * max_row_width);
        }
        
        // We check the size
        if (C.size1() != nrows || C.size2() != ncols)
            C.resize(nrows, ncols, false);
        
        // We create the c_ptr auxiliar variable
        std::size_t* c_ptr = new std::size_t[nrows + 1];        
        c_ptr[0] = 0;

        #pragma omp parallel
        {
        #ifdef _OPENMP
            const int tid = omp_get_thread_num();
        #else
            const int tid = 0;
        #endif

            Idx* t_col = &tmp_col[tid][0];

            #pragma omp for
            for(Idx i = 0; i < static_cast<Idx>(nrows); ++i) {
                Idx row_beg = index1_a[i];
                Idx row_end = index1_a[i+1];

                c_ptr[i+1] = ProdRowWidth( index2_a + row_beg, index2_a + row_end, index1_b, index2_b, t_col, t_col + max_row_width, t_col + 2 * max_row_width );
            }
        }

        // We initialize the sparse matrix
        std::partial_sum(c_ptr, c_ptr + nrows + 1, c_ptr);
        const std::size_t nonzero_values = c_ptr[nrows];
        Idx* aux_index1_c = new Idx[nonzero_values];
        Idx* aux_index2_c = new Idx[nonzero_values];
        Val* aux_val_c = new Val[nonzero_values];
        
        #pragma omp parallel
        {
        #ifdef _OPENMP
            const int tid = omp_get_thread_num();
        #else
            const int tid = 0;
        #endif

            Idx* t_col = tmp_col[tid].data();
            Val *t_val = tmp_val[tid].data();

            #pragma omp for
            for(Idx i = 0; i < static_cast<Idx>(nrows); ++i) {
                Idx row_beg = index1_a[i];
                Idx row_end = index1_a[i+1];

                ProdRow(index2_a + row_beg, index2_a + row_end, values_a + row_beg,
                        index1_b, index2_b, values_b, aux_index2_c + c_ptr[i], aux_val_c + c_ptr[i], t_col, t_val, t_col + max_row_width, t_val + max_row_width );
                
                for (std::size_t j = 0; j < c_ptr[i]; j++) {
                    aux_index1_c[j] = i;
                }
            }
        }
        
        // We finally push back
        for (std::size_t i = 0; i < nonzero_values; i++) {
            C.push_back(aux_index1_c[i], aux_index2_c[i], aux_val_c[i]);
        }
    }
    
    ///@}
    ///@name Access
    ///@{
    
    ///@}
    ///@name Inquiry
    ///@{
    
    ///@}
    ///@name Input and output
    ///@{
    
    /// Turn back information as a string.
    std::string Info() const
    {
        return "SparseMatrixMultiplicationUtility";
    }
    
    /// Print information about this object.
    void PrintInfo (std::ostream& rOStream) const
    {
        rOStream << "SparseMatrixMultiplicationUtility";
    }
    
    /// Print object's data.
    void PrintData (std::ostream& rOStream) const
    {
    }
    
    ///@}
    ///@name Friends
    ///@{
    
    ///@}
protected:
    ///@name Protected static Member Variables
    ///@{
    
    ///@}
    ///@name Protected member Variables
    ///@{
    
    ///@}
    ///@name Protected Operators
    ///@{
    
    ///@}
    ///@name Protected Operations
    ///@{
    
    ///@}
    ///@name Protected  Access
    ///@{
    
    ///@}
    ///@name Protected Inquiry
    ///@{
    
    ///@}
    ///@name Protected LifeCycle
    ///@{
    
    ///@}
private:
    ///@name Static Member Variables
    ///@{
    
    ///@}
    ///@name Member Variables
    ///@{
    
    ///@}
    ///@name Private Operators
    ///@{
    
    ///@}
    ///@name Private Operations
    ///@{
    
    /**
     * @brief This method is designed to reorder the rows by columns
     * @param Columns The columns of the problem
     * @param Values The values (to be ordered with the rows)
     * @param Size The size of the colums
     */
    template <typename Col, typename Val>
    static inline void SortRow(
        Col* Columns, 
        Val* Values, 
        const std::size_t Size
        ) 
    {
        for(std::size_t j = 1; j < Size; ++j) {
            const Col c = Columns[j];
            const Val v = Values[j];

            std::size_t i = j - 1;

            while(i >= 0 && Columns[i] > c) {
                Columns[i + 1] = Columns[i];
                Values[i + 1] = Values[i];
                i--;
            }

            Columns[i + 1] = c;
            Values[i + 1] = v;
        }
    }
    
    /**
     * 
     */
    template <bool need_out, class Idx>
    static Idx* MergeRows(
            const Idx* col1, 
            const Idx* col1_end,
            const Idx* col2, 
            const Idx* col2_end,
            Idx* col3
            )
    {
        while(col1 != col1_end && col2 != col2_end) {
            Idx c1 = *col1;
            Idx c2 = *col2;

            if (c1 < c2) {
                if (need_out) *col3 = c1;
                ++col1;
            } else if (c1 == c2) {
                if (need_out) *col3 = c1;
                ++col1;
                ++col2;
            } else {
                if (need_out) *col3 = c2;
                ++col2;
            }
            ++col3;
        }

        if (need_out) {
            if (col1 < col1_end) {
                return std::copy(col1, col1_end, col3);
            } else if (col2 < col2_end) {
                return std::copy(col2, col2_end, col3);
            } else {
                return col3;
            }
        } else {
            return col3 + (col1_end - col1) + (col2_end - col2);
        }
    }

    /**
     *  
     */
    template <class Idx, class Val>
    static Idx* MergeRows(
            const Val &alpha1, 
            const Idx* col1, 
            const Idx* col1_end, 
            const Val *val1,
            const Val &alpha2, 
            const Idx* col2, 
            const Idx* col2_end, 
            const Val *val2,
            Idx* col3, 
            Val *val3
            )
    {
        while(col1 != col1_end && col2 != col2_end) {
            Idx c1 = *col1;
            Idx c2 = *col2;

            if (c1 < c2) {
                ++col1;

                *col3 = c1;
                *val3 = alpha1 * (*val1++);
            } else if (c1 == c2) {
                ++col1;
                ++col2;

                *col3 = c1;
                *val3 = alpha1 * (*val1++) + alpha2 * (*val2++);
            } else {
                ++col2;

                *col3 = c2;
                *val3 = alpha2 * (*val2++);
            }

            ++col3;
            ++val3;
        }

        while(col1 < col1_end) {
            *col3++ = *col1++;
            *val3++ = alpha1 * (*val1++);
        }

        while(col2 < col2_end) {
            *col3++ = *col2++;
            *val3++ = alpha2 * (*val2++);
        }

        return col3;
    }

    /**
     * 
     */
    template <class Idx>
    static Idx ProdRowWidth(
            const Idx* acol, 
            const Idx* acol_end,
            const Idx* bptr, 
            const Idx* bcol,
            Idx* tmp_col1, 
            Idx* tmp_col2, 
            Idx* tmp_col3
            )
    {
        const Idx nrow = acol_end - acol;

        /* No rows to merge, nothing to do */
        if (nrow == 0) return 0;

        /* Single row, just copy it to output */
        if (nrow == 1) return bptr[*acol + 1] - bptr[*acol];

        /* Two rows, merge them */
        if (nrow == 2) {
            int a1 = acol[0];
            int a2 = acol[1];

            return MergeRows<false>( bcol + bptr[a1], bcol + bptr[a1+1], bcol + bptr[a2], bcol + bptr[a2+1], tmp_col1) - tmp_col1;
        }

        /* Generic case (more than two rows).
        *
        * Merge rows by pairs, then merge the results together.
        * When merging two rows, the result is always wider (or equal).
        * Merging by pairs allows to work with short rows as often as possible.
        */
        // Merge first two.
        Idx a1 = *acol++;
        Idx a2 = *acol++;
        Idx c_col1 = MergeRows<true>(  bcol + bptr[a1], bcol + bptr[a1+1], bcol + bptr[a2], bcol + bptr[a2+1], tmp_col1 ) - tmp_col1;

        // Go by pairs.
        while(acol + 1 < acol_end) {
            a1 = *acol++;
            a2 = *acol++;

            Idx c_col2 = MergeRows<true>(  bcol + bptr[a1], bcol + bptr[a1+1], bcol + bptr[a2], bcol + bptr[a2+1], tmp_col2 ) - tmp_col2;

            if (acol == acol_end) {
                return MergeRows<false>( tmp_col1, tmp_col1 + c_col1, tmp_col2, tmp_col2 + c_col2, tmp_col3 ) - tmp_col3;
            } else {
                c_col1 = MergeRows<true>( tmp_col1, tmp_col1 + c_col1, tmp_col2, tmp_col2 + c_col2, tmp_col3 ) - tmp_col3;

                std::swap(tmp_col1, tmp_col3);
            }
        }

        // Merge the tail.
        a2 = *acol;
        return MergeRows<false>( tmp_col1, tmp_col1 + c_col1, bcol + bptr[a2], bcol + bptr[a2+1], tmp_col2 ) - tmp_col2;
    }

    /**
     * 
     */
    template <class Idx, class Val>
    static void ProdRow(
            const Idx* acol, 
            const Idx* acol_end, 
            const Val *aval,
            const Idx* bptr, 
            const Idx* bcol, 
            const Val *bval,
            Idx* out_col, 
            Val *out_val,
            Idx* tm2_col, 
            Val *tm2_val,
            Idx* tm3_col, 
            Val *tm3_val
            )
    {
        const Idx nrow = acol_end - acol;

        /* No rows to merge, nothing to do */
        if (nrow == 0) return;

        /* Single row, just copy it to output */
        if (nrow == 1) {
            Idx ac = *acol;
            Val av = *aval;

            const Val *bv = bval + bptr[ac];
            const Idx* bc = bcol + bptr[ac];
            const Idx* be = bcol + bptr[ac+1];

            while(bc != be) {
                *out_col++ = *bc++;
                *out_val++ = av * (*bv++);
            }

            return;
        }

        /* Two rows, merge them */
        if (nrow == 2) {
            Idx ac1 = acol[0];
            Idx ac2 = acol[1];

            Val av1 = aval[0];
            Val av2 = aval[1];

            MergeRows( av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1], av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2], out_col, out_val );
        }

        /* Generic case (more than two rows).
        *
        * Merge rows by pairs, then merge the results together.
        * When merging two rows, the result is always wider (or equal).
        * Merging by pairs allows to work with short rows as often as possible.
        */
        // Merge first two.
        Idx ac1 = *acol++;
        Idx ac2 = *acol++;

        Val av1 = *aval++;
        Val av2 = *aval++;

        Idx* tm1_col = out_col;
        Val *tm1_val = out_val;

        Idx c_col1 = MergeRows( av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1], av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2], tm1_col, tm1_val ) - tm1_col;

        // Go by pairs.
        while(acol + 1 < acol_end) {
            ac1 = *acol++;
            ac2 = *acol++;

            av1 = *aval++;
            av2 = *aval++;

            Idx c_col2 = MergeRows( av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1], av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2], tm2_col, tm2_val ) - tm2_col;

            c_col1 = MergeRows( amgcl::math::identity<Val>(), tm1_col, tm1_col + c_col1, tm1_val, amgcl::math::identity<Val>(), tm2_col, tm2_col + c_col2, tm2_val, tm3_col, tm3_val ) - tm3_col;

            std::swap(tm3_col, tm1_col);
            std::swap(tm3_val, tm1_val);
        }

        // Merge the tail if there is one.
        if (acol < acol_end) {
            ac2 = *acol++;
            av2 = *aval++;

            c_col1 = MergeRows( amgcl::math::identity<Val>(), tm1_col, tm1_col + c_col1, tm1_val, av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2], tm3_col, tm3_val ) - tm3_col;

            std::swap(tm3_col, tm1_col);
            std::swap(tm3_val, tm1_val);
        }

        // If we are lucky, tm1 now points to out.
        // Otherwise, copy the results.
        if (tm1_col != out_col) {
            std::copy(tm1_col, tm1_col + c_col1, out_col);
            std::copy(tm1_val, tm1_val + c_col1, out_val);
        }
        
        return;
    }
    
    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    ///@}

}; // Class SparseMatrixMultiplicationUtility

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

// /****************************** INPUT STREAM FUNCTION ******************************/
// /***********************************************************************************/
// 
// template<class TPointType, class TPointerType>
// inline std::istream& operator >> (std::istream& rIStream,
//                                   SparseMatrixMultiplicationUtility& rThis);
// 
// /***************************** OUTPUT STREAM FUNCTION ******************************/
// /***********************************************************************************/
// 
// template<class TPointType, class TPointerType>
// inline std::ostream& operator << (std::ostream& rOStream,
//                                   const SparseMatrixMultiplicationUtility& rThis)
// {
//     return rOStream;
// }

///@}

}  // namespace Kratos.

#endif // KRATOS_TREE_CONTACT_SEARCH_H_INCLUDED  defined 
