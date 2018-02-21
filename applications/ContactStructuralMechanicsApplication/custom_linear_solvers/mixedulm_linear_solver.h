// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:             BSD License
//                                       license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#if !defined(KRATOS_MIXEDULM_SOLVER_H_INCLUDED )
#define  KRATOS_MIXEDULM_SOLVER_H_INCLUDED

// System includes
#include <string>
#include <iostream>
#include <sstream>
#include <cstddef>

// External includes
#include <boost/numeric/ublas/vector.hpp>

// Project includes
#include "includes/define.h"
#include "includes/model_part.h"
#include "linear_solvers/reorderer.h"
#include "linear_solvers/iterative_solver.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "utilities/openmp_utils.h"
#include "contact_structural_mechanics_application_variables.h"
#include "custom_utilities/sparse_matrix_multiplication_utility.h"

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
 * @class MixedULMLinearSolver
 * @ingroup ContactStructuralMechanicsApplication  
 * @brief This solver is designed for the solution of mixed U-LM problems (this solver in particular is optimized for dual LM, to avoid the resolution).
 * @details It uses a block structure diving the matrix in UU LMLM ULM LMU blocks
 * and uses "standard" linear solvers for the different blocks as well as a GMRES for the outer part
 * @author Vicente Mataix Ferrandiz
*/
template<class TSparseSpaceType, class TDenseSpaceType,
         class TPreconditionerType = Preconditioner<TSparseSpaceType, TDenseSpaceType>,
         class TReordererType = Reorderer<TSparseSpaceType, TDenseSpaceType> >
class MixedULMLinearSolver :
    public IterativeSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>
{
public:
    ///@name Type Definitions
    ///@{
    
    /// Pointer definition of MixedULMLinearSolver
    KRATOS_CLASS_POINTER_DEFINITION (MixedULMLinearSolver);
    
    /// The base class corresponds to the an iterative solver
    typedef IterativeSolver<TSparseSpaceType, TDenseSpaceType, TPreconditionerType, TReordererType> BaseType;
    
    /// The base class for the linear solver
    typedef LinearSolver<TSparseSpaceType, TDenseSpaceType, TReordererType> LinearSolverType;
    
    /// The pointer to a linear solver
    typedef typename LinearSolverType::Pointer LinearSolverPointerType;
    
    /// The sparse matrix type
    typedef typename TSparseSpaceType::MatrixType SparseMatrixType;
    
    /// The vector type
    typedef typename TSparseSpaceType::VectorType VectorType;
    
    /// The dense matrix type
    typedef typename TDenseSpaceType::MatrixType DenseMatrixType;
    
    /// The dense vector type
    typedef typename TDenseSpaceType::VectorType DenseVectorType;
    
    /// The node type
    typedef Node<3> NodeType;
    
    /// The array containing the dofs
    typedef typename ModelPart::DofsArrayType DofsArrayType;
    
    /// An array of conditions
    typedef ModelPart::ConditionsContainerType ConditionsArrayType;
    
    /// An array of nodes
    typedef ModelPart::NodesContainerType NodesArrayType;
    
    /// The size type
    typedef std::size_t  SizeType;
    
    /// The index type
    typedef std::size_t  IndexType;
    
    /// A vector of indexes
    typedef vector<IndexType> IndexVectorType;
    
    ///@}
    ///@name Enums
    ///@{

    /// This enum is used to identify each index whick kind is
    enum class BlockType {
            OTHER,
            MASTER,
            SLAVE_INACTIVE,
            SLAVE_ACTIVE,
            LM_INACTIVE,
            LM_ACTIVE
            };
    ///@}
    ///@name Life Cycle
    ///@{
    
    /**
     * @brief Default constructor
     * @param pSolverDispBlock The linear solver used for the displacement block
     * @param MaxTolerance The maximal tolrance considered
     * @param MaxIterationNumber The maximal number of iterations
     */
    MixedULMLinearSolver (
        LinearSolverPointerType pSolverDispBlock,
        const double MaxTolerance,
        const std::size_t MaxIterationNumber
        ) : BaseType (MaxTolerance, MaxIterationNumber),
            mpSolverDispBlock(pSolverDispBlock)
    {
        // Initializing the remaining variables
        mBlocksAreAllocated = false;
        mIsInitialized = false;
    }
    
    /**
     * @brief Second constructor, it uses a Kratos parameters as input instead of direct input
     * @param pSolverDispBlock The linear solver used for the displacement block
     * @param ThisParameters The configuration parameters considered
     */
    
    MixedULMLinearSolver(
            LinearSolverPointerType pSolverDispBlock,
            Parameters ThisParameters =  Parameters(R"({})")
            ): BaseType (),
               mpSolverDispBlock(pSolverDispBlock)

    {
        KRATOS_TRY

        // Now validate agains defaults -- this also ensures no type mismatch
        Parameters default_parameters = GetDefaultParameters(); 
        ThisParameters.ValidateAndAssignDefaults(default_parameters);
        
        // Initializing the remaining variables
        this->SetTolerance( ThisParameters["tolerance"].GetDouble() );
        this->SetMaxIterationsNumber( ThisParameters["max_iteration_number"].GetInt() );
        mBlocksAreAllocated = false;
        mIsInitialized = false;
        
        KRATOS_CATCH("")
    }

    
    /// Copy constructor.
    MixedULMLinearSolver (const MixedULMLinearSolver& Other)
    {
        KRATOS_ERROR << "Copy constructor not correctly implemented" << std::endl;
    }
    
    /// Destructor.
    ~MixedULMLinearSolver() override {}
    
    ///@}
    ///@name Operators
    ///@{
    
    /// Assignment operator.
    MixedULMLinearSolver& operator= (const MixedULMLinearSolver& Other)
    {
        return *this;
    }
    
    ///@}
    ///@name Operations
    ///@{
    
    /** 
     * @brief This function is designed to be called as few times as possible. It creates the data structures
     * that only depend on the connectivity of the matrix (and not on its coefficients)
     * @details So that the memory can be allocated once and expensive operations can be done only when strictly
     * needed
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
    */
    void Initialize (
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {
        if (mBlocksAreAllocated == true) {
            mpSolverDispBlock->Initialize(mKDispModified, mDisp, mResidualDisp);
            mIsInitialized = true;
        } else
            KRATOS_WARNING("MixedULM Initialize") << "Linear solver intialization is deferred to the moment at which blocks are available" << std::endl;
    }
    
    /** 
     * @brief This function is designed to be called every time the coefficients change in the system
     * that is, normally at the beginning of each solve.
     * @details For example if we are implementing a direct solver, this is the place to do the factorization
     * so that then the backward substitution can be performed effectively more than once
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
    */
    void InitializeSolutionStep (
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {     
        // Copy to local matrices
        if (mBlocksAreAllocated == false) {
            FillBlockMatrices (true, rA, rX, rB);
            mBlocksAreAllocated = true;
        } else {
            FillBlockMatrices (false, rA, rX, rB);
            mBlocksAreAllocated = true;
        }
        
        if(mIsInitialized == false) 
            this->Initialize(rA,rX,rB);

        mpSolverDispBlock->InitializeSolutionStep(mKDispModified, mDisp, mResidualDisp);
    }

    /** 
     * @brief This function actually performs the solution work, eventually taking advantage of what was done before in the
     * @details Initialize and InitializeSolutionStep functions.
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
    */
    void PerformSolutionStep (
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {
        // Auxiliar size
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();

        // Get the u and lm residuals
        GetUPart (rB, mResidualDisp);

        // Solve u block
        noalias(mDisp)  = ZeroVector(total_size);
        mpSolverDispBlock->Solve (mKDispModified, mDisp, mResidualDisp);

        // Now we compute the residual of the LM
        GetLMAPart (rB, mResidualLMActive);
        GetLMIPart (rB, mResidualLMInactive);

        // Solve LM
        // LM = D⁻1*rLM
        noalias(mLMActive) = ZeroVector(mLMActiveIndices.size());
        noalias(mLMInactive) = ZeroVector(mLMInactiveIndices.size());
        TSparseSpaceType::Mult (mKLMAModified, mResidualLMActive, mLMActive);
        TSparseSpaceType::Mult (mKLMIModified, mResidualLMInactive, mLMInactive);

        // Write back solution
        SetUPart(rX, mDisp);
        SetLMAPart(rX, mLMActive);
        SetLMIPart(rX, mLMInactive);
    }
    
    /** 
     * @brief This function is designed to be called at the end of the solve step.
     * @details For example this is the place to remove any data that we do not want to save for later
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
    */
    void FinalizeSolutionStep (
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {
        mpSolverDispBlock->FinalizeSolutionStep(mKDispModified, mDisp, mResidualDisp);
    }
    
    /** 
     * @brief This function is designed to clean up all internal data in the solver.
     * @details Clear is designed to leave the solver object as if newly created. After a clear a new Initialize is needed
     */
    void Clear() override
    {        
        mKDispModified.clear(); /// The modified displacement block
        mKLMAModified.clear();  /// The modified LM block (diagonal)
        mBlocksAreAllocated = false;
        mpSolverDispBlock->Clear();
        mDisp.clear();
        mLMActive.clear();
        mLMInactive.clear();
        mResidualDisp.clear();
        mResidualLMActive.clear();
        mIsInitialized = false;
    }

    /** 
     * @brief Normal solve method.
     * @details Solves the linear system Ax=b and puts the result on SystemVector& rX. rVectorx is also th initial guess for iterative methods.
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
     */
    bool Solve(
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {
        if (mIsInitialized == false)
            this->Initialize (rA,rX,rB);

        this->InitializeSolutionStep (rA,rX,rB);

        this->PerformSolutionStep (rA,rX,rB);

        this->FinalizeSolutionStep (rA,rX,rB);

        return false;
    }

    /** 
     * @brief Multi solve method for solving a set of linear systems with same coefficient matrix.
     * @details Solves the linear system Ax=b and puts the result on SystemVector& rX. rVectorx is also th initial guess for iterative methods.
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
     */
    bool Solve (
        SparseMatrixType& rA, 
        DenseMatrixType& rX, 
        DenseMatrixType& rB
        ) override
    {
        return false;
    }

    /** 
     * @brief Some solvers may require a minimum degree of knowledge of the structure of the matrix. To make an example
     * when solving a mixed u-p problem, it is important to identify the row associated to v and p.
     * @details Another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers
     * which require knowledge on the spatial position of the nodes associated to a given dof.
     * This function tells if the solver requires such data
     */
    bool AdditionalPhysicalDataIsNeeded() override
    {
        return true;
    }

    /** 
     * @brief Some solvers may require a minimum degree of knowledge of the structure of the matrix. 
     * @details To make an example when solving a mixed u-p problem, it is important to identify the row associated to v and p. Another example is the automatic prescription of rotation null-space for smoothed-aggregation solvers which require knowledge on the spatial position of the nodes associated to a given dof. This function is the place to eventually provide such data
     * @param rA System matrix
     * @param rX Solution vector. It's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
     * @todo If there are problems with active/inactive just fix dofs
     */
    void ProvideAdditionalData (
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        DofsArrayType& rDofSet,
        ModelPart& rModelPart
        ) override
    {
        // Allocating auxiliar parameters
        IndexType node_id;
        NodeType::Pointer pnode;
        
        // Count LM dofs
        SizeType n_lm_inactive_dofs = 0, n_lm_active_dofs = 0;
        SizeType n_master_dofs = 0;
        SizeType n_slave_inactive_dofs = 0, n_slave_active_dofs = 0;
        SizeType tot_active_dofs = 0;
        for (auto& i_dof : rDofSet) {
            node_id = i_dof.Id();
            pnode = rModelPart.pGetNode(node_id);
            if (i_dof.EquationId() < rA.size1()) {
                tot_active_dofs++;
                if (i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_X || 
                    i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Y || 
                    i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Z) {
                    if (pnode->Is(ACTIVE))
                        n_lm_active_dofs++;
                    else
                        n_lm_inactive_dofs++;
                } else if (i_dof.GetVariable().Key() == DISPLACEMENT_X || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Y || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Z) {
                    if (pnode->Is(INTERFACE)) {
                        if (pnode->Is(MASTER)) {
                            n_master_dofs++;
                        } else if (pnode->Is(SLAVE)) {
                            if (pnode->Is(ACTIVE))
                                n_slave_active_dofs++;
                            else
                                n_slave_inactive_dofs++;
                        }
                    }
                }
            }
        }

        KRATOS_ERROR_IF(tot_active_dofs != rA.size1()) << "Total system size does not coincide with the free dof map" << std::endl;

        // Resize arrays as needed
        mMasterIndices.resize (n_master_dofs,false);
        mSlaveInactiveIndices.resize (n_slave_inactive_dofs,false);
        mSlaveActiveIndices.resize (n_slave_active_dofs,false);
        mLMInactiveIndices.resize (n_lm_inactive_dofs,false);
        mLMActiveIndices.resize (n_lm_active_dofs,false);

        const SizeType other_dof_size = tot_active_dofs - n_lm_inactive_dofs - n_lm_active_dofs - n_master_dofs - n_slave_inactive_dofs - n_slave_active_dofs;
        mOtherIndices.resize (other_dof_size,false);
        mGlobalToLocalIndexing.resize (tot_active_dofs,false);
        mWhichBlockType.resize (tot_active_dofs, BlockType::OTHER);
        
        /**
         * Construct aux_lists as needed
         * "other_counter[i]" i will contain the position in the global system of the i-th NON-LM node
         * "lm_active_counter[i]" will contain the in the global system of the i-th NON-LM node
         * mGlobalToLocalIndexing[i] will contain the position in the local blocks of the
         */
        SizeType lm_inactive_counter = 0, lm_active_counter = 0;
        SizeType master_counter = 0;
        SizeType slave_inactive_counter = 0, slave_active_counter = 0;
        SizeType other_counter = 0;
        IndexType global_pos = 0;
        for (auto& i_dof : rDofSet) {
            node_id = i_dof.Id();
            pnode = rModelPart.pGetNode(node_id);
            if (i_dof.EquationId() < rA.size1()) {
                if (i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_X || 
                    i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Y || 
                    i_dof.GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Z) {
                    if (pnode->Is(ACTIVE)) {
                        mLMActiveIndices[lm_active_counter] = global_pos;
                        mGlobalToLocalIndexing[global_pos] = lm_active_counter;
                        mWhichBlockType[global_pos] = BlockType::LM_ACTIVE;
                        lm_active_counter++;
                    } else {
                        mLMInactiveIndices[lm_inactive_counter] = global_pos;
                        mGlobalToLocalIndexing[global_pos] = lm_inactive_counter;
                        mWhichBlockType[global_pos] = BlockType::LM_INACTIVE;
                        lm_inactive_counter++;
                    }
                } else if (i_dof.GetVariable().Key() == DISPLACEMENT_X || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Y || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Z) {
                    if (pnode->Is(INTERFACE)) {
                        if (pnode->Is(MASTER)) {
                            mMasterIndices[master_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = master_counter;
                            mWhichBlockType[global_pos] = BlockType::MASTER;
                            master_counter++;
                            
                        } else if (pnode->Is(SLAVE)) {
                            if (pnode->Is(ACTIVE)) {
                                mSlaveActiveIndices[slave_active_counter] = global_pos;
                                mGlobalToLocalIndexing[global_pos] = slave_active_counter;
                                mWhichBlockType[global_pos] = BlockType::SLAVE_ACTIVE;
                                slave_active_counter++;
                            } else {
                                mSlaveInactiveIndices[slave_inactive_counter] = global_pos;
                                mGlobalToLocalIndexing[global_pos] = slave_inactive_counter;
                                mWhichBlockType[global_pos] = BlockType::SLAVE_INACTIVE;
                                slave_inactive_counter++;
                            }
                        } else { // We need to consider always an else to ensure that the system size is consistent
                            mOtherIndices[other_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = other_counter;
                            other_counter++;
                        }
                    } else { // We need to consider always an else to ensure that the system size is consistent
                        mOtherIndices[other_counter] = global_pos;
                        mGlobalToLocalIndexing[global_pos] = other_counter;
                        other_counter++;
                    }
                } else {
                    mOtherIndices[other_counter] = global_pos;
                    mGlobalToLocalIndexing[global_pos] = other_counter;
                    other_counter++;
                }
                global_pos++;
            }
        }
        
        // Doing some check
        KRATOS_ERROR_IF_NOT(lm_active_counter == slave_active_counter) << "The number of LM active DoFs and displacement active DoFs does not coincide" << std::endl;
        KRATOS_ERROR_IF_NOT(lm_inactive_counter == slave_inactive_counter) << "The number of LM active DoFs and displacement active DoFs does not coincide" << std::endl;
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
    std::string Info() const override
    {
        return "Mixed displacement LM linear solver";
    }
    
    /// Print information about this object.
    void PrintInfo (std::ostream& rOStream) const override
    {
        rOStream << "Mixed displacement LM linear solver";
    }
    
    /// Print object's data.
    void PrintData (std::ostream& rOStream) const override
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
    
    /** 
     * @brief T his function generates the subblocks of matrix A
     * @details as A = ( KNN  KNM    KNSI    KNSA     0        0    ) u
     *                 ( KMN  KMM    KMSI    KMSA    -MI^T    -MA^T ) u_master
     *                 ( KSIN KSIM   KSISI   KSISA   DII^T    DIA^T ) u_slave_inactive
     *                 ( KSAN KSAM   KSASI   KSASA   DAI^T    DAA^T ) u_slave_active
     *                 (  0    0      0      0       ALMI        0  ) LMInactive
     *                 (  0   KLMAM  KLMASI  KLMASA   0     KLMALMA ) LMActive
     * We will call as A = ( KNN  KNM    KNSI    KNSA     0      0      ) u
     *                     ( KMN  KMM    KMSI    KMSA   KMLMI   KMLMA   ) u_master
     *                     ( KSIN KSIM   KSISI   KSISA  KSILMI  KSILMA  ) u_slave_inactive
     *                     ( KSAN KSAM   KSASI   KSASA  KSALMI  KSALMA  ) u_slave_active
     *                     (  0    0      0      0      KLMILMI   0     ) LMInactive
     *                     (  0   KLMAM  KLMASI  KLMASA   0     KLMALMA ) LMActive
     * Subblocks are allocated or nor depending on the value of "NeedAllocation"
     * @param rA System matrix    
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
     * @todo Use push_back all the time before adding values!!!
     */
    void FillBlockMatrices (
        bool NeedAllocation, 
        SparseMatrixType& rA,
        VectorType& rX, 
        VectorType& rB
        )
    {
        KRATOS_TRY
        
        // Get access to A data
        const std::size_t* index1 = rA.index1_data().begin();
        const std::size_t* index2 = rA.index2_data().begin();
        const double* values = rA.value_data().begin();
      
        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType lm_inactive_size = mLMInactiveIndices.size();
        
//         SparseMatrixType KMLMI(master_size, lm_inactive_size);          /// The master-inactive LM block (this is the big block of M) // NOTE: Not necessary, you can fix the value directly         
//         SparseMatrixType KSILMI(slave_inactive_size, lm_inactive_size); /// The inactive slave-inactive LM block (this is the big block of D, diagonal) // NOTE: Not necessary, you can fix the value directly
//         SparseMatrixType KSILMA(slave_inactive_size, lm_active_size);   /// The inactive slave-active LM block (this is the big block of D) // NOTE: For dual LM is zero!!!
//         SparseMatrixType KSALMI(slave_active_size, lm_inactive_size);   /// The active slave-inactive LM block (this is the big block of D)  // NOTE: For dual LM is zero!!!
        
        SparseMatrixType KMLMA(lm_active_size, lm_active_size);         /// The master-active LM block (this is the big block of M) 
        SparseMatrixType KLMALMA(lm_active_size, lm_active_size);       /// The active LM-active LM block
        SparseMatrixType KSALMA(slave_active_size, lm_active_size);     /// The active slave-active LM block (this is the big block of D, diagonal) 
        SparseMatrixType KLMILMI(lm_inactive_size, lm_inactive_size);   /// The inactive LM- inactive LM block (diagonal)
        
        if (NeedAllocation)
            AllocateBlocks();

        const SizeType other_dof_initial_index = 0;
        const SizeType master_dof_initial_index = other_dof_size;
        const SizeType slave_inactive_dof_initial_index = master_dof_initial_index + master_size;
        const SizeType slave_active_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;
        
        // Allocate the blocks by push_back
        for (unsigned int i=0; i<rA.size1(); i++) {
            unsigned int row_begin = index1[i];
            unsigned int row_end   = index1[i+1];
            unsigned int local_row_id = mGlobalToLocalIndexing[i];

            if ( mWhichBlockType[i] == BlockType::OTHER) { //either KNN or KNM or KNSI or KNSA
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)                // KNN block
                        mKDispModified.push_back ( other_dof_initial_index + local_row_id, other_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)          // KNM block
                        mKDispModified.push_back ( other_dof_initial_index + local_row_id, master_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE)  // KNSI block
                        mKDispModified.push_back ( other_dof_initial_index + local_row_id, slave_inactive_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)    // KNSA block
                        mKDispModified.push_back ( other_dof_initial_index + local_row_id, slave_active_dof_initial_index + local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::MASTER) { //either KMN or KMM or KMSI or KMLM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)                // KMN block
                        mKDispModified.push_back ( master_dof_initial_index + local_row_id, other_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)          // KNMM block
                        mKDispModified.push_back ( master_dof_initial_index + local_row_id, master_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE)  // KMSI block
                        mKDispModified.push_back ( master_dof_initial_index + local_row_id, slave_inactive_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)    // KMSA block
                        mKDispModified.push_back ( master_dof_initial_index + local_row_id, slave_active_dof_initial_index + local_col_id, value);
//                     else if ( mWhichBlockType[col_index] == BlockType::LM_INACTIVE)    // KMLMI block
//                         KMLMI.push_back ( local_row_id, local_col_id, value);
                    else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE)      // KMLMA block
                        KMLMA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { //either KSIN or KSIM or KSISI or KSISA or KSILM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)               // KSIN block
                        mKDispModified.push_back ( slave_inactive_dof_initial_index + local_row_id, other_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)         // KSIMM block
                        mKDispModified.push_back ( slave_inactive_dof_initial_index + local_row_id, master_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) // KSISI block
                        mKDispModified.push_back ( slave_inactive_dof_initial_index + local_row_id, slave_inactive_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)   // KSISA block
                        mKDispModified.push_back ( slave_inactive_dof_initial_index + local_row_id, slave_active_dof_initial_index + local_col_id, value);
//                     else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE)     // KSILMA block
//                         KSILMA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { //either KSAN or KSAM or KSASA or KSASA or KSALM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)               // KSAN block
                        mKSAN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)         // KSAM block
                        mKSAM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) // KSASI block
                        mKSASI.push_back ( local_row_id, local_col_id, value); 
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)   // KSASA block
                        mKSASA.push_back ( local_row_id, local_col_id, value);
                    else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE)     // KSALMA block (diagonal)
                        KSALMA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { // KLMILMI
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) // KLMILMI block (diagonal)
                        KLMILMI.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { //either KLMAM or KLMASI or KLMASA
                for (unsigned int j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    const double value = values[j];
                    const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::MASTER)              // KLMM block
                        mKDispModified.push_back ( slave_active_dof_initial_index + local_row_id, master_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) // KLMSI block
                        mKDispModified.push_back ( slave_active_dof_initial_index + local_row_id, slave_inactive_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)   // KLMSA block
                        mKDispModified.push_back ( slave_active_dof_initial_index + local_row_id, slave_active_dof_initial_index + local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE)      // KLMALMA block
                        KLMALMA.push_back ( local_row_id, local_col_id, value);
                }
            }
        }
        
        // We compute directly the inverse of the KSALMA matrix 
        // KSALMA it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        for (IndexType i = 0; i < mKLMAModified.size1(); ++i) {
            const double value = KSALMA(i, i);
            if (value > 0.0)
                mKLMAModified.push_back(i, i, 1.0/value);
            else // Auxiliar value
                mKLMAModified.push_back(i, i, 1.0);
        }
        
        // We compute directly the inverse of the KLMILMI matrix 
        // KLMILMI it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        for (IndexType i = 0; i < mKLMAModified.size1(); ++i) {
            const double value = KLMILMI(i, i);
            if (value > 0.0)
                mKLMIModified.push_back(i, i, 1.0/value);
            else // Auxiliar value
                mKLMIModified.push_back(i, i, 1.0);
        }
        
        // Compute the P and C operators
        MatrixMatrixProd(KMLMA,   mKLMAModified, mPOperator);
        MatrixMatrixProd(KLMALMA, mKLMAModified, mCOperator);
        
        // Auxiliar indexes
        const SizeType master_initial_index = other_dof_size + master_size;
        const SizeType aslave_initial_index = master_initial_index + slave_inactive_size;
        
        // We proceed with the auxiliar products for the master blocks
        SparseMatrixType master_auxKSAN(master_size, other_dof_size); 
        MatrixMatrixProd(mPOperator, mKSAN, master_auxKSAN);
        AddingSparseMatrixToCondensed(master_auxKSAN, other_dof_size, 0);
        SparseMatrixType master_auxKSAM(master_size, master_size); 
        MatrixMatrixProd(mPOperator, mKSAM, master_auxKSAM);
        AddingSparseMatrixToCondensed(master_auxKSAM, other_dof_size, other_dof_size);
        SparseMatrixType master_auxKSASI(master_size, slave_inactive_size); 
        MatrixMatrixProd(mPOperator, mKSASI, master_auxKSASI);
        AddingSparseMatrixToCondensed(master_auxKSASI, other_dof_size, master_initial_index);
        SparseMatrixType master_auxKSASA(master_size, slave_active_size); 
        MatrixMatrixProd(mPOperator, mKSASA, master_auxKSASA);
        AddingSparseMatrixToCondensed(master_auxKSASI, other_dof_size, aslave_initial_index);
        
        // We proceed with the auxiliar products for the active slave blocks
        SparseMatrixType aslave_auxKSAN(slave_active_size, other_dof_size); 
        MatrixMatrixProd(mCOperator, mKSAN, aslave_auxKSAN);
        AddingSparseMatrixToCondensed(aslave_auxKSAN, aslave_initial_index, 0);
        SparseMatrixType aslave_auxKSAM(slave_active_size, master_size); 
        MatrixMatrixProd(mCOperator, mKSAM, aslave_auxKSAM);
        AddingSparseMatrixToCondensed(aslave_auxKSAM, aslave_initial_index, other_dof_size);
        SparseMatrixType aslave_auxKSASI(slave_active_size, slave_inactive_size); 
        MatrixMatrixProd(mCOperator, mKSASI, aslave_auxKSASI);
        AddingSparseMatrixToCondensed(aslave_auxKSASI, aslave_initial_index, master_initial_index);
        SparseMatrixType aslave_auxKSASA(slave_active_size, slave_active_size); 
        MatrixMatrixProd(mCOperator, mKSASA, aslave_auxKSASA);
        AddingSparseMatrixToCondensed(aslave_auxKSASI, aslave_initial_index, aslave_initial_index);
        
        KRATOS_CATCH ("")
    }
    
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
        
    LinearSolverPointerType mpSolverDispBlock; /// The pointer to the displacement linear solver 
     
    bool mBlocksAreAllocated; /// The flag that indicates if the blocks are allocated
    bool mIsInitialized;      /// The flag that indicates if the solution is mIsInitialized
    
    IndexVectorType mMasterIndices;         /// The vector storing the indices of the master nodes in contact
    IndexVectorType mSlaveInactiveIndices;  /// The vector storing the indices of the slave nodes in contact (Inactive)
    IndexVectorType mSlaveActiveIndices;    /// The vector storing the indices of the slave nodes in contact (Active)
    IndexVectorType mLMInactiveIndices;     /// The vector storing the indices of the LM (Inactive)
    IndexVectorType mLMActiveIndices;       /// The vector storing the indices of the LM (Active)
    IndexVectorType mOtherIndices;          /// The vector containing the indices for other DoF
    IndexVectorType mGlobalToLocalIndexing; /// This vector stores the correspondance between the local and global
    std::vector<BlockType> mWhichBlockType; /// This vector stores the LM block belongings
    
    SparseMatrixType mKDispModified; /// The modified displacement block
    SparseMatrixType mKLMAModified;  /// The modified active LM block (inverted diagonal)
    SparseMatrixType mKLMIModified;  /// The modified inactive LM block (inverted diagonal)
    
    SparseMatrixType mKSAN;    /// The slave active-displacement block
    SparseMatrixType mKSAM;    /// The active slave-master block
    SparseMatrixType mKSASI;   /// The active slave-inactive slave block
    SparseMatrixType mKSASA;   /// The inactive slave-active slave block
    
    SparseMatrixType mPOperator; /// The operator used for the master blocks 
    SparseMatrixType mCOperator; /// The operator used for the active slave block
    
    VectorType mResidualLMActive;   /// The residual of the active lagrange multipliers
    VectorType mResidualLMInactive; /// The residual of the inactive lagrange multipliers
    VectorType mResidualDisp;       /// The residual of the rest of displacements
    
    VectorType mLMActive;           /// The solution of the active lagrange multiplies
    VectorType mLMInactive;         /// The solution of the inactive lagrange multiplies
    VectorType mDisp;               /// The solution of the rest of displacements
        
    ///@}
    ///@name Private Operators
    ///@{
    
    ///@}
    ///@name Private Operations
    ///@{
    
    /**
     * @brief It allocates all the blocks and operators
     */
    inline void AllocateBlocks() 
    {
        // We clear the matrixes
        mKDispModified.clear(); /// The modified displacement block
        mKLMAModified.clear();  /// The modified active LM block (diagonal)
        mKLMIModified.clear();  /// The modified inaactive LM block (diagonal)

        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType lm_inactive_size = mLMInactiveIndices.size();
        const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;
        
        // We do the allocation
        mKDispModified.resize(total_size, total_size);            /// The modified displacement block
        mKLMAModified.resize(lm_active_size, lm_active_size);     /// The modified active LM block (diagonal)
        mKLMIModified.resize(lm_inactive_size, lm_inactive_size); /// The modified inactve LM block (diagonal)
        
        mKSAN.resize(slave_active_size, other_dof_size);       /// The slave active-displacement block        
        mKSAM.resize(slave_active_size, master_size);          /// The active slave-master block
        mKSASI.resize(slave_active_size, slave_inactive_size); /// The active slave-inactive slave block
        mKSASA.resize(slave_active_size, slave_active_size);   /// The active slave-slave active block
        
        mPOperator.resize(master_size, slave_active_size);    /// The operator used for the master blocks
        mCOperator.resize(lm_active_size, slave_active_size); /// The operator used for the active slave block 
        
        mResidualLMActive.resize(lm_active_size );     /// The residual corresponding the active LM
        mResidualLMInactive.resize(lm_inactive_size ); /// The residual corresponding the inactive LM
        mResidualDisp.resize(total_size );             /// The residual of the displacements
        
        mLMActive.resize(lm_active_size);     /// The solution of the active LM
        mLMInactive.resize(lm_inactive_size); /// The solution of the inactive LM
        mDisp.resize(total_size);             /// The solution of the displacement
    }

    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to u-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualU The vector containing the residual relative to the displacements
     */
    inline void GetUPart (
        const VectorType& rTotalResidual, 
        VectorType& ResidualU
        )
    {
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mLMActiveIndices.size();
        if (ResidualU.size() != total_size )
            ResidualU.resize (total_size, false);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mOtherIndices.size()); i++)
            ResidualU[i] = rTotalResidual[mOtherIndices[i]];
        
        // The corresponding residual for the active slave DoF's
        VectorType aux_res_active_slave(mSlaveActiveIndices.size());
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveActiveIndices.size()); i++)
            aux_res_active_slave[i] = rTotalResidual[mSlaveActiveIndices[i]];
        
        // We compute the complementary residual for the master dofs        
        VectorType aux_complement_master_residual(mMasterIndices.size());
        TSparseSpaceType::Mult(mPOperator, aux_res_active_slave, aux_complement_master_residual);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
            ResidualU[mOtherIndices.size() + i] = rTotalResidual[mMasterIndices[i]] + aux_complement_master_residual[i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveInactiveIndices.size()); i++)
            ResidualU[mOtherIndices.size() + mMasterIndices.size() + i] = rTotalResidual[mSlaveInactiveIndices[i]];
        
        // We compute the complementary residual for the master dofs        
        VectorType aux_complement_active_lm_residual(mLMActiveIndices.size());
        TSparseSpaceType::Mult(mCOperator, aux_res_active_slave, aux_complement_active_lm_residual);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mLMActiveIndices.size()); i++)
            ResidualU[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i] = rTotalResidual[mLMActiveIndices[i]] + aux_complement_active_lm_residual[i];
    }

    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to active lm-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param rResidualLM The vector containing the residual relative to the active LM
     */
    inline void GetLMAPart (
        const VectorType& rTotalResidual, 
        VectorType& rResidualLM
        )
    {
        // We get the displacement residual of the active slave nodes
        if (rResidualLM.size() != mSlaveActiveIndices.size() )
            rResidualLM.resize (mSlaveActiveIndices.size(), false);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(rResidualLM.size()); i++)
            rResidualLM[i] = rTotalResidual[mSlaveActiveIndices[i]];
        
        // From the computed displacements we get the components of the displacements for each block
        VectorType disp_N(mOtherIndices.size());
        VectorType disp_M(mMasterIndices.size());
        VectorType disp_SI(mSlaveInactiveIndices.size());
        VectorType disp_SA(mSlaveActiveIndices.size());
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mOtherIndices.size()); i++)
            disp_N[i] = mDisp[i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
            disp_M[i] = mDisp[mOtherIndices.size() + i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveInactiveIndices.size()); i++)
            disp_SI[i] = mDisp[mOtherIndices.size() + mMasterIndices.size() + i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveActiveIndices.size()); i++)
            disp_SI[i] = mDisp[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i];
        
        // We add the other 
        VectorType aux_mult;
        TSparseSpaceType::Mult(mKSAN, disp_N, aux_mult);
        TSparseSpaceType::UnaliasedAdd (rResidualLM, -1.0, aux_mult);
        TSparseSpaceType::Mult(mKSAM, disp_M, aux_mult);
        TSparseSpaceType::UnaliasedAdd (rResidualLM, -1.0, aux_mult);
        TSparseSpaceType::Mult(mKSASI, disp_SI, aux_mult);
        TSparseSpaceType::UnaliasedAdd (rResidualLM, -1.0, aux_mult);
        TSparseSpaceType::Mult(mKSASA, disp_SA, aux_mult);
        TSparseSpaceType::UnaliasedAdd (rResidualLM, -1.0, aux_mult);
    }
    
    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to inactive lm-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param rResidualLM The vector containing the residual relative to the inactive LM
     */
    inline void GetLMIPart (
        const VectorType& rTotalResidual, 
        VectorType& rResidualLM
        )
    {
        // We get the displacement residual of the active slave nodes
        if (rResidualLM.size() != mSlaveInactiveIndices.size() )
            rResidualLM.resize (mSlaveInactiveIndices.size(), false);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(rResidualLM.size()); i++)
            rResidualLM[i] = rTotalResidual[mSlaveInactiveIndices[i]];
    }

    /**
     * @brief This method writes the displacement part
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualU The vector containing the residual relative to the displacements
     */
    inline void SetUPart (
        VectorType& rTotalResidual, 
        const VectorType& ResidualU
        )
    {
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mOtherIndices.size()); i++)
            rTotalResidual[mOtherIndices[i]] = ResidualU[i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
            rTotalResidual[mMasterIndices[i]] = ResidualU[mOtherIndices.size() + i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveInactiveIndices.size()); i++)
            rTotalResidual[mSlaveInactiveIndices[i]] = ResidualU[mOtherIndices.size() + mMasterIndices.size() + i];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveActiveIndices.size()); i++)
            rTotalResidual[mSlaveActiveIndices[i]] = ResidualU[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i];
    }

    /**
     * @brief This method writes the active Lagrange Multiplier part
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualLM The vector containing the residual relative to the LM
     */
    inline void SetLMAPart (
        VectorType& rTotalResidual, 
        const VectorType& ResidualLM
        )
    {
        #pragma omp parallel for
        for (int i = 0; i< static_cast<int>(ResidualLM.size()); i++)
            rTotalResidual[mLMActiveIndices[i]] = ResidualLM[i];
    }
    
    /**
     * @brief This method writes the inaactive Lagrange Multiplier part
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualLM The vector containing the residual relative to the LM
     */
    inline void SetLMIPart (
        VectorType& rTotalResidual, 
        const VectorType& ResidualLM
        )
    {
        #pragma omp parallel for
        for (int i = 0; i< static_cast<int>(ResidualLM.size()); i++)
            rTotalResidual[mLMInactiveIndices[i]] = ResidualLM[i];
    }
    
    /**
     * @brief Matrix-matrix product C = A·B
     * @detail This method uses a template for each matrix 
     * @param rA The first matrix 
     * @param rB The second matrix 
     * @param rC The resulting matrix 
     */
    template <class AMatrix, class BMatrix, class CMatrix>
    void MatrixMatrixProd(
        const AMatrix& rA, 
        const BMatrix& rB, 
        CMatrix& rC
        ) 
    {        
    #ifdef _OPENMP
        const int nt = omp_get_max_threads();
    #else
        const int nt = 1;
    #endif

        if (nt > 16) {
//             SparseMatrixMultiplicationUtility::MatrixMultiplicationRMerge(rA, rB, rC); // TODO: change to MatrixMultiplicationRMerge when working!!!!
            SparseMatrixMultiplicationUtility::MatrixMultiplicationSaad(rA, rB, rC);
        } else {
            SparseMatrixMultiplicationUtility::MatrixMultiplicationSaad(rA, rB, rC);
        }
    }
    
    /**
     * @brief This method appends to the condensed matrix the given matrix
     * @param rA The matrix to add
     * @param InitialRowIndex The initial index for the row
     * @param InitialColumIndex The initial index for the column
     */
    void AddingSparseMatrixToCondensed(
        const SparseMatrixType& rA,
        const SizeType InitialRowIndex = 0,
        const SizeType InitialColumIndex = 0
        ) 
    {
        // Get access to A data
        const std::size_t* index1 = rA.index1_data().begin();
        const std::size_t* index2 = rA.index2_data().begin();
        const double* values = rA.value_data().begin();
        
        for (unsigned int i=0; i<rA.size1(); i++) {
            unsigned int row_begin = index1[i];
            unsigned int row_end   = index1[i+1];
            
            for (unsigned int j=row_begin; j<row_end; j++) {
                const std::size_t col_index = index2[j];
                const double value = values[j];
                // If the element does not exist we do a push back
                if (!mKDispModified.find_element (i + InitialRowIndex, col_index + InitialColumIndex))
                    mKDispModified.push_back(i + InitialRowIndex, col_index + InitialColumIndex, value);
                else
                    mKDispModified(i + InitialRowIndex, col_index + InitialColumIndex) += value;
            }
        }
    }
    
    /**
     * @brief This method returns the defaulr parameters in order to avoid code duplication
     * @return Returns the default parameters
     */
    
    Parameters GetDefaultParameters()
    {
        Parameters default_parameters( R"(
        {
            "solver_type": "MixedULMLinearSolver",
            "displacement_solver" : {
                    "solver_type":"BICGSTABSolver"
            },
            "LM_solver" : {
                    "solver_type":"CGSolver"
            },
            "tolerance" : 1.0e-6,
            "max_iteration_number" : 200
        }  )" );
        
        return default_parameters;
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
}; // Class MixedULMLinearSolver
///@}
///@name Type Definitions
///@{
///@}
///@name Input and output
///@{
/// input stream function
template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
inline std::istream& operator >> (std::istream& IStream,
                                  MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>& rThis)
{
    return IStream;
}
/// output stream function
template<class TSparseSpaceType, class TDenseSpaceType, class TPreconditionerType, class TReordererType>
inline std::ostream& operator << (std::ostream& rOStream,
                                  const MixedULMLinearSolver<TSparseSpaceType, TDenseSpaceType,TPreconditionerType, TReordererType>& rThis)
{
    rThis.PrintInfo (rOStream);
    rOStream << std::endl;
    rThis.PrintData (rOStream);
    return rOStream;
}
///@}
}  // namespace Kratos.
#endif // KRATOS_MIXEDULM_SOLVER_H_INCLUDED  defined
