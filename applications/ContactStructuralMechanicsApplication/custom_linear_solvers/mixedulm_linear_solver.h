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
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType lm_inactive_size = mLMInactiveIndices.size();
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();

        // Get the u and lm residuals
        GetUPart (rB, mResidualDisp);
        
        // Solve u block
        noalias(mDisp)  = ZeroVector(total_size);
        mpSolverDispBlock->Solve (mKDispModified, mDisp, mResidualDisp);

        // Write back solution
        SetUPart(rX, mDisp);
        
        // Solve LM
        if (lm_active_size > 0) {
            // Now we compute the residual of the LM
            GetLMAPart (rB, mResidualLMActive);
            // LM = D⁻1*rLM
            noalias(mLMActive) = ZeroVector(mLMActiveIndices.size());
            TSparseSpaceType::Mult (mKLMAModified, mResidualLMActive, mLMActive);
            // Write back solution
            SetLMAPart(rX, mLMActive);
        }
        
        if (lm_inactive_size > 0) {
            // Now we compute the residual of the LM
            GetLMIPart (rB, mResidualLMInactive);
            // LM = D⁻1*rLM
            noalias(mLMInactive) = ZeroVector(mLMInactiveIndices.size());
            TSparseSpaceType::Mult (mKLMIModified, mResidualLMInactive, mLMInactive);
            // Write back solution
            SetLMIPart(rX, mLMInactive);
        }
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
        mBlocksAreAllocated = false;
        mpSolverDispBlock->Clear();
        
        // We clear the matrixes and vectors
        mKDispModified.clear(); /// The modified displacement block
        mKLMAModified.clear();  /// The modified active LM block (diagonal)
        mKLMIModified.clear();  /// The modified inaactive LM block (diagonal)

        mKSAN.clear();  /// The slave active-displacement block        
        mKSAM.clear();  /// The active slave-master block
        mKSASI.clear(); /// The active slave-inactive slave block
        mKSASA.clear(); /// The active slave-slave active block
        
        mPOperator.clear(); /// The operator used for the master blocks
        mCOperator.clear(); /// The operator used for the active slave block 
        
        mResidualLMActive.clear();   /// The residual corresponding the active LM
        mResidualLMInactive.clear(); /// The residual corresponding the inactive LM
        mResidualDisp.clear();       /// The residual of the displacements
        
        mLMActive.clear();   /// The solution of the active LM
        mLMInactive.clear(); /// The solution of the inactive LM
        mDisp.clear();       /// The solution of the displacement
        
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
                        ++lm_active_counter;
                    } else {
                        mLMInactiveIndices[lm_inactive_counter] = global_pos;
                        mGlobalToLocalIndexing[global_pos] = lm_inactive_counter;
                        mWhichBlockType[global_pos] = BlockType::LM_INACTIVE;
                        ++lm_inactive_counter;
                    }
                } else if (i_dof.GetVariable().Key() == DISPLACEMENT_X || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Y || 
                    i_dof.GetVariable().Key() == DISPLACEMENT_Z) {
                    if (pnode->Is(INTERFACE)) {
                        if (pnode->Is(MASTER)) {
                            mMasterIndices[master_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = master_counter;
                            mWhichBlockType[global_pos] = BlockType::MASTER;
                            ++master_counter;
                            
                        } else if (pnode->Is(SLAVE)) {
                            if (pnode->Is(ACTIVE)) {
                                mSlaveActiveIndices[slave_active_counter] = global_pos;
                                mGlobalToLocalIndexing[global_pos] = slave_active_counter;
                                mWhichBlockType[global_pos] = BlockType::SLAVE_ACTIVE;
                                ++slave_active_counter;
                            } else {
                                mSlaveInactiveIndices[slave_inactive_counter] = global_pos;
                                mGlobalToLocalIndexing[global_pos] = slave_inactive_counter;
                                mWhichBlockType[global_pos] = BlockType::SLAVE_INACTIVE;
                                ++slave_inactive_counter;
                            }
                        } else { // We need to consider always an else to ensure that the system size is consistent
                            mOtherIndices[other_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = other_counter;
                            ++other_counter;
                        }
                    } else { // We need to consider always an else to ensure that the system size is consistent
                        mOtherIndices[other_counter] = global_pos;
                        mGlobalToLocalIndexing[global_pos] = other_counter;
                        ++other_counter;
                    }
                } else {
                    mOtherIndices[other_counter] = global_pos;
                    mGlobalToLocalIndexing[global_pos] = other_counter;
                    ++other_counter;
                }
                ++global_pos;
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
        const bool NeedAllocation, 
        SparseMatrixType& rA,
        VectorType& rX, 
        VectorType& rB
        )
    {
        KRATOS_TRY
      
        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType lm_inactive_size = mLMInactiveIndices.size();
        
        if (NeedAllocation)
            AllocateBlocks();

        // Get access to A data
        const std::size_t* index1 = rA.index1_data().begin();
        const std::size_t* index2 = rA.index2_data().begin();
        const double* values = rA.value_data().begin();
        
        // Allocate the auxiliar blocks by push_back
        SparseMatrixType KMLMA(lm_active_size, lm_active_size);         /// The master-active LM block (this is the big block of M) 
        SparseMatrixType KLMALMA(lm_active_size, lm_active_size);       /// The active LM-active LM block
        SparseMatrixType KSALMA(slave_active_size, lm_active_size);     /// The active slave-active LM block (this is the big block of D, diagonal) 
        SparseMatrixType KLMILMI(lm_inactive_size, lm_inactive_size);   /// The inactive LM- inactive LM block (diagonal)
        
        // We iterate over original matrix
        for (IndexType i=0; i<rA.size1(); i++) {
            const IndexType row_begin = index1[i];
            const IndexType row_end   = index1[i+1];
            const IndexType local_row_id = mGlobalToLocalIndexing[i];

            if ( mWhichBlockType[i] == BlockType::MASTER) { // KMLMA
                for (IndexType j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KMLMA block
                        const double value = values[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        KMLMA.push_back ( local_row_id, local_col_id, value);
                    }
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { //either KSAN or KSAM or KSASA or KSASA or KSALM
                for (IndexType j=row_begin; j<row_end; j++) {
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
                for (IndexType j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) { // KLMILMI block (diagonal)
                        const double value = values[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        KLMILMI.push_back ( local_row_id, local_col_id, value);
                    }
                }
            } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { // KLMALMA
                for (IndexType j=row_begin; j<row_end; j++) {
                    const IndexType col_index = index2[j];
                    if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KLMALMA block
                        const double value = values[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        KLMALMA.push_back ( local_row_id, local_col_id, value);
                    }
                }
            }
        }
               
        // We compute directly the inverse of the KSALMA matrix 
        // KSALMA it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        // TODO: this can be optimized in OMP
        for (IndexType i = 0; i < mKLMAModified.size1(); ++i) {
            const double value = KSALMA(i, i);
            if (std::abs(value) > 0.0)
                mKLMAModified.push_back(i, i, 1.0/value);
            else // Auxiliar value
                mKLMAModified.push_back(i, i, 1.0);
        }
        
        // We compute directly the inverse of the KLMILMI matrix 
        // KLMILMI it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        // TODO: this can be optimized in OMP
        for (IndexType i = 0; i < mKLMIModified.size1(); ++i) {
            const double value = KLMILMI(i, i);
            if (std::abs(value) > 0.0)
                mKLMIModified.push_back(i, i, 1.0/value);
            else // Auxiliar value
                mKLMIModified.push_back(i, i, 1.0);
        }
        
        // Compute the P and C operators
        if (slave_active_size > 0) {
            MatrixMatrixProd(KMLMA,   mKLMAModified, mPOperator);
            MatrixMatrixProd(KLMALMA, mKLMAModified, mCOperator);
        }
        
        // We proceed with the auxiliar products for the master blocks
        SparseMatrixType master_auxKSAN(master_size, other_dof_size); 
        SparseMatrixType master_auxKSAM(master_size, master_size); 
        SparseMatrixType master_auxKSASI(master_size, slave_inactive_size); 
        SparseMatrixType master_auxKSASA(master_size, slave_active_size); 
        
        if (slave_active_size > 0) {
            MatrixMatrixProd(mPOperator, mKSAN, master_auxKSAN);
            MatrixMatrixProd(mPOperator, mKSAM, master_auxKSAM);
            if (slave_inactive_size > 0)
                MatrixMatrixProd(mPOperator, mKSASI, master_auxKSASI);
            MatrixMatrixProd(mPOperator, mKSASA, master_auxKSASA);
        }
        
        // We proceed with the auxiliar products for the active slave blocks
        SparseMatrixType aslave_auxKSAN(slave_active_size, other_dof_size); 
        SparseMatrixType aslave_auxKSAM(slave_active_size, master_size); 
        SparseMatrixType aslave_auxKSASI(slave_active_size, slave_inactive_size);
        SparseMatrixType aslave_auxKSASA(slave_active_size, slave_active_size); 
        
        if (slave_active_size > 0) {
            MatrixMatrixProd(mCOperator, mKSAN, aslave_auxKSAN);
            MatrixMatrixProd(mCOperator, mKSAM, aslave_auxKSAM);
            if (slave_inactive_size > 0)
                MatrixMatrixProd(mCOperator, mKSASI, aslave_auxKSASI);
            MatrixMatrixProd(mCOperator, mKSASA, aslave_auxKSASA);
        }
        
        // Auxiliar indexes        
        const SizeType other_dof_initial_index = 0;
        const SizeType master_dof_initial_index = other_dof_size;
        const SizeType slave_inactive_dof_initial_index = master_dof_initial_index + master_size;
        const SizeType slave_active_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;
        const SizeType assembling_slave_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;
        
        // The auxiliar index structure
        const SizeType nrows = mKDispModified.size1();
        const SizeType ncols = mKDispModified.size2();
        std::ptrdiff_t* K_disp_modified_ptr = new std::ptrdiff_t[nrows + 1];
        K_disp_modified_ptr[0] = 0;
        
        // Creating a buffer for parallel vector fill
        const int num_threads = OpenMPUtils::GetNumThreads();
        std::vector<std::vector<std::ptrdiff_t>> buffer_markers(num_threads, std::vector<std::ptrdiff_t>(ncols, -1));
        std::vector<std::ptrdiff_t> marker(ncols, -1);
        
        #pragma omp parallel
        {
            const int id = OpenMPUtils::ThisThread();
            
            #pragma omp for
            for (int i=0; i<static_cast<int>(rA.size1()); i++) {
                const IndexType row_begin = index1[i];
                const IndexType row_end   = index1[i + 1];
                
                std::ptrdiff_t K_disp_modified_cols = 0;
                
                if ( mWhichBlockType[i] == BlockType::OTHER) { //either KNN or KNM or KNSI or KNSA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + other_dof_initial_index;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KNN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KNSI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KNSA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        }
                    }
                    K_disp_modified_ptr[local_row_id + 1] = K_disp_modified_cols;
                } else if ( mWhichBlockType[i] == BlockType::MASTER) { //either KMN or KMM or KMSI or KMLM
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + master_dof_initial_index;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KMN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNMM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KMSI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KMSA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        }
                    }
                    K_disp_modified_ptr[local_row_id + 1] = K_disp_modified_cols;
                } else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { //either KSIN or KSIM or KSISI or KSISA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + slave_inactive_dof_initial_index;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                // KSIN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {        // KSIM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSISI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KSISA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        }
                    }
                    K_disp_modified_ptr[local_row_id + 1] = K_disp_modified_cols;
                } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { //either KLMAM or KLMASI or KLMASA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + assembling_slave_dof_initial_index;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::MASTER) {                // KLMAM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KLMASI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KLMASA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = local_row_id;
                            ++K_disp_modified_cols;
                        }
                    }
                    K_disp_modified_ptr[local_row_id + 1] = K_disp_modified_cols;
                }
            }
            
            // Combine buffers together
            #pragma omp single
            {
                for( auto& buffer_marker : buffer_markers) {
                    for (std::size_t i_marker = 0; i_marker < ncols; ++i_marker) {
                        if (buffer_marker[i_marker] > -1) {
                            marker[i_marker] = buffer_marker[i_marker];
                        }
                    }
                }
            }
        }
        
        #pragma omp parallel
        {
            if (slave_active_size > 0) {
                // Get access to master_auxKSAN data
                ComputeNonZeroBlocks(master_auxKSAN, K_disp_modified_ptr, marker,  master_dof_initial_index, other_dof_initial_index);
                
                // Get access to master_auxKSAM data
                ComputeNonZeroBlocks(master_auxKSAM, K_disp_modified_ptr, marker,  master_dof_initial_index, master_dof_initial_index);
                
                // Get access to master_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeNonZeroBlocks(master_auxKSASI, K_disp_modified_ptr, marker,  master_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to master_auxKSASA data
                ComputeNonZeroBlocks(master_auxKSASA, K_disp_modified_ptr, marker,  master_dof_initial_index, assembling_slave_dof_initial_index);
                
                // Get access to aslave_auxKSAN data
                ComputeNonZeroBlocks(aslave_auxKSAN, K_disp_modified_ptr, marker,  assembling_slave_dof_initial_index, other_dof_initial_index);
                
                // Get access to aslave_auxKSAM data
                ComputeNonZeroBlocks(aslave_auxKSAM, K_disp_modified_ptr, marker,  assembling_slave_dof_initial_index, master_dof_initial_index);
                
                // Get access to aslave_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeNonZeroBlocks(aslave_auxKSASI, K_disp_modified_ptr, marker,  assembling_slave_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to aslave_auxKSASA data
                ComputeNonZeroBlocks(aslave_auxKSASA, K_disp_modified_ptr, marker,  assembling_slave_dof_initial_index, assembling_slave_dof_initial_index);
            }
        }
        
        // We initialize the final sparse matrix
        std::partial_sum(K_disp_modified_ptr, K_disp_modified_ptr + nrows + 1, K_disp_modified_ptr);
        const std::size_t nonzero_values = K_disp_modified_ptr[nrows];
        std::ptrdiff_t* aux_index2_K_disp_modified = new std::ptrdiff_t[nonzero_values];
        double* aux_val_K_disp_modified = new double[nonzero_values];
        
        std::fill(buffer_markers.begin(), buffer_markers.end(), std::vector<std::ptrdiff_t>(ncols, -1));
        std::fill(marker.begin(), marker.end(), -1);
        
        #pragma omp parallel
        {
            const int id = OpenMPUtils::ThisThread();
            
            #pragma omp for
            for (int i=0; i<static_cast<int>(rA.size1()); i++) {
                const IndexType row_begin_A = index1[i];
                const IndexType row_end_A   = index1[i + 1];
                
                if ( mWhichBlockType[i] == BlockType::OTHER) { //either KNN or KNM or KNSI or KNSA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + other_dof_initial_index;
                    std::ptrdiff_t row_beg = K_disp_modified_ptr[local_row_id];
                    std::ptrdiff_t row_end = row_beg;
                    for (IndexType j=row_begin_A; j<row_end_A; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        const double value = values[j];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KNN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KNSI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KNSA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        }
                    }
                } else if ( mWhichBlockType[i] == BlockType::MASTER) { //either KMN or KMM or KMSI or KMLM
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + master_dof_initial_index;
                    std::ptrdiff_t row_beg = K_disp_modified_ptr[local_row_id];
                    std::ptrdiff_t row_end = row_beg;
                    for (IndexType j=row_begin_A; j<row_end_A; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        const double value = values[j];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KMN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNMM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KMSI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KMSA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        }
                    }
                } else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { //either KSIN or KSIM or KSISI or KSISA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + slave_inactive_dof_initial_index;
                    std::ptrdiff_t row_beg = K_disp_modified_ptr[local_row_id];
                    std::ptrdiff_t row_end = row_beg;
                    for (IndexType j=row_begin_A; j<row_end_A; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        const double value = values[j];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                // KSIN block
                            buffer_markers[id][local_col_id + other_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {        // KSIM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSISI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {  // KSISA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        }
                    }
                } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { //either KLMAM or KLMASI or KLMASA
                    const IndexType local_row_id = mGlobalToLocalIndexing[i] + assembling_slave_dof_initial_index;
                    std::ptrdiff_t row_beg = K_disp_modified_ptr[local_row_id];
                    std::ptrdiff_t row_end = row_beg;
                    for (IndexType j=row_begin_A; j<row_end_A; j++) {
                        const IndexType col_index = index2[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        const double value = values[j];
                        if (mWhichBlockType[col_index] == BlockType::MASTER) {                // KLMAM block
                            buffer_markers[id][local_col_id + master_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KLMASI block
                            buffer_markers[id][local_col_id + slave_inactive_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KLMASA block
                            buffer_markers[id][local_col_id + slave_active_dof_initial_index] = row_end;
                            aux_index2_K_disp_modified[row_end] = col_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        }
                    }
                }
            }
            
            // Combine buffers together
            #pragma omp single
            {
                for( auto& buffer_marker : buffer_markers) {
                    for (std::size_t i_marker = 0; i_marker < ncols; ++i_marker) {
                        if (buffer_marker[i_marker] > -1) {
                            marker[i_marker] = buffer_marker[i_marker];
                        }
                    }
                }
            }
            
        }
        
        #pragma omp parallel
        {
            if (slave_active_size > 0) {
                // Get access to master_auxKSAN data
                ComputeAuxiliarValuesBlocks(master_auxKSAN, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, master_dof_initial_index, other_dof_initial_index);
                
                // Get access to master_auxKSAM data
                ComputeAuxiliarValuesBlocks(master_auxKSAM, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, master_dof_initial_index, master_dof_initial_index);
                
                // Get access to master_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeAuxiliarValuesBlocks(master_auxKSASI, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, master_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to master_auxKSASA data
                ComputeAuxiliarValuesBlocks(master_auxKSASA, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, master_dof_initial_index, assembling_slave_dof_initial_index);
                
                // Get access to aslave_auxKSAN data
                ComputeAuxiliarValuesBlocks(aslave_auxKSAN, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, assembling_slave_dof_initial_index, other_dof_initial_index);
                
                // Get access to aslave_auxKSAM data
                ComputeAuxiliarValuesBlocks(aslave_auxKSAM, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, assembling_slave_dof_initial_index, master_dof_initial_index);
                
                // Get access to aslave_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeAuxiliarValuesBlocks(aslave_auxKSASI, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, assembling_slave_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to aslave_auxKSASA data
                ComputeAuxiliarValuesBlocks(aslave_auxKSASA, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, assembling_slave_dof_initial_index, assembling_slave_dof_initial_index);
            }
        }
        
        #pragma omp parallel
        {
            // We reorder the rows
            #pragma omp for
            for (int i=0; i<static_cast<int>(nrows); i++) {
                const std::ptrdiff_t row_beg = K_disp_modified_ptr[i];
                const std::ptrdiff_t row_end = K_disp_modified_ptr[i + 1];
                SparseMatrixMultiplicationUtility::SortRow(aux_index2_K_disp_modified + row_beg, aux_val_K_disp_modified + row_beg, row_end - row_beg);
            }
        }
        
        // Finally we build the final matrix
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKDispModified, nrows, ncols, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified);
        
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
     * @brief This is a method to check the block containing nonzero values
     * @param AuxK The auxiliar block
     * @param KPtr The nonzero rows array
     * @param Marker A marker to check the already asigned values
     * @param InitialIndexRow The initial row index of the auxiliar block in the final matrix 
     * @param InitialIndexColumn The initial column index of the auxiliar block in the final matrix 
     * @todo Check the col_index!!!!!!
     */
    inline void ComputeNonZeroBlocks(
        const SparseMatrixType& AuxK,
        std::ptrdiff_t* KPtr,
        std::vector<std::ptrdiff_t>& Marker,
        const SizeType InitialIndexRow,
        const SizeType InitialIndexColumn
        )
    {
        // Get access to aux_K data
        const IndexType* aux_K_index1 = AuxK.index1_data().begin();
        const IndexType* aux_K_index2 = AuxK.index2_data().begin();
        
        #pragma omp for
        for (int i=0; i<static_cast<int>(AuxK.size1()); i++) {
            const IndexType row_begin = aux_K_index1[i];
            const IndexType row_end   = aux_K_index1[i + 1];
            
            std::ptrdiff_t& K_disp_modified_cols = KPtr[InitialIndexRow + i + 1];
            
            for (IndexType j=row_begin; j<row_end; j++) {
                const IndexType col_index = InitialIndexColumn + aux_K_index2[j];
                if (Marker[col_index] != static_cast<std::ptrdiff_t>(InitialIndexRow + i)) {
                    Marker[col_index] = InitialIndexRow +  i;
                    ++K_disp_modified_cols;
                }
            }
        }
    }
    
    /**
     * @brief This is a method to compute the contribution of the auxiliar blocks
     * @param AuxK The auxiliar block
     * @param KPtr The nonzero rows array
     * @param AuxIndex2 The indexes of the non zero columns
     * @param AuxVals The values of the final matrix
     * @param Marker A marker to check the already asigned values
     * @param InitialIndexRow The initial row index of the auxiliar block in the final matrix 
     * @param InitialIndexColumn The initial column index of the auxiliar block in the final matrix 
     * @todo Check the col_index!!!!!!
     */
    inline void ComputeAuxiliarValuesBlocks(
        const SparseMatrixType& AuxK,
        const std::ptrdiff_t* KPtr,
        std::ptrdiff_t* AuxIndex2,
        double* AuxVals,
        std::vector<std::ptrdiff_t>& Marker,
        const SizeType InitialIndexRow,
        const SizeType InitialIndexColumn
        )
    {
        // Get access to aux_K data
        const double* aux_values = AuxK.value_data().begin();
        const IndexType* aux_K_index1 = AuxK.index1_data().begin();
        const IndexType* aux_K_index2 = AuxK.index2_data().begin();
        
        #pragma omp for
        for (int i=0; i<static_cast<int>(AuxK.size1()); i++) {
            const IndexType aux_K_row_begin = aux_K_index1[i];
            const IndexType aux_K_row_end   = aux_K_index1[i + 1];
            
            std::ptrdiff_t row_beg = KPtr[i + InitialIndexRow];
            std::ptrdiff_t row_end = row_beg;
            
            for (IndexType j=aux_K_row_begin; j<aux_K_row_end; j++) {
                const IndexType col_index = InitialIndexColumn + aux_K_index2[j];
                if (Marker[col_index] < row_beg) {
                    Marker[col_index] = row_end;
                    AuxIndex2[row_end] = col_index;
                    AuxVals[row_end] = aux_values[j];
                    ++row_end;
                } else {
                    AuxVals[Marker[col_index]] += aux_values[j];
                }
            }
        }
    }
    
    /**
     * @brief It allocates all the blocks and operators
     */
    inline void AllocateBlocks() 
    {
        // We clear the matrixes
        mKDispModified.clear(); /// The modified displacement block
        mKLMAModified.clear();  /// The modified active LM block (diagonal)
        mKLMIModified.clear();  /// The modified inaactive LM block (diagonal)

        mKSAN.clear();  /// The slave active-displacement block        
        mKSAM.clear();  /// The active slave-master block
        mKSASI.clear(); /// The active slave-inactive slave block
        mKSASA.clear(); /// The active slave-slave active block
        
        mPOperator.clear(); /// The operator used for the master blocks
        mCOperator.clear(); /// The operator used for the active slave block 
        
        mResidualLMActive.clear();   /// The residual corresponding the active LM
        mResidualLMInactive.clear(); /// The residual corresponding the inactive LM
        mResidualDisp.clear();       /// The residual of the displacements
        
        mLMActive.clear();   /// The solution of the active LM
        mLMInactive.clear(); /// The solution of the inactive LM
        mDisp.clear();       /// The solution of the displacement
        
        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType lm_inactive_size = mLMInactiveIndices.size();
        const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;
        
        // We do the allocation
        mKDispModified.resize(total_size, total_size, false);            /// The modified displacement block
        mKLMAModified.resize(lm_active_size, lm_active_size, false);     /// The modified active LM block (diagonal)
        mKLMIModified.resize(lm_inactive_size, lm_inactive_size, false); /// The modified inactve LM block (diagonal)
        
        mKSAN.resize(slave_active_size, other_dof_size, false);       /// The slave active-displacement block        
        mKSAM.resize(slave_active_size, master_size, false);          /// The active slave-master block
        mKSASI.resize(slave_active_size, slave_inactive_size, false); /// The active slave-inactive slave block
        mKSASA.resize(slave_active_size, slave_active_size, false);   /// The active slave-slave active block
        
        mPOperator.resize(master_size, slave_active_size, false);    /// The operator used for the master blocks
        mCOperator.resize(lm_active_size, slave_active_size, false); /// The operator used for the active slave block 
        
        mResidualLMActive.resize(lm_active_size, false );     /// The residual corresponding the active LM
        mResidualLMInactive.resize(lm_inactive_size, false ); /// The residual corresponding the inactive LM
        mResidualDisp.resize(total_size );             /// The residual of the displacements
        
        mLMActive.resize(lm_active_size, false);     /// The solution of the active LM
        mLMInactive.resize(lm_inactive_size, false); /// The solution of the inactive LM
        mDisp.resize(total_size, false);             /// The solution of the displacement
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
        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
        const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;
        
        // Resize in case the size is not correct
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
        
        if (slave_active_size > 0) {
            // We compute the complementary residual for the master dofs        
            VectorType aux_complement_master_residual(mMasterIndices.size());
            TSparseSpaceType::Mult(mPOperator, aux_res_active_slave, aux_complement_master_residual);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
                ResidualU[mOtherIndices.size() + i] = rTotalResidual[mMasterIndices[i]] + aux_complement_master_residual[i];
        } else {
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
                ResidualU[mOtherIndices.size() + i] = rTotalResidual[mMasterIndices[i]];
        }
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveInactiveIndices.size()); i++)
            ResidualU[mOtherIndices.size() + mMasterIndices.size() + i] = rTotalResidual[mSlaveInactiveIndices[i]];
        
        if (slave_active_size > 0) {
            // We compute the complementary residual for the master dofs        
            VectorType aux_complement_active_lm_residual(mLMActiveIndices.size());
            TSparseSpaceType::Mult(mCOperator, aux_res_active_slave, aux_complement_active_lm_residual);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(mLMActiveIndices.size()); i++)
                ResidualU[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i] = rTotalResidual[mLMActiveIndices[i]] + aux_complement_active_lm_residual[i];
        } else {
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(mLMActiveIndices.size()); i++)
                ResidualU[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i] = rTotalResidual[mLMActiveIndices[i]];
        }
    }

    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to active lm-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param rResidualLMA The vector containing the residual relative to the active LM
     */
    inline void GetLMAPart(
        const VectorType& rTotalResidual, 
        VectorType& rResidualLMA
        )
    {        
        // Auxiliar sizes
        const SizeType other_dof_size = mOtherIndices.size();
        const SizeType master_size = mMasterIndices.size();
        const SizeType slave_inactive_size = mSlaveInactiveIndices.size();
        const SizeType slave_active_size = mSlaveActiveIndices.size();
            
        // We add the other 
        if (slave_active_size > 0) {
            
            // We get the displacement residual of the active slave nodes
            if (rResidualLMA.size() != slave_active_size )
                rResidualLMA.resize (slave_active_size, false);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(rResidualLMA.size()); i++)
                rResidualLMA[i] = rTotalResidual[mSlaveActiveIndices[i]];
            
            // From the computed displacements we get the components of the displacements for each block
            VectorType disp_N(other_dof_size);
            VectorType disp_M(master_size);
            VectorType disp_SI(slave_inactive_size);
            VectorType disp_SA(slave_active_size);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(other_dof_size); i++)
                disp_N[i] = mDisp[i];
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(master_size); i++)
                disp_M[i] = mDisp[other_dof_size + i];
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(slave_inactive_size); i++)
                disp_SI[i] = mDisp[other_dof_size + master_size + i];
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(slave_active_size); i++)
                disp_SA[i] = mDisp[other_dof_size + master_size + slave_inactive_size + i];
        
            VectorType aux_mult(slave_active_size);
            TSparseSpaceType::Mult(mKSAN, disp_N, aux_mult);
            TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
            TSparseSpaceType::Mult(mKSAM, disp_M, aux_mult);
            TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
            if (slave_inactive_size > 0) {
                TSparseSpaceType::Mult(mKSASI, disp_SI, aux_mult);
                TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
            }
            TSparseSpaceType::Mult(mKSASA, disp_SA, aux_mult);
            TSparseSpaceType::UnaliasedAdd (rResidualLMA, -1.0, aux_mult);
        }
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
            SparseMatrixMultiplicationUtility::MatrixMultiplicationRMerge(rA, rB, rC);
        } else {
            SparseMatrixMultiplicationUtility::MatrixMultiplicationSaad(rA, rB, rC);
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
