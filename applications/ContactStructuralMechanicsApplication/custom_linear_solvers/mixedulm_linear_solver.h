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
#include "custom_utilities/logging_settings.hpp"

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
        
        KRATOS_DETAIL("mKDispModified") << mKDispModified << std::endl;
        KRATOS_DETAIL("mResidualDisp") << mResidualDisp << std::endl;
        
        // Solve u block
        noalias(mDisp)  = ZeroVector(total_size);
        mpSolverDispBlock->Solve (mKDispModified, mDisp, mResidualDisp);

        // Write back solution
        SetUPart(rX, mDisp);
        
        // Solve LM
        if (lm_active_size > 0) {
            // Now we compute the residual of the LM
            GetLMAPart (rB, mResidualLMActive);
            KRATOS_DETAIL("mKLMAModified") << mKLMAModified << std::endl;
            KRATOS_DETAIL("mResidualLMActive") << mResidualLMActive << std::endl;
            // LM = D⁻1*rLM
            noalias(mLMActive) = ZeroVector(mLMActiveIndices.size());
            TSparseSpaceType::Mult (mKLMAModified, mResidualLMActive, mLMActive);
            // Write back solution
            SetLMAPart(rX, mLMActive);
        }
        
        if (lm_inactive_size > 0) {
            // Now we compute the residual of the LM
            GetLMIPart (rB, mResidualLMInactive);
            KRATOS_DETAIL("mKLMIModified") << mKLMIModified << std::endl;
            KRATOS_DETAIL("mResidualLMInactive") << mResidualLMInactive << std::endl;
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
     * @todo Filter zero terms (reduce size new matrix)
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
        const double tolerance = std::numeric_limits<double>::epsilon();
        
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
        
        std::ptrdiff_t* KMLMA_ptr = new std::ptrdiff_t[KMLMA.size1() + 1];
        KMLMA_ptr[0] = 0;
        std::ptrdiff_t* mKSAN_ptr = new std::ptrdiff_t[mKSAN.size1() + 1];
        mKSAN_ptr[0] = 0;
        std::ptrdiff_t* mKSAM_ptr = new std::ptrdiff_t[mKSAM.size1() + 1];
        mKSAM_ptr[0] = 0;
        std::ptrdiff_t* mKSASI_ptr = new std::ptrdiff_t[mKSASI.size1() + 1];
        mKSASI_ptr[0] = 0;
        std::ptrdiff_t* mKSASA_ptr = new std::ptrdiff_t[mKSASA.size1() + 1];
        mKSASA_ptr[0] = 0;
        std::ptrdiff_t* KSALMA_ptr = new std::ptrdiff_t[KSALMA.size1() + 1];
        KSALMA_ptr[0] = 0;
        std::ptrdiff_t* KLMILMI_ptr = new std::ptrdiff_t[KLMILMI.size1() + 1];
        KLMILMI_ptr[0] = 0;
        std::ptrdiff_t* KLMALMA_ptr = new std::ptrdiff_t[KLMALMA.size1() + 1];
        KLMALMA_ptr[0] = 0;
        
        #pragma omp parallel
        {            
            // We iterate over original matrix
            #pragma omp for
            for (IndexType i=0; i<rA.size1(); i++) {
                const IndexType row_begin = index1[i];
                const IndexType row_end   = index1[i+1];
                const IndexType local_row_id = mGlobalToLocalIndexing[i];

                std::ptrdiff_t KMLMA_cols = 0;
                std::ptrdiff_t mKSAN_cols = 0;
                std::ptrdiff_t mKSAM_cols = 0;
                std::ptrdiff_t mKSASI_cols = 0;
                std::ptrdiff_t mKSASA_cols = 0;
                std::ptrdiff_t KSALMA_cols = 0;
                std::ptrdiff_t KLMILMI_cols = 0;
                std::ptrdiff_t KLMALMA_cols = 0;
                
                if ( mWhichBlockType[i] == BlockType::MASTER) { // KMLMA
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KMLMA block
//                             const double value = values[j];
//                             const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
//                             KMLMA.push_back ( local_row_id, local_col_id, value);
                            ++KMLMA_cols;
                        }
                    }
                    KMLMA_ptr[local_row_id + 1] = KMLMA_cols;
                } else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { //either KSAN or KSAM or KSASA or KSASA or KSALM
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
//                         const double value = values[j];
//                         const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KSAN block
//                             mKSAN.push_back ( local_row_id, local_col_id, value);
                            ++mKSAN_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KSAM block
//                             mKSAM.push_back ( local_row_id, local_col_id, value);
                            ++mKSAM_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSASI block
//                             mKSASI.push_back ( local_row_id, local_col_id, value); 
                            ++mKSASI_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KSASA block
//                             mKSASA.push_back ( local_row_id, local_col_id, value);
                            ++mKSASA_cols;
                        } else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) {     // KSALMA block (diagonal)
//                             KSALMA.push_back ( local_row_id, local_col_id, value);
                            ++KSALMA_cols;
                        }
                    }
                    mKSAN_ptr[local_row_id + 1]   = mKSAN_cols;
                    mKSAM_ptr[local_row_id + 1]   = mKSAM_cols;
                    mKSASI_ptr[local_row_id + 1]  = mKSASI_cols;
                    mKSASA_ptr[local_row_id + 1]  = mKSASA_cols;
                    KSALMA_ptr[local_row_id + 1]  = KSALMA_cols;
                } else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { // KLMILMI
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) { // KLMILMI block (diagonal)
//                             const double value = values[j];
//                             const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
//                             KLMILMI.push_back ( local_row_id, local_col_id, value);
                            ++KLMILMI_cols;
                        }
                    }
                    KLMILMI_ptr[local_row_id + 1] = KLMILMI_cols;
                } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { // KLMALMA
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KLMALMA block
//                             const double value = values[j];
//                             const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
//                             KLMALMA.push_back ( local_row_id, local_col_id, value);
                            ++KLMALMA_cols;
                        }
                    }
                    KLMALMA_ptr[local_row_id + 1] = KLMALMA_cols;
                }
            }
        }
        
        // We initialize the blocks sparse matrix
        std::partial_sum(KMLMA_ptr, KMLMA_ptr + KMLMA.size1() + 1, KMLMA_ptr);
        const std::size_t KMLMA_nonzero_values = KMLMA_ptr[KMLMA.size1()];
        std::ptrdiff_t* aux_index2_KMLMA= new std::ptrdiff_t[KMLMA_nonzero_values];
        double* aux_val_KMLMA= new double[KMLMA_nonzero_values];
        
        std::partial_sum(mKSAN_ptr, mKSAN_ptr + mKSAN.size1() + 1, mKSAN_ptr);
        const std::size_t mKSAN_nonzero_values = mKSAN_ptr[mKSAN.size1()];
        std::ptrdiff_t* aux_index2_mKSAN= new std::ptrdiff_t[mKSAN_nonzero_values];
        double* aux_val_mKSAN= new double[mKSAN_nonzero_values];
        
        std::partial_sum(mKSAM_ptr, mKSAM_ptr + mKSAM.size1() + 1, mKSAM_ptr);
        const std::size_t mKSAM_nonzero_values = mKSAM_ptr[mKSAM.size1()];
        std::ptrdiff_t* aux_index2_mKSAM= new std::ptrdiff_t[mKSAM_nonzero_values];
        double* aux_val_mKSAM= new double[mKSAM_nonzero_values];
        
        std::partial_sum(mKSASI_ptr, mKSASI_ptr + mKSASI.size1() + 1, mKSASI_ptr);
        const std::size_t mKSASI_nonzero_values = mKSASI_ptr[mKSASI.size1()];
        std::ptrdiff_t* aux_index2_mKSASI= new std::ptrdiff_t[mKSASI_nonzero_values];
        double* aux_val_mKSASI= new double[mKSASI_nonzero_values];
        
        std::partial_sum(mKSASA_ptr, mKSASA_ptr + mKSASA.size1() + 1, mKSASA_ptr);
        const std::size_t mKSASA_nonzero_values = mKSASA_ptr[mKSASA.size1()];
        std::ptrdiff_t* aux_index2_mKSASA= new std::ptrdiff_t[mKSASA_nonzero_values];
        double* aux_val_mKSASA = new double[mKSASA_nonzero_values];
        
        std::partial_sum(KSALMA_ptr, KSALMA_ptr + KSALMA.size1() + 1, KSALMA_ptr);
        const std::size_t KSALMA_nonzero_values = KSALMA_ptr[KSALMA.size1()];
        std::ptrdiff_t* aux_index2_KSALMA= new std::ptrdiff_t[KSALMA_nonzero_values];
        double* aux_val_KSALMA = new double[KSALMA_nonzero_values];
        
        std::partial_sum(KLMILMI_ptr, KLMILMI_ptr + KLMILMI.size1() + 1, KLMILMI_ptr);
        const std::size_t KLMILMI_nonzero_values = KLMILMI_ptr[KLMILMI.size1()];
        std::ptrdiff_t* aux_index2_KLMILMI= new std::ptrdiff_t[KLMILMI_nonzero_values];
        double* aux_val_KLMILMI = new double[KLMILMI_nonzero_values];
        
        std::partial_sum(KLMALMA_ptr, KLMALMA_ptr + KLMALMA.size1() + 1, KLMALMA_ptr);
        const std::size_t KLMALMA_nonzero_values = KLMALMA_ptr[KLMALMA.size1()];
        std::ptrdiff_t* aux_index2_KLMALMA = new std::ptrdiff_t[KLMALMA_nonzero_values];
        double* aux_val_KLMALMA = new double[KLMALMA_nonzero_values];
        
        #pragma omp parallel
        {            
            // We iterate over original matrix
            #pragma omp for
            for (IndexType i=0; i<rA.size1(); i++) {
                const IndexType row_begin = index1[i];
                const IndexType row_end   = index1[i+1];
                const IndexType local_row_id = mGlobalToLocalIndexing[i];

                if ( mWhichBlockType[i] == BlockType::MASTER) { // KMLMA
                    std::ptrdiff_t KMLMA_row_beg = KMLMA_ptr[local_row_id];
                    std::ptrdiff_t KMLMA_row_end = KMLMA_row_beg;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KMLMA block
                            const double value = values[j];
                            const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                            aux_index2_KMLMA[KMLMA_row_end] = local_col_id;
                            aux_val_KMLMA[KMLMA_row_end] = value;
                            ++KMLMA_row_end;
                        }
                    }
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_KMLMA + KMLMA_row_beg, aux_val_KMLMA + KMLMA_row_beg, KMLMA_row_end - KMLMA_row_beg);
                } else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { //either KSAN or KSAM or KSASA or KSASA or KSALM
                    std::ptrdiff_t mKSAN_row_beg = mKSAN_ptr[local_row_id];
                    std::ptrdiff_t mKSAN_row_end = mKSAN_row_beg;
                    std::ptrdiff_t mKSAM_row_beg = mKSAM_ptr[local_row_id];
                    std::ptrdiff_t mKSAM_row_end = mKSAM_row_beg;
                    std::ptrdiff_t mKSASI_row_beg = mKSASI_ptr[local_row_id];
                    std::ptrdiff_t mKSASI_row_end = mKSASI_row_beg;
                    std::ptrdiff_t mKSASA_row_beg = mKSASA_ptr[local_row_id];
                    std::ptrdiff_t mKSASA_row_end = mKSASA_row_beg;
                    std::ptrdiff_t KSALMA_row_beg = KSALMA_ptr[local_row_id];
                    std::ptrdiff_t KSALMA_row_end = KSALMA_row_beg;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        const double value = values[j];
                        const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                        if (mWhichBlockType[col_index] == BlockType::OTHER) {                 // KSAN block
                            aux_index2_mKSAN[mKSAN_row_end] = local_col_id;
                            aux_val_mKSAN[mKSAN_row_end] = value;
                            ++mKSAN_row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KSAM block
                            aux_index2_mKSAM[mKSAM_row_end] = local_col_id;
                            aux_val_mKSAM[mKSAM_row_end] = value;
                            ++mKSAM_row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSASI block
                            aux_index2_mKSASI[mKSASI_row_end] = local_col_id;
                            aux_val_mKSASI[mKSASI_row_end] = value;
                            ++mKSASI_row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KSASA block
                            aux_index2_mKSASA[mKSASA_row_end] = local_col_id;
                            aux_val_mKSASA[mKSASA_row_end] = value;
                            ++mKSASA_row_end;
                        } else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE) {     // KSALMA block (diagonal)
                            aux_index2_KSALMA[KSALMA_row_end] = local_col_id;
                            aux_val_KSALMA[KSALMA_row_end] = value;
                            ++KSALMA_row_end;
                        }
                    }
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_mKSAN + mKSAN_row_beg, aux_val_mKSAN + mKSAN_row_beg, mKSAN_row_end - mKSAN_row_beg);
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_mKSAM + mKSAM_row_beg, aux_val_mKSAM + mKSAM_row_beg, mKSAM_row_end - mKSAM_row_beg);
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_mKSASI + mKSASI_row_beg, aux_val_mKSASI + mKSASI_row_beg, mKSASI_row_end - mKSASI_row_beg);
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_mKSASA + mKSASA_row_beg, aux_val_mKSASA + mKSASA_row_beg, mKSASA_row_end - mKSASA_row_beg);
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_KSALMA + KSALMA_row_beg, aux_val_KSALMA + KSALMA_row_beg, KSALMA_row_end - KSALMA_row_beg);
                } else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { // KLMILMI
                    std::ptrdiff_t KLMILMI_row_beg = KLMILMI_ptr[local_row_id];
                    std::ptrdiff_t KLMILMI_row_end = KLMILMI_row_beg;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) { // KLMILMI block (diagonal)
                            const double value = values[j];
                            const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                            aux_index2_KLMILMI[KLMILMI_row_end] = local_col_id;
                            aux_val_KLMILMI[KLMILMI_row_end] = value;
                            ++KLMILMI_row_end;
                        }
                    }
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_KLMILMI + KLMILMI_row_beg, aux_val_KLMILMI + KLMILMI_row_beg, KLMILMI_row_end - KLMILMI_row_beg);
                } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { // KLMALMA
                    std::ptrdiff_t KLMALMA_row_beg = KLMALMA_ptr[local_row_id];
                    std::ptrdiff_t KLMALMA_row_end = KLMALMA_row_beg;
                    for (IndexType j=row_begin; j<row_end; j++) {
                        const IndexType col_index = index2[j];
                        if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE) { // KLMALMA block
                            const double value = values[j];
                            const IndexType local_col_id = mGlobalToLocalIndexing[col_index];
                            aux_index2_KLMALMA[KLMALMA_row_end] = local_col_id;
                            aux_val_KLMALMA[KLMALMA_row_end] = value;
                            ++KLMALMA_row_end;
                        }
                    }
                    SparseMatrixMultiplicationUtility::SortRow(aux_index2_KLMALMA + KLMALMA_row_beg, aux_val_KLMALMA + KLMALMA_row_beg, KLMALMA_row_end - KLMALMA_row_beg);
                }
            }
        }
        
        // Finally we build the final matrix
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(KMLMA, KMLMA.size1(), KMLMA.size2(), KMLMA_ptr, aux_index2_KMLMA, aux_val_KMLMA);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKSAN, mKSAN.size1(), mKSAN.size2(), mKSAN_ptr, aux_index2_mKSAN, aux_val_mKSAN);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKSAM, mKSAM.size1(), mKSAM.size2(), mKSAM_ptr, aux_index2_mKSAM, aux_val_mKSAM);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKSASI, mKSASI.size1(), mKSASI.size2(), mKSASI_ptr, aux_index2_mKSASI, aux_val_mKSASI);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKSASA, mKSASA.size1(), mKSASA.size2(), mKSASA_ptr, aux_index2_mKSASA, aux_val_mKSASA);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(KSALMA, KSALMA.size1(), KSALMA.size2(), KSALMA_ptr, aux_index2_KSALMA, aux_val_KSALMA);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(KLMILMI, KLMILMI.size1(), KLMILMI.size2(), KLMILMI_ptr, aux_index2_KLMILMI, aux_val_KLMILMI);
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(KLMALMA, KLMALMA.size1(), KLMALMA.size2(), KLMALMA_ptr, aux_index2_KLMALMA, aux_val_KLMALMA);
        
        // We compute directly the inverse of the KSALMA matrix 
        // KSALMA it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        // TODO: this can be optimized in OMP
        for (IndexType i = 0; i < mKLMAModified.size1(); ++i) {
            const double value = KSALMA(i, i);
            if (std::abs(value) > tolerance)
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
            if (std::abs(value) > tolerance)
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
        const SizeType assembling_slave_dof_initial_index = slave_inactive_dof_initial_index + slave_inactive_size;
        
        // The auxiliar index structure
        const SizeType nrows = mKDispModified.size1();
        const SizeType ncols = mKDispModified.size2();
        std::ptrdiff_t* K_disp_modified_ptr = new std::ptrdiff_t[nrows + 1];
        K_disp_modified_ptr[0] = 0;
        
        // Creating a buffer for parallel vector fill
        std::vector<std::ptrdiff_t> marker(nrows * ncols, -1);
        
        #pragma omp parallel
        {            
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KNSI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KNSA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = 0;
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNMM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KMSI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KMSA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = 0;
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {        // KSIM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSISI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KSISA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = 0;
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
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KLMASI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KLMASA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = 0;
                            ++K_disp_modified_cols;
                        }
                    }
                    K_disp_modified_ptr[local_row_id + 1] = K_disp_modified_cols;
                }
            }
        }
        
        #pragma omp parallel
        {
            if (slave_active_size > 0) {
                // Get access to master_auxKSAN data
                ComputeNonZeroBlocks(master_auxKSAN, K_disp_modified_ptr, marker, nrows, ncols,   master_dof_initial_index, other_dof_initial_index);
                
                // Get access to master_auxKSAM data
                ComputeNonZeroBlocks(master_auxKSAM, K_disp_modified_ptr, marker, nrows, ncols,   master_dof_initial_index, master_dof_initial_index);
                
                // Get access to master_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeNonZeroBlocks(master_auxKSASI, K_disp_modified_ptr, marker, nrows, ncols,   master_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to master_auxKSASA data
                ComputeNonZeroBlocks(master_auxKSASA, K_disp_modified_ptr, marker, nrows, ncols,   master_dof_initial_index, assembling_slave_dof_initial_index);
                
                // Get access to aslave_auxKSAN data
                ComputeNonZeroBlocks(aslave_auxKSAN, K_disp_modified_ptr, marker, nrows, ncols,   assembling_slave_dof_initial_index, other_dof_initial_index);
                
                // Get access to aslave_auxKSAM data
                ComputeNonZeroBlocks(aslave_auxKSAM, K_disp_modified_ptr, marker, nrows, ncols,   assembling_slave_dof_initial_index, master_dof_initial_index);
                
                // Get access to aslave_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeNonZeroBlocks(aslave_auxKSASI, K_disp_modified_ptr, marker, nrows, ncols,   assembling_slave_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to aslave_auxKSASA data
                ComputeNonZeroBlocks(aslave_auxKSASA, K_disp_modified_ptr, marker, nrows, ncols,   assembling_slave_dof_initial_index, assembling_slave_dof_initial_index);
            }
        }
        
        // We initialize the final sparse matrix
        std::partial_sum(K_disp_modified_ptr, K_disp_modified_ptr + nrows + 1, K_disp_modified_ptr);
        const std::size_t nonzero_values = K_disp_modified_ptr[nrows];
        std::ptrdiff_t* aux_index2_K_disp_modified = new std::ptrdiff_t[nonzero_values];
        double* aux_val_K_disp_modified = new double[nonzero_values];
        
        #pragma omp parallel
        {            
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + other_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + master_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KNSI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + slave_inactive_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KNSA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + assembling_slave_dof_initial_index;
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + other_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {         // KNMM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + master_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KMSI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + slave_inactive_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KMSA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + assembling_slave_dof_initial_index;
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
                            marker[nrows * local_row_id + local_col_id + other_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + other_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::MASTER) {        // KSIM block
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + master_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KSISI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + slave_inactive_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {  // KSISA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + assembling_slave_dof_initial_index;
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
                            marker[nrows * local_row_id + local_col_id + master_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + master_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) { // KLMASI block
                            marker[nrows * local_row_id + local_col_id + slave_inactive_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + slave_inactive_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        } else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE) {   // KLMASA block
                            marker[nrows * local_row_id + local_col_id + assembling_slave_dof_initial_index] = row_end + 1;
                            aux_index2_K_disp_modified[row_end] = local_col_id + assembling_slave_dof_initial_index;
                            aux_val_K_disp_modified[row_end] = value;
                            ++row_end;
                        }
                    }
                }
            }
        }
        
        // TODO: Think about OMP
        // Filling the remaining marker
        std::ptrdiff_t aux_marker = 0;
        for (std::size_t i = 0; i < nrows; i++) {
            // Look for the maximum first
            for (std::size_t j = 0; j < ncols; j++) {
                const IndexType aux_index = nrows * i + j;
                if (marker[aux_index] > -1) {
                    if (marker[aux_index] > aux_marker)
                        aux_marker = marker[aux_index];
                }
            }
            // Assign now
            for (std::size_t j = 0; j < ncols; j++) {
                const IndexType aux_index = nrows * i + j;
                if (marker[aux_index] > -1) {
                    if (marker[aux_index] == 0) { // Case pending to assign
                        ++aux_marker;
                        marker[aux_index] = aux_marker;
                        aux_index2_K_disp_modified[aux_marker - 1] = -1;
                    }
                }
            }
        }
        
        #pragma omp parallel
        {
            if (slave_active_size > 0) {
                // Get access to master_auxKSAN data
                ComputeAuxiliarValuesBlocks(master_auxKSAN, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows, master_dof_initial_index, other_dof_initial_index);
                
                // Get access to master_auxKSAM data
                ComputeAuxiliarValuesBlocks(master_auxKSAM, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  master_dof_initial_index, master_dof_initial_index);
                
                // Get access to master_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeAuxiliarValuesBlocks(master_auxKSASI, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  master_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to master_auxKSASA data
                ComputeAuxiliarValuesBlocks(master_auxKSASA, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  master_dof_initial_index, assembling_slave_dof_initial_index);
                
                // Get access to aslave_auxKSAN data
                ComputeAuxiliarValuesBlocks(aslave_auxKSAN, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  assembling_slave_dof_initial_index, other_dof_initial_index);
                
                // Get access to aslave_auxKSAM data
                ComputeAuxiliarValuesBlocks(aslave_auxKSAM, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  assembling_slave_dof_initial_index, master_dof_initial_index);
                
                // Get access to aslave_auxKSASI data
                if (slave_inactive_size > 0)
                    ComputeAuxiliarValuesBlocks(aslave_auxKSASI, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  assembling_slave_dof_initial_index, slave_inactive_dof_initial_index);
                
                // Get access to aslave_auxKSASA data
                ComputeAuxiliarValuesBlocks(aslave_auxKSASA, aux_index2_K_disp_modified, aux_val_K_disp_modified, marker, nrows,  assembling_slave_dof_initial_index, assembling_slave_dof_initial_index);
            }
        }
        
        // We reorder the rows
        #pragma omp parallel
        {
            #pragma omp for
            for (int i_row=0; i_row<static_cast<int>(nrows); i_row++) {
                const std::ptrdiff_t row_beg = K_disp_modified_ptr[i_row];
                const std::ptrdiff_t row_end = K_disp_modified_ptr[i_row + 1];
                
                for(std::ptrdiff_t j = 1; j < row_end - row_beg; ++j) {
                    const std::ptrdiff_t c = aux_index2_K_disp_modified[j + row_beg];
                    const double v = aux_val_K_disp_modified[j + row_beg];

                    std::ptrdiff_t i = j - 1;

                    while(i >= 0 && aux_index2_K_disp_modified[i + row_beg] > c) {
                        aux_index2_K_disp_modified[i + 1 + row_beg] = aux_index2_K_disp_modified[i + row_beg];
                        aux_val_K_disp_modified[i + 1 + row_beg] = aux_val_K_disp_modified[i + row_beg];
                        i--;
                    }
                    
                    aux_index2_K_disp_modified[i + 1 + row_beg] = c;
                    aux_val_K_disp_modified[i + 1 + row_beg] = v;
                }
            }
        }
        
        // Finally we build the final matrix
        SparseMatrixMultiplicationUtility::CreateSolutionMatrix(mKDispModified, nrows, ncols, K_disp_modified_ptr, aux_index2_K_disp_modified, aux_val_K_disp_modified);
        
//         // DEBUG
//         CheckMatrix(rA);
//         LOG_MATRIX_PRETTY(rA)
//         LOG_MATRIX_PRETTY(mKDispModified)
        
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
     * @param NRows The total number of rows of the final matrix
     * @param NCols The total number of columns of the final matrix
     * @param InitialIndexRow The initial row index of the auxiliar block in the final matrix 
     * @param InitialIndexColumn The initial column index of the auxiliar block in the final matrix 
     * @todo Check the col_index!!!!!!
     */
    inline void ComputeNonZeroBlocks(
        const SparseMatrixType& AuxK,
        std::ptrdiff_t* KPtr,
        std::vector<std::ptrdiff_t>& Marker,
        const SizeType NRows,
        const SizeType NCols,
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
                const IndexType index_marker = NRows * (i + InitialIndexRow) + col_index;
                if (Marker[index_marker] < 0) {
                    Marker[index_marker] = 0; 
                    ++K_disp_modified_cols;
                }
            }
        }
    }
    
    /**
     * @brief This is a method to compute the contribution of the auxiliar blocks
     * @param AuxK The auxiliar block
     * @param AuxIndex2 The indexes of the non zero columns
     * @param AuxVals The values of the final matrix
     * @param Marker A marker to check the already asigned values
     * @param NRows The total number of rows of the final matrix
     * @param InitialIndexRow The initial row index of the auxiliar block in the final matrix 
     * @param InitialIndexColumn The initial column index of the auxiliar block in the final matrix 
     * @todo Check the col_index!!!!!!
     */
    inline void ComputeAuxiliarValuesBlocks(
        const SparseMatrixType& AuxK,
        std::ptrdiff_t* AuxIndex2,
        double* AuxVals,
        const std::vector<std::ptrdiff_t>& Marker,
        const SizeType NRows,
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
            
            for (IndexType j=aux_K_row_begin; j<aux_K_row_end; j++) {
                const IndexType col_index = InitialIndexColumn + aux_K_index2[j];
                const IndexType index = Marker[NRows * (i + InitialIndexRow) + col_index] - 1; 
                if (AuxIndex2[index] == -1) {
                    AuxIndex2[index] = col_index;
                    AuxVals[index]  = -aux_values[j];
                } else {
                    AuxVals[index] += -aux_values[j];
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
        const SizeType lm_active_size = mLMActiveIndices.size();
        const SizeType total_size = other_dof_size + master_size + slave_inactive_size + slave_active_size;
        
        // Resize in case the size is not correct
        if (ResidualU.size() != total_size )
            ResidualU.resize (total_size, false);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(other_dof_size); i++)
            ResidualU[i] = rTotalResidual[mOtherIndices[i]];
        
        // The corresponding residual for the active slave DoF's
        VectorType aux_res_active_slave(slave_active_size);
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(slave_active_size); i++)
            aux_res_active_slave[i] = rTotalResidual[mSlaveActiveIndices[i]];
        
        if (slave_active_size > 0) {
            // We compute the complementary residual for the master dofs        
            VectorType aux_complement_master_residual(master_size);
            TSparseSpaceType::Mult(mPOperator, aux_res_active_slave, aux_complement_master_residual);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(master_size); i++)
                ResidualU[other_dof_size + i] = rTotalResidual[mMasterIndices[i]] - aux_complement_master_residual[i];
        } else {
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(master_size); i++)
                ResidualU[other_dof_size + i] = rTotalResidual[mMasterIndices[i]];
        }
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(slave_inactive_size); i++)
            ResidualU[other_dof_size + master_size + i] = rTotalResidual[mSlaveInactiveIndices[i]];
        
        if (slave_active_size > 0) {
            // We compute the complementary residual for the master dofs        
            VectorType aux_complement_active_lm_residual(lm_active_size);
            TSparseSpaceType::Mult(mCOperator, aux_res_active_slave, aux_complement_active_lm_residual);
            
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(lm_active_size); i++)
                ResidualU[other_dof_size + master_size + slave_inactive_size + i] = rTotalResidual[mLMActiveIndices[i]] - aux_complement_active_lm_residual[i];
        } else {
            #pragma omp parallel for
            for (int i = 0; i<static_cast<int>(lm_active_size); i++)
                ResidualU[other_dof_size + master_size + slave_inactive_size + i] = rTotalResidual[mLMActiveIndices[i]];
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
     * @brief This method is intended to use to check the matrix
     * @param rA The matrix to be checked 
     */
    double CheckMatrix (const SparseMatrixType& rA)
    {
        // Get access to A data
        const std::size_t* index1 = rA.index1_data().begin();
        const std::size_t* index2 = rA.index2_data().begin();
        const double* values = rA.value_data().begin();
        double norm = 0.0;
        for (std::size_t i=0; i<rA.size1(); i++) {
            std::size_t row_begin = index1[i];
            std::size_t row_end   = index1[i+1];
            if (row_end - row_begin == 0)
                KRATOS_WARNING("Checking sparse matrix") << "Line " << i << " has no elements" << std::endl;
            
            for (std::size_t j=row_begin; j<row_end; j++) {
                KRATOS_ERROR_IF( index2[j] > rA.size2() ) << "Array above size of A" << std::endl;
                norm += values[j]*values[j];
            }
        }
        
        return std::sqrt (norm);
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