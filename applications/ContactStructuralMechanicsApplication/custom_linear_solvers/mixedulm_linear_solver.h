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
     * @param rModelPart The the model part of the problem to solve 
     * @param MaxIterationNumber The maximal number of iterations
     */
    MixedULMLinearSolver (
        LinearSolverPointerType pSolverDispBlock,
        ModelPart& rModelPart,
        const double MaxTolerance,
        const std::size_t MaxIterationNumber
        ) : BaseType (MaxTolerance, MaxIterationNumber),
            mpSolverDispBlock(pSolverDispBlock),
            mrModelPart(rModelPart)
    {
        // Initializing the remaining variables
        mBlocksAreAllocated = false;
        mIsInitialized = false;
    }
    
    /**
     * @brief Second constructor, it uses a Kratos parameters as input instead of direct input
     * @param pSolverDispBlock The linear solver used for the displacement block
     * @param rModelPart The the model part of the problem to solve 
     * @param ThisParameters The configuration parameters considered
     */
    
    MixedULMLinearSolver(
            LinearSolverPointerType pSolverDispBlock,
            ModelPart& rModelPart,
            Parameters ThisParameters =  Parameters(R"({})")
            ): BaseType (),
               mpSolverDispBlock(pSolverDispBlock),
               mrModelPart(rModelPart)

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
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();
        noalias(mLM) = ZeroVector(mLMActiveIndices.size());
        noalias(mDisp)  = ZeroVector(total_size);

        // Get the u and lm residuals
        GetUPart (rB, mResidualDisp);

        // Solve u block
        mpSolverDispBlock->Solve (mKDispModified, mDisp, mResidualDisp);

        // Now we compute the residual of the LM
        GetLMPart (rB, mResidualLM);

        // Solve LM
        // LM = D⁻1*rLM
        TSparseSpaceType::Mult (mKLMAModified, mResidualLM, mLM);

        // Write back solution
        SetUPart(rX, mDisp);
        SetLMPart(rX, mLM);
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
        mLM.clear();
        mResidualDisp.clear();
        mResidualLM.clear();
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
     * @todo Update this for dual LM
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

        SparseMatrixType KNN(mOtherIndices.size(), mOtherIndices.size());                     /// The displacement-displacement block
        SparseMatrixType KNM(mOtherIndices.size(), mMasterIndices.size());                    /// The displacement-master block
        SparseMatrixType KMN(mMasterIndices.size(), mOtherIndices.size());                    /// The master-displacement block
        SparseMatrixType KNSI(mOtherIndices.size(), mSlaveInactiveIndices.size());            /// The displacement-slave inactive block 
        SparseMatrixType KNSA(mOtherIndices.size(), mSlaveActiveIndices.size());              /// The displacement-slave active block 
        SparseMatrixType KSIN(mSlaveInactiveIndices.size(), mOtherIndices.size());            /// The slave inactive-displacement block
        SparseMatrixType KMM(mMasterIndices.size(), mMasterIndices.size());                   /// The master-master block
        SparseMatrixType KSIM(mSlaveInactiveIndices.size(), mMasterIndices.size());           /// The slave inactive-master block        
        SparseMatrixType KMSI(mMasterIndices.size(), mSlaveInactiveIndices.size());           /// The master-inactive slave block
        SparseMatrixType KMSA(mMasterIndices.size(), mSlaveActiveIndices.size());             /// The master-active slave block
        SparseMatrixType KMLMI(mMasterIndices.size(), mLMInactiveIndices.size());             /// The master-inactive LM block (this is the big block of M)
        SparseMatrixType KSISI(mSlaveInactiveIndices.size(), mSlaveInactiveIndices.size());   /// The slave inactive-inactive slave block
        SparseMatrixType KSISA(mSlaveInactiveIndices.size(), mSlaveActiveIndices.size());     /// The inactive slave-active slave block
//         SparseMatrixType KSALMI(mSlaveActiveIndices.size(), mLMInactiveIndices.size());       /// The active slave-inactive LM block (this is the big block of D) // NOTE: Not necessary, you can fix the value directly 
        SparseMatrixType KSALMA(mSlaveActiveIndices.size(), mLMActiveIndices.size());         /// The active slave-active LM block (this is the big block of D, diagonal) 
        
//         SparseMatrixType KSILMI(mSlaveInactiveIndices.size(), mLMInactiveIndices.size());     /// The inactive slave-inactive LM block (this is the big block of D, diagonal) // NOTE: Not necessary, you can fix the value directly
        SparseMatrixType KSILMA(mSlaveInactiveIndices.size(), mLMActiveIndices.size());       /// The inactive slave-active LM block (this is the big block of D) 
        SparseMatrixType KLMAM(mLMActiveIndices.size(), mMasterIndices.size());               /// The active LM-master block (this is the contact contribution, which gives the quadratic convergence)
        SparseMatrixType KLMASI(mLMActiveIndices.size(), mSlaveInactiveIndices.size());       /// The active LM-inactive slave block (this is the contact contribution, which gives the quadratic convergence) 
        SparseMatrixType KLMASA(mLMActiveIndices.size(), mSlaveActiveIndices.size());         /// The active LM-active slave block (this is the contact contribution, which gives the quadratic convergence)
//         SparseMatrixType KLMILMI(mLMInactiveIndices.size(), mLMInactiveIndices.size());       /// The inactive LM- inactive LM block (diagonal) // NOTE: Not necessary, you can fix the value directly
        
        if (!mrModelPart.GetProcessInfo()[ACTIVE_SET_CONVERGED])
            RebuildActiveSet();
        
        if (NeedAllocation)
            AllocateBlocks();

        // Allocate the blocks by push_back
        for (unsigned int i=0; i<rA.size1(); i++) {
            unsigned int row_begin = index1[i];
            unsigned int row_end   = index1[i+1];
            unsigned int local_row_id = mGlobalToLocalIndexing[i];

            if ( mWhichBlockType[i] == BlockType::OTHER) { //either KNN or KNM or KNSI or KNSA
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)                // KNN block
                        KNN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)          // KNM block
                        KNM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE)  // KNSI block
                        KNSI.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)    // KNSA block
                        KNSA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::MASTER) { //either KMN or KMM or KMSI or KMLM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)                // KMN block
                        KMN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)          // KNMM block
                        KMM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE)  // KMSI block
                        KMSI.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)    // KMSA block
                        KMSA.push_back ( local_row_id, local_col_id, value);
                    else if ( mWhichBlockType[col_index] == BlockType::LM_INACTIVE)    // KMLMI block
                        KMLMI.push_back ( local_row_id, local_col_id, value);
                    else                                                               // KMLMA block
                        mKMLMA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE_INACTIVE) { //either KSIN or KSIM or KSISI or KSISA or KSILM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)               // KSIN block
                        KSIN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER)         // KSIMM block
                        KSIM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) // KSISI block
                        KSISI.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)   // KSISA block
                        KSISA.push_back ( local_row_id, local_col_id, value);
                    else if ( mWhichBlockType[col_index] == BlockType::LM_ACTIVE)     // KSILMA block
                        KSILMA.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE_ACTIVE) { //either KSAN or KSAM or KSASA or KSASA or KSALM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
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
//             } else if ( mWhichBlockType[i] == BlockType::LM_INACTIVE) { // KLMILMI
//                 for (unsigned int j=row_begin; j<row_end; j++) {
//                     unsigned int col_index = index2[j];
//                     double value = values[j];
//                     unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
//                     if (mWhichBlockType[col_index] == BlockType::LM_INACTIVE) // KLMILMI block (diagonal)
//                         KLMILMI.push_back ( local_row_id, local_col_id, value);
//                 }
            } else if ( mWhichBlockType[i] == BlockType::LM_ACTIVE) { //either KLMAM or KLMASI or KLMASA
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::MASTER)              // KLMM block
                        KLMAM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_INACTIVE) // KLMSI block
                        KLMASI.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE_ACTIVE)   // KLMSA block
                        KLMASA.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::LM_ACTIVE)      // KLMALMA block
                        mKLMALMA.push_back ( local_row_id, local_col_id, value);
                }
            }
        }
        
        // We compute directly the inverse of the KSALMA matrix 
        // KSALMA it is supposed to be a diagonal matrix (in fact it is the key point of this formulation)
        // (NOTE: technically it is not a stiffness matrix, we give that name)
        for (std::size_t i = 0; i < mKLMAModified.size1(); ++i) {
            const double value = KSALMA(i, i);
            if (value > 0.0)
                mKLMAModified.push_back(i, i, 1.0/value);
            else // Auxiliar value
                mKLMAModified.push_back(i, i, 1.0);
        }

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
    
    ModelPart& mrModelPart; /// The the model part of the problem to solve 
     
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
    
    SparseMatrixType mKDispModified;  /// The modified displacement block
    SparseMatrixType mKLMAModified;   /// The modified LM block (inverted diagonal)
    
    SparseMatrixType mKSAN;    /// The slave active-displacement block
    SparseMatrixType mKSAM;    /// The active slave-master block
    SparseMatrixType mKSASI;   /// The active slave-inactive slave block
    SparseMatrixType mKSASA;   /// The inactive slave-active slave block
    SparseMatrixType mKMLMA;   /// The master-active LM block (this is the big block of M) 
    SparseMatrixType mKLMALMA; /// The active LM-active LM block
    
    VectorType mResidualLM;   /// The residual of the lagrange multipliers
    VectorType mResidualDisp; /// The residual of the rest of displacements
    VectorType mLM;           /// The solution of the lagrange multiplies
    VectorType mDisp;         /// The solution of the rest of displacements
        
    ///@}
    ///@name Private Operators
    ///@{
    
    ///@}
    ///@name Private Operations
    ///@{
    
    /**
     * @brief This method rebuilds the active/inactive set
     */
    inline void RebuildActiveSet() 
    {
        NodesArrayType& nodes_array = mrModelPart.GetSubModelPart("Contact").Nodes();
        for(int i = 0; i < static_cast<int>(nodes_array.size()); ++i) 
        {
            auto it_node = nodes_array.begin() + i;
            
//             if // TODO: FINISH ME!!!!!
        }
    }
    
    /**
     * @brief It allocates all the blocks and operators
     */
    inline void AllocateBlocks() 
    {
        // We clear the matrixes
        mKDispModified.clear(); /// The modified displacement block
        mKLMAModified.clear();   /// The modified LM block (diagonal)

        // We do the allocation
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();
        mKDispModified.resize(total_size, total_size);                           /// The modified displacement block
        mKLMAModified.resize(mLMActiveIndices.size(), mLMActiveIndices.size());  /// The modified LM block (diagonal)
        mKSAN.resize(mSlaveActiveIndices.size(), mOtherIndices.size());          /// The slave active-displacement block        
        mKSAM.resize(mSlaveActiveIndices.size(), mMasterIndices.size());         /// The active slave-master block
        mKSASI.resize(mSlaveActiveIndices.size(), mSlaveInactiveIndices.size()); /// The active slave-inactive slave block
        mKSASA.resize(mSlaveActiveIndices.size(), mSlaveActiveIndices.size());   /// The active slave-slave active block
        mKMLMA.resize(mLMActiveIndices.size(), mLMActiveIndices.size());         /// The master-active LM block (this is the big block of M) 
        mKLMALMA.resize(mLMActiveIndices.size(), mLMActiveIndices.size());       /// The active LM-active LM block
        
        mResidualLM.resize(mLMActiveIndices.size() );
        mResidualDisp.resize(total_size );
        
        mLM.resize(mLMActiveIndices.size());
        mDisp.resize(total_size);
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
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + mSlaveActiveIndices.size();
        if (ResidualU.size() != total_size )
            ResidualU.resize (total_size, false);
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mOtherIndices.size()); i++)
            ResidualU[i] = rTotalResidual[mOtherIndices[i]];
        
        VectorType aux_res(mSlaveActiveIndices.size());
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveActiveIndices.size()); i++)
            aux_res[i] = rTotalResidual[mSlaveActiveIndices[i]];
        
//         SparseMatrixType aux_matrix();
//         TSparseSpaceType::Mult(
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mMasterIndices.size()); i++)
            ResidualU[mOtherIndices.size() + i] = rTotalResidual[mMasterIndices[i]];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveInactiveIndices.size()); i++)
            ResidualU[mOtherIndices.size() + mMasterIndices.size() + i] = rTotalResidual[mSlaveInactiveIndices[i]];
        
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(mSlaveActiveIndices.size()); i++)
            ResidualU[mOtherIndices.size() + mMasterIndices.size() + mSlaveInactiveIndices.size() + i] = rTotalResidual[mLMActiveIndices[i]];
    }

    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to lm-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param rResidualLM The vector containing the residual relative to the LM
     */
    inline void GetLMPart (
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
     * @brief This method writes the Lagrange Multiplier part
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualLM The vector containing the residual relative to the LM
     */
    inline void SetLMPart (
        VectorType& rTotalResidual, 
        const VectorType& ResidualLM
        )
    {
        #pragma omp parallel for
        for (int i = 0; i< static_cast<int>(ResidualLM.size()); i++)
            rTotalResidual[mLMActiveIndices[i]] = ResidualLM[i];
    }

    /**
     * @brief Compute the Pressure System Matrix
     * @details Compute the System Matrix A = L - D*Inv(Diag(S))*G. The multiplication
     * is performed in random order, so each row will be stored in a temporary
     * variable, ordered and copied in input matrix A.
     * @param A The system matrix
     * @param K
     * @param rG
     * @param rD
     * @param rL
     * @param diagK
     */
    void CalculateShurComplement (
        SparseMatrixType& A,
        SparseMatrixType& K,
        SparseMatrixType& rG,
        SparseMatrixType& rD,
        SparseMatrixType& rL,
        VectorType& diagK
    )
    {
        // Retrieve matrices

        // Compute Inv(Diag(S))
        VectorType& rIDiagS = diagK;

        typedef vector<int> IndexVectorType;
        typedef typename boost::numeric::ublas::matrix_row< SparseMatrixType > RowType;

        const int diagonal_size = static_cast<int>(diagK.size());
        
        #pragma omp parallel for
        for ( int i = 0; i < diagonal_size; i++)
            rIDiagS[i] = 1.0/diagK[i];
        
        OpenMPUtils::PartitionVector Partition;
        int NumThreads = OpenMPUtils::GetNumThreads();
        OpenMPUtils::DivideInPartitions (A.size1(),NumThreads,Partition);
        #pragma omp parallel
        {
            const int k = OpenMPUtils::ThisThread();
            VectorType current_row(K.size2());

            for (unsigned int i = 0; i < rL.size1(); i++) current_row[i] = 0.0;
            
            IndexVectorType next = IndexVectorType(rL.size1());
            for (unsigned int m=0; m < rL.size1(); m++) next[m] = -1;
            std::size_t number_terms = 0; // Full positions in a row
            std::vector<unsigned int> used_cols = std::vector<unsigned int>();
            used_cols.reserve (rL.size1());
            
            for ( int row_index = Partition[k] ; row_index != Partition[k+1] ; row_index++ ) {
                RowType row_D (rD,row_index);
                RowType row_L (rL,row_index);
                int head = -2;
                std::size_t Length = 0;
                // Write L in A
                for ( auto it_L = row_L.begin();  it_L != row_L.end(); it_L++ ) {
                    current_row (it_L.index() ) = *it_L;
                    if ( next[it_L.index()] == -1) {
                        next[it_L.index()] = head;
                        head = it_L.index();
                        Length++;
                    }
                }
                // Substract D*Inv(Diag(S))*G
                for ( auto it_D = row_D.begin(); it_D != row_D.end(); it_D++ ) {
                    RowType row_G (rG,it_D.index() );
                    for ( auto it_G = row_G.begin(); it_G != row_G.end(); it_G++ ) {
                        current_row[it_G.index()] -= (*it_D) * rIDiagS[it_D.index()] * (*it_G);
                        if ( next[it_G.index()] == -1) {
                            next[it_G.index()] = head;
                            head = it_G.index();
                            Length++;
                        }
                    }
                }
                
                // Identify full terms for ordering
                for ( std::size_t i = 0; i < Length; i++) {
                    if ( next[head] != -1 ) {
                        used_cols.push_back (head);
                        number_terms++;
                    }
                    int temp = head;
                    head = next[head];
                    // Clear 'next' for next iteration
                    next[temp] = -1;
                }
                
                // Sort Column indices
                SortCols (used_cols,number_terms);
                // Fill matrix row, then clean temporary variables.
                RowType RowA (A,row_index);
                std::size_t n = 0;
                unsigned int Col;
                for ( auto ItA = RowA.begin(); ItA != RowA.end(); ItA++) {
                    Col = used_cols[n++];
                    *ItA = current_row[Col];
                    current_row[Col] = 0;
                }
                number_terms = 0;
                used_cols.resize (0,false);
            }
        }
    }

    /**
     * @brief Helper function for Sytem matrix functions
     * @param ColList The list of columns
     * @param NumCols The number of columns
     */
    void SortCols (
        std::vector<SizeType>& ColList,
        SizeType& NumCols
        )
    {
        bool swap = true;
        SizeType d = NumCols;
        int temp;
        while ( swap || d > 1 ) {
            swap = false;
            d = (d+1) /2;
            for ( SizeType i=0; i< (NumCols - d); i++) {
                if ( ColList[i+d] < ColList[i] ) {
                    temp = ColList[i+d];
                    ColList[i+d] = ColList[i];
                    ColList[i] = temp;
                    swap = true;
                }
            }
        }
    }

    /**
     * @brief Identify non-zero tems in the system matrix
     * @param A The system matrix
     */
    void ConstructSystemMatrix(SparseMatrixType& A)
    {
        typedef OpenMPUtils::PartitionVector PartitionVector;
        typedef typename boost::numeric::ublas::matrix_row< SparseMatrixType > RowType;

        PartitionVector partition;
        const int number_threads = OpenMPUtils::GetNumThreads();

        OpenMPUtils::DivideInPartitions(A.size1(), number_threads, partition);

        for ( int k = 0 ; k < number_threads ; k++) {
            // This code is serial, the pragma is here to ensure that each
            // row block is assigned to the processor that will fill it
            #pragma omp parallel
            if ( OpenMPUtils::ThisThread() == k) {

                IndexVectorType next(rL.size1());
                for (SizeType m = 0; m < rL.size1(); m++) next[m] = -1;

                SizeType number_terms = 0; // Full positions in a row
                std::vector<SizeType> used_cols;
                used_cols.reserve(rL.size1());

                for ( int row_index = partition[k] ; row_index != partition[k+1] ; row_index++ ) {
                    RowType row_D(rD, row_index);
                    RowType row_L(rL, row_index);

                    int head = -2;
                    SizeType length = 0;

                    // Terms filled by L
                    for ( auto it_L = row_L.begin(); it_L != row_L.end(); it_L++ ){
                        if ( next[it_L.index()] == -1) {
                            next[it_L.index()] = head;
                            head = it_L.index();
                            length++;
                        }
                    }

                    // Additional terms due to D*Inv(Diag(S))*G
                    for ( auto it_D = row_D.begin();  it_D != row_D.end(); it_D++ ) {
                        RowType row_G(rG,it_D.index());

                        for ( auto it_G = row_G.begin(); it_G != row_G.end(); it_G++ ) {
                            if ( next[it_G.index()] == -1) {
                                next[it_G.index()] = head;
                                head = it_G.index();
                                length++;
                            }
                        }
                    }

                    // Identify full terms for ordering
                    for ( std::size_t i = 0; i < length; i++) {
                        if ( next[head] != -1 ) {
                            used_cols.push_back(head);
                            number_terms++;
                        }

                        int temp = head;
                        head = next[head];

                        // Clear 'Next' for next iteration
                        next[temp] = -1;
                    }

                    // Sort Column indices
                    SortCols(used_cols,number_terms);

                    // Store row in matrix, clean temporary variables
                    for ( unsigned int i = 0; i < number_terms; i++) {
                        A.push_back(row_index,used_cols[i],0);
                    }
                    number_terms = 0;
                    used_cols.resize(0,false);
                }
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
