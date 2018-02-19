//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:                 BSD License
//                                         Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
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
#include "linear_solvers/reorderer.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/model_part.h"
#include "linear_solvers/iterative_solver.h"
#include "utilities/openmp_utils.h"

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
 * @author Vicente Mataix Ferrándiz
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
            SLAVE,
            LM
            };
    ///@}
    ///@name Life Cycle
    ///@{
    
    /**
     * @brief Default constructor
     * @param pSolverDispBlock The linear solver used for the displacement block
     * @param pSolverLMBlock The linear solver for the LM block
     * @param MaxTolerance The maximal tolrance considered
     * @param MaxIterationNumber The maximal number of iterations
     * @param KrylovSpaceDimension The dimension of the Krylov space
     */
    MixedULMLinearSolver (
        LinearSolverPointerType pSolverDispBlock,
        LinearSolverPointerType pSolverLMBlock,
        const double MaxTolerance,
        const std::size_t MaxIterationNumber,
        const std::size_t KrylovSpaceDimension
        ) : BaseType (MaxTolerance, MaxIterationNumber),
            mpSolverDispBlock(pSolverDispBlock),
            mpSolverLMBlock(pSolverLMBlock),
            mKrylovSpaceDimension(KrylovSpaceDimension)
    {
        // Initializing the remaining variables
        mBlocksAreAllocated = false;
        mIsInitialized = false;
    }
    
    /**
     * @brief Second constructor, it uses a Kratos parameters as input instead of direct input
     * @param pSolverDispBlock The linear solver used for the displacement block
     * @param pSolverLMBlock The linear solver for the LM block
     * @param ThisParameters The configuration parameters considered
     */
    
    MixedULMLinearSolver(
            LinearSolverPointerType pSolverDispBlock,
            LinearSolverPointerType pSolverLMBlock,
            Parameters ThisParameters =  Parameters(R"({})")
            ): BaseType (),
               mpSolverDispBlock(pSolverDispBlock),
               mpSolverLMBlock(pSolverLMBlock)

    {
        KRATOS_TRY

        // Now validate agains defaults -- this also ensures no type mismatch
        Parameters default_parameters = GetDefaultParameters(); 
        ThisParameters.ValidateAndAssignDefaults(default_parameters);

        this->SetTolerance( ThisParameters["tolerance"].GetDouble() );
        this->SetMaxIterationsNumber( ThisParameters["MaxIterationsation"].GetInt() );
        mKrylovSpaceDimension = ThisParameters["gmres_krylov_space_dimension"].GetInt();
        
        // Initializing the remaining variables
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
//             mpSolverDispBlock->Initialize(mKNN, mDisp, mResidualDisp);
//             mpSolverLMBlock->Initialize(mLMLM, mLM, mResidualLM);
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
            FillBlockMatrices (true, rA);
            mBlocksAreAllocated = true;
        } else {
            FillBlockMatrices (false, rA);
            mBlocksAreAllocated = true;
        }
        
        if(mIsInitialized == false) this->Initialize(rA,rX,rB);

        // Initialize solvers // TODO: mDisp is not in the correct order!!!!
        mpSolverDispBlock->InitializeSolutionStep(mKDispModified, mDisp, mResidualDisp);
        mpSolverLMBlock->InitializeSolutionStep(mKLMModified, mLM, mResidualLM);
    }

    /** 
     * @brief This function actually performs the solution work, eventually taking advantage of what was done before in the
     * Initialize and InitializeSolutionStep functions.
     * @param rA System matrix
     * @param rX Solution vector. it's also the initial guess for iterative linear solvers.
     * @param rB Right hand side vector.
    */
    void PerformLMLMolutionStep (
        SparseMatrixType& rA, 
        VectorType& rX, 
        VectorType& rB
        ) override
    {
        const std::size_t m = mKrylovSpaceDimension;
        const std::size_t MaxIterations = BaseType::GetMaxIterationsNumber();
        const double tol = BaseType::GetTolerance();
        GMRESSolve (rA, rX, rB, m, MaxIterations, tol);
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
        mpSolverLMBlock->FinalizeSolutionStep(mKLMModified, mLM, mResidualLM);
    }
    
    /** 
     * @brief This function is designed to clean up all internal data in the solver.
     * @details Clear is designed to leave the solver object as if newly created. After a clear a new Initialize is needed
     */
    void Clear() override
    {        
        mKDispModified.clear();  /// The modified displacement block
        mKLMModified.clear();  /// The modified LM block (diagonal)
        mBlocksAreAllocated = false;
        mpSolverDispBlock->Clear();
        mpSolverLMBlock->Clear();
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

        this->PerformLMLMolutionStep (rA,rX,rB);

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
     */
    void ProvideAdditionalData (
        SparseMatrixType& rA,
        VectorType& rX,
        VectorType& rB,
        typename ModelPart::DofsArrayType& rDofSet,
        ModelPart& rModelPart
        ) override
    {
        //count LM dofs
        SizeType n_lm_dofs = 0, n_master_dofs = 0, n_slave_dofs = 0;
        SizeType tot_active_dofs = 0;
        for (auto it = rDofSet.begin(); it!=rDofSet.end(); it++) {
            if (it->EquationId() < rA.size1()) {
                tot_active_dofs++;
                if (it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_X || 
                    it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Y || 
                    it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Z) {
                        n_lm_dofs++;
                } else if (it->GetVariable().Key() == DISPLACEMENT_X || 
                    it->GetVariable().Key() == DISPLACEMENT_Y || 
                    it->GetVariable().Key() == DISPLACEMENT_Z) {
                    const IndexType node_id = it->Id();
                    NodeType::Pointer pnode = rModelPart.pGetNode(node_id);
                    if (pnode->Is(INTERFACE)) {
                        if (pnode->Is(MASTER)) {
                            n_master_dofs++;
                        } else if (pnode->Is(SLAVE)) { // TODO: Fix all the INACTIVE slave nodes
                            n_slave_dofs++;
                        }
                    }
                }
            }
        }

        KRATOS_ERROR_IF(tot_active_dofs != rA.size1()) << "Total system size does not coincide with the free dof map" << std::endl;

        // Resize arrays as needed
        mMasterIndices.resize (n_master_dofs,false);
        mSlaveIndices.resize (n_slave_dofs,false);
        mLMIndices.resize (n_lm_dofs,false);

        const SizeType other_dof_size = tot_active_dofs - n_lm_dofs - n_master_dofs - n_slave_dofs;
        mOtherIndices.resize (other_dof_size,false);
        mGlobalToLocalIndexing.resize (tot_active_dofs,false);
        mWhichBlockType.resize (tot_active_dofs, BlockType::OTHER);
        
        /**
         * Construct aux_lists as needed
         * "other_counter[i]" i will contain the position in the global system of the i-th NON-LM node
         * "lm_counter[i]" will contain the in the global system of the i-th NON-LM node
         * mGlobalToLocalIndexing[i] will contain the position in the local blocks of the
         */
        SizeType lm_counter = 0, slave_counter = 0, master_counter = 0;
        SizeType other_counter = 0;
        IndexType global_pos = 0;
        for (auto it = rDofSet.begin(); it!=rDofSet.end(); it++) {
            if (it->EquationId() < rA.size1()) {
                if (it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_X || 
                    it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Y || 
                    it->GetVariable().Key() == VECTOR_LAGRANGE_MULTIPLIER_Z) {
                    mLMIndices[lm_counter] = global_pos;
                    mGlobalToLocalIndexing[global_pos] = lm_counter;
                    mWhichBlockType[global_pos] = BlockType::LM;
                    lm_counter++;
                } else if (it->GetVariable().Key() == DISPLACEMENT_X || 
                    it->GetVariable().Key() == DISPLACEMENT_Y || 
                    it->GetVariable().Key() == DISPLACEMENT_Z) {
                    const IndexType node_id = it->Id();
                    NodeType::Pointer pnode = rModelPart.pGetNode(node_id);
                    if (pnode->Is(INTERFACE)) {
                        if (pnode->Is(MASTER)) {
                            mMasterIndices[master_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = master_counter;
                            mWhichBlockType[global_pos] = BlockType::MASTER;
                            master_counter++;
                            
                        } else if (pnode->Is(SLAVE)) { // TODO: Fix all the INACTIVE slave nodes
                            mSlaveIndices[slave_counter] = global_pos;
                            mGlobalToLocalIndexing[global_pos] = slave_counter;
                            mWhichBlockType[global_pos] = BlockType::SLAVE;
                            slave_counter++;
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
     * @details as A = ( KNN KNM  KNS    0  ) u
     *                 ( KMN KMM  KMS  -M^T ) u_master
     *                 ( KSN KSM  KSS   D^T ) u_slave
     *                 (  0  KLMM KLMS   0  ) LM
     * Subblocks are allocated or nor depending on the value of "NeedAllocation"
     * @param rA System matrix
     * @todo Update this for dual LM
     */
    void FillBlockMatrices (
        bool NeedAllocation, 
        SparseMatrixType& rA
        )
    {
        KRATOS_TRY
        
        // Get access to A data
        const std::size_t* index1 = rA.index1_data().begin();
        const std::size_t* index2 = rA.index2_data().begin();
        const double* values = rA.value_data().begin();

        SparseMatrixType KNN(mOtherIndices.size(), mOtherIndices.size());   /// The displacement-displacement block
        SparseMatrixType KNM(mOtherIndices.size(), mMasterIndices.size());  /// The displacement-master block
        SparseMatrixType KMN(mMasterIndices.size(), mOtherIndices.size());  /// The master-displacement block
        SparseMatrixType KNS(mOtherIndices.size(), mSlaveIndices.size());   /// The slave-displacement block
        SparseMatrixType KSN(mSlaveIndices.size(), mOtherIndices.size());   /// The displacement-slave block
        SparseMatrixType KMM(mMasterIndices.size(), mMasterIndices.size()); /// The master-master block
        SparseMatrixType KSM(mSlaveIndices.size(), mMasterIndices.size());  /// The slave-master block
        SparseMatrixType KMS(mMasterIndices.size(), mSlaveIndices.size());  /// The master-slave block
        SparseMatrixType KSS(mSlaveIndices.size(), mSlaveIndices.size());   /// The slave-slave block
        SparseMatrixType KMLM(mMasterIndices.size(), mLMIndices.size());    /// The master-LM block (this is the big block of M) 
        SparseMatrixType KLMM(mLMIndices.size(), mMasterIndices.size());    /// The LM-master block (this is the contact contribution, which gives the quadratic convergence)
        SparseMatrixType KSLM(mSlaveIndices.size(), mLMIndices.size());     /// The slave-LM block (this is the big block of D, diagonal) 
        SparseMatrixType KLMS(mLMIndices.size(), mSlaveIndices.size());     /// The LM-slave block (this is the contact contribution, which gives the quadratic convergence)
    //     SparseMatrixType mLMLM; /// The LM-LM block (This block is zero due to the LM nature, commented) 
        
        if (NeedAllocation)
            AllocateBlocks();

        // Allocate the blocks by push_back
        for (unsigned int i=0; i<rA.size1(); i++) {
            unsigned int row_begin = index1[i];
            unsigned int row_end   = index1[i+1];
            unsigned int local_row_id = mGlobalToLocalIndexing[i];

            if ( mWhichBlockType[i] == BlockType::OTHER) { //either KNN or KNM or KNS
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)       // KNN block
                        KNN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER) // KNM block
                        KNM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE)  // KNS block
                        KNS.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::MASTER) { //either KMN or KMM or KMS or KMLM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)       // KMN block
                        KMN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER) // KNMM block
                        KMM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE)  // KMS block
                        KMS.push_back ( local_row_id, local_col_id, value);
                    else                                                      // KMLM block
                        KMLM.push_back ( local_row_id, local_col_id, value);
                }
            } else if ( mWhichBlockType[i] == BlockType::SLAVE) { //either KSN or KSM or KSS or KSLM
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::OTHER)       // KSN block
                        KSN.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::MASTER) // KSMM block
                        KSM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE)  // KSS block
                        KSS.push_back ( local_row_id, local_col_id, value);
                    else                                                      // KSLM block
                        KSLM.push_back ( local_row_id, local_col_id, value);
                }
            } else { //either KLMM or KLMS
                for (unsigned int j=row_begin; j<row_end; j++) {
                    unsigned int col_index = index2[j];
                    double value = values[j];
                    unsigned int local_col_id = mGlobalToLocalIndexing[col_index];
                    if (mWhichBlockType[col_index] == BlockType::MASTER)     // KLMM block
                        KLMM.push_back ( local_row_id, local_col_id, value);
                    else if (mWhichBlockType[col_index] == BlockType::SLAVE) // KLMS block
                        KLMS.push_back ( local_row_id, local_col_id, value);
                }
            }
        }

//         // Allocate the schur complement
//         ConstructSystemMatrix(mKDispModified, mKLMModified);
// 
//         Vector diagK (mOtherIndices.size() );
//         ComputeDiagonalByLumping (K,diagK);
// 
//         // Fill the shur complement
//         CalculateShurComplement(S,K,G,D,L,diagK);

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
    LinearSolverPointerType mpSolverLMBlock;   /// The pointer to the LM linear solver
     
    std::size_t mKrylovSpaceDimension; /// The Krylov space dimension
    bool mBlocksAreAllocated;          /// The flag that indicates if the blocks are allocated
    bool mIsInitialized;               /// The flag that indicates if the solution is mIsInitialized
    
    IndexVectorType mMasterIndices;         /// The vector storing the indices of the master nodes in contact
    IndexVectorType mSlaveIndices;          /// The vector storing the indices of the slave nodes in contact
    IndexVectorType mLMIndices;             /// The vector storing the indices of the LM
    IndexVectorType mOtherIndices;          /// The vector containing the indices for other DoF
    IndexVectorType mGlobalToLocalIndexing; /// This vector stores the correspondance between the local and global
    std::vector<BlockType> mWhichBlockType; /// This vector stores the LM block belongings
    
    SparseMatrixType mKDispModified; /// The modified displacement block
    SparseMatrixType mKLMModified;   /// The modified LM block (diagonal)
    
    Vector mResidualLM;     /// The residual of the lagrange multipliers
    Vector mResidualSlave;  /// The residual of the slave displacements
    Vector mResidualMaster; /// The residual of the master displacements
    Vector mResidualDisp;   /// The residual of the rest of displacements
    
    Vector mLM;             /// The solution of the lagrange multiplies
    Vector mSlave;          /// The solution of the slave displacements
    Vector mMaster;         /// The solution of the master displacements
    Vector mDisp;           /// The solution of the rest of displacements
        
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
        mKLMModified.clear();   /// The modified LM block (diagonal)

        // We do the allocation
        const SizeType total_size = mOtherIndices.size() + mMasterIndices.size() + mSlaveIndices.size();
        mKDispModified.resize(total_size, total_size);             /// The modified displacement block
        mKLMModified.resize(mLMIndices.size(), mLMIndices.size()); /// The modified LM block (diagonal)
        
        mResidualLM.resize(mLMIndices.size() );
        mResidualDisp.resize(mOtherIndices.size() );
        mLM.resize(mLMIndices.size());
        mDisp.resize(mOtherIndices.size());
    }
    
    /**
     * @brief This method generates a plane rotation
     */
    inline void GeneratePlaneRotation (
        const double dx, 
        const double dy, 
        double& cs, 
        double& sn
        )
    {
        if (dy == 0.0) {
            cs = 1.0;
            sn = 0.0;
        } else if (dx == 0.0) {
            cs = 0.0;
            sn = 1.0;
        } else {
            const double rnorm = 1.0/sqrt (dx*dx + dy*dy);
            cs = fabs (dx) * rnorm;
            sn = cs * dy / dx;
        }
    }

    /**
     * @brief This method applies a plane rotation
     */
    inline void ApplyPlaneRotation (
        double& dx, 
        double& dy, 
        const double cs, 
        const double sn
        )
    {
        double temp  =  cs * dx + sn * dy;
        dy = cs * dy - sn * dx;
        dx = temp;
    }

    /**
     * @brief This is an auxiliar method for the GMRES solver
     */
    void Update (
        VectorType& y, 
        VectorType& x, 
        int k, 
        Matrix& h, 
        VectorType& s, 
        std::vector< VectorType >& V
        )
    {
        for (unsigned int i=0; i<s.size(); i++)
            y[i] = s[i];

        // Backsolve:
        for (int i = k; i >= 0; --i) {
            y (i) /= h (i,i);
            for (int j = i - 1; j >= 0; --j)
                y (j) -= h (j,i) * y (i);
        }
        
        //  Create new search dir
        for (int j = 0; j <= k; ++j)
            TSparseSpaceType::UnaliasedAdd (x, y[j], V[j]);   // x +=  y(j)* V[j];
    }

    /**
     * @brief This is a simple GMRES solver
     * @param A The system matrix
     * @param x The solution of the problem
     * @param b The residual of the problem
     * @param m The Krylov space dimension
     * @param MaxIterations The maximal number of iterations
     * @param Tolerance The tolerance of the resolution
     * @return 1 if converged, 0 if not
     * @todo Move a separate solver
     */
    int GMRESSolve( 
        SparseMatrixType& A,
        VectorType& x,
        const VectorType& b,
        unsigned int& m,
        unsigned int& MaxIterations,
        double& Tolerance
        )
    {
        const SizeType dim = A.size1();
        
        KRATOS_ERROR_IF( m == 0) << "The dimension of the GMRES krylov space can not be set to zero. Please change the value of m" << std::endl;
        
        if (m > MaxIterations)
            m = MaxIterations;
        
        VectorType s (m+1), sn (m+1), w (dim), r (dim), y (m+1);
        Vector  cs (m+1);
        Matrix  H (m+1, m+1);
        int restart = 0;
        
        // Preconditioner solve b and store in Minv_b
        Vector preconditioned_b (dim);

        // Apply preconditioner
        SolveBlockPreconditioner (b,preconditioned_b);
        double normb = TSparseSpaceType::TwoNorm (preconditioned_b);

        if (normb < 1e-16) //ARBITRARY SMALL NUMBER!
            normb = 1e-16;
        
        // r = b - Ax
        TSparseSpaceType::Mult (A,x,r);
        TSparseSpaceType::ScaleAndAdd (1.00, b, -1.00, r); //r = b - r
        
        // Apply preconditioner and overwrite r
        SolveBlockPreconditioner (r,r);
        const double rel_tol = Tolerance * normb;
        double beta = TSparseSpaceType::TwoNorm (r);
        if (beta <= rel_tol) { 
            Tolerance = beta / normb;
            MaxIterations = 0;
            return 0;
        }
        
        IndexType j;
        
        std::vector< VectorType > V (m+1);
        for (j = 0; j <= m; ++j)
            V[j].resize (dim,false);
        j = 1;
        
        while (j <= MaxIterations) {
            TSparseSpaceType::Assign (V[0], 1.0/beta, r); //V[0] = r /(T)beta;
            TSparseSpaceType::SetToZero (s);
            s[0] = beta;
            for (IndexType i = 0; (i < m) && (j <= MaxIterations); ++i, ++j) {
                TSparseSpaceType::Mult (A,V[i],w); //w = A*V[i];
                
                // Apply preconditioner and overwrite r
                SolveBlockPreconditioner (w,w);
                for (IndexType k = 0; k <= i; k++) {
                    H (k, i) = TSparseSpaceType::Dot (V[k], w);
                    w -= H (k, i) * V[k];
                }
                const double normw = TSparseSpaceType::TwoNorm (w);
                H (i+1, i) = normw;
                
                
                // This breakdown is a good one ...
                if (normw == 0)
                    TSparseSpaceType::Copy (V[i+1], w); //V[i+1] = w;
                else
                    TSparseSpaceType::Assign (V[i+1], 1.0/normw, w); //V[i+1] = w / normw;
                
                for (unsigned int k = 0; k < i; k++)
                    ApplyPlaneRotation (H (k,i), H (k+1,i), cs (k), sn (k) );
                
                GeneratePlaneRotation (H (i,i), H (i+1,i), cs (i), sn (i) );
                ApplyPlaneRotation (H (i,i), H (i+1,i), cs (i), sn (i) );
                ApplyPlaneRotation (s (i), s (i+1), cs (i), sn (i) );
                
                beta = std::abs(s (i+1) );
                
                KRATOS_INFO("GMRES Iteration ") <<  j << " estimated res ratio = " << beta << std::endl;

                if (beta <= rel_tol) {
                    this->Update (y, x, i, H, s, V);
                    return 0;
                }
            }
            
            this->Update (y,x, m - 1, H, s, V);
            
            // r = b - Ax
            TSparseSpaceType::Mult (A,x,r);
            TSparseSpaceType::ScaleAndAdd (1.00, b, -1.00, r); //r = b - r
            beta = TSparseSpaceType::TwoNorm (r);
            
            KRATOS_INFO("GMRES Number of Iterations ") << "at convergence = " << j << std::endl;
            
            if (beta < rel_tol) {
                return 0;
            }
            ++restart;
        }

        return 1;
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
        if (ResidualU.size() != mOtherIndices.size() )
            ResidualU.resize (mOtherIndices.size(), false);
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(ResidualU.size()); i++)
            ResidualU[i] = rTotalResidual[mOtherIndices[i]];
    }

    /**
     * @brief This function extracts from a vector which has the size of the overall r, the part that corresponds to lm-dofs
     * @param rTotalResidual The total residual of the problem 
     * @param ResidualLM The vector containing the residual relative to the LM
     */
    inline void GetLMPart (
        const VectorType& rTotalResidual, 
        VectorType& ResidualLM
        )
    {
        if (ResidualLM.size() != mLMIndices.size() )
            ResidualLM.resize (mLMIndices.size(), false);
        #pragma omp parallel for
        for (int i = 0; i<static_cast<int>(ResidualLM.size()); i++)
            ResidualLM[i] = rTotalResidual[mLMIndices[i]];
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
        for (int i = 0; i< static_cast<int>(ResidualU.size()); i++)
            rTotalResidual[mOtherIndices[i]] = ResidualU[i];
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
            rTotalResidual[mLMIndices[i]] = ResidualLM[i];
    }

    /**
     * @brief This method computes the diagonal by lumping the values
     * @param rA The matrix to compute the diagonal
     * @param diagA The resulting diagonal matrix
     */
    inline void ComputeDiagonalByLumping (
        const SparseMatrixType& rA,
        VectorType& diagA
        )
    {
        if (diagA.size() != rA.size1() )
            diagA.resize (rA.size1() );
        
        const std::size_t* index1 =rA.index1_data().begin();
        const double* values = rA.value_data().begin();

        #pragma omp parallel for
        for (int i=0; i< static_cast<int>(rA.size1()); i++) {
            unsigned int row_begin = index1[i];
            unsigned int row_end   = index1[i+1];
            double temp = 0.0;
            for (unsigned int j=row_begin; j<row_end; j++)
                temp += values[j]*values[j];

            diagA[i] = std::sqrt(temp);
        }
    }

    /**
     * @brief This method solves the block preconditioner
     * @param rTotalResidual The total residual of the problem
     * @param x The solution of the problem
     */
    void SolveBlockPreconditioner (
        const VectorType& rTotalResidual, 
        VectorType& x
        )
    {
        noalias(mLM) = ZeroVector(mOtherIndices.size());
        noalias(mDisp)  = ZeroVector(mOtherIndices.size());
        Vector u_aux (mOtherIndices.size() );
        Vector lm_aux (mLMIndices.size() );
        
        // Get diagonal of K (to be removed)
//         Vector diag_K (mOtherIndices.size() );
//         ComputeDiagonalByLumping (mKDispModified,diag_K);

        // Get the u and lm residuals
        GetUPart (rTotalResidual, mResidualDisp);
        GetLMPart (rTotalResidual, mResidualLM);

        // Solve u block
        mpSolverDispBlock->Solve (mKDispModified,mDisp,mResidualDisp);

//         // Correct LM block
//         // rlm -= D*u
//         TSparseSpaceType::Mult (mKMN,mDisp,lm_aux);
//         TSparseSpaceType::UnaliasedAdd (mResidualLM,-1.0,lm_aux);

        // Solve LM
        // LM = S⁻1*rlm
        mpSolverLMBlock->Solve (mKLMModified,mLM,mResidualLM);

//         // Correct u block
//         // u = G*lm
//         TSparseSpaceType::Mult (mKNM,mLM,u_aux);
//         #pragma omp parallel for
//         for (int i=0; i< static_cast<int>(mDisp.size()); i++)
//             mDisp[i] += u_aux[i]/diag_K[i];

        // Write back solution
        SetUPart (x,mDisp);
        SetLMPart (x,mLM);
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
            "velocity_solver" : {
                    "solver_type":"BICGSTABSolver"
            },
            "LM_solver" : {
                    "solver_type":"CGSolver"
            }
            "tolerance" : 1.0e-6,
            "MaxIterationsation" : 200,
            "gmres_krylov_space_dimension" : 100
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
