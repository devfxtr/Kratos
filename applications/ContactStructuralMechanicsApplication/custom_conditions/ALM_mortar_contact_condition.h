// KRATOS  ___|  |       |       |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//           | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License: BSD License
//   license: StructuralMechanicsApplication/license.txt
//
//  Main authors:  Vicente Mataix Ferrandiz
//

#if !defined(KRATOS_ALM_MORTAR_CONTACT_CONDITION_H_INCLUDED )
#define  KRATOS_ALM_MORTAR_CONTACT_CONDITION_H_INCLUDED

// System includes

// External includes

// Project includes
#include "contact_structural_mechanics_application_variables.h"
#include "custom_conditions/paired_condition.h"
#include "utilities/math_utils.h"
#include "includes/kratos_flags.h"
#include "includes/checks.h"
#include "includes/mortar_classes.h"

/* Utilities */
#include "utilities/exact_mortar_segmentation_utility.h"
#include "custom_utilities/derivatives_utilities.h"
#include "custom_utilities/logging_settings.hpp"

/* Geometries */
#include "geometries/line_2d_2.h"
#include "geometries/triangle_3d_3.h"

namespace Kratos 
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{
    
    typedef Point                                     PointType;
    typedef Node<3>                                    NodeType;
    typedef Geometry<NodeType>                     GeometryType;
    typedef Geometry<PointType>               GeometryPointType;
    ///Type definition for integration methods
    typedef GeometryData::IntegrationMethod   IntegrationMethod;
    
///@}
///@name  Enum's
///@{
    
    /** 
     * @brief We use this to differentiate between cases of friction
     */
    enum class FrictionalCase {FRICTIONLESS = 0, FRICTIONLESS_COMPONENTS = 1, FRICTIONAL = 2 };
    
///@}
///@name  Functions
///@{
    
///@}
///@name Kratos Classes
///@{
    
/** @brief AugmentedLagrangianMethodMortarContactCondition
 * @details This is a contact condition which employes the mortar method with dual lagrange multiplier 
 * The method has been taken from the Alexander Popps thesis:
 * Popp, Alexander: Mortar Methods for Computational Contact Mechanics and General Interface Problems, Technische Universität München, jul 2012
 */
template< unsigned int TDim, unsigned int TNumNodes, FrictionalCase TFrictional, bool TNormalVariation>
class AugmentedLagrangianMethodMortarContactCondition
    : public PairedCondition 
{
public:
    ///@name Type Definitions
    ///@{
        
    /// Counted pointer of AugmentedLagrangianMethodMortarContactCondition
    KRATOS_CLASS_POINTER_DEFINITION( AugmentedLagrangianMethodMortarContactCondition );

    typedef PairedCondition                                                                               BaseType;
    
    typedef typename BaseType::VectorType                                                               VectorType;

    typedef typename BaseType::MatrixType                                                               MatrixType;

    typedef typename BaseType::IndexType                                                                 IndexType;

    typedef typename BaseType::GeometryType::Pointer                                           GeometryPointerType;

    typedef typename BaseType::NodesArrayType                                                       NodesArrayType;

    typedef typename BaseType::PropertiesType::Pointer                                       PropertiesPointerType;
    
    typedef typename std::conditional<TNumNodes == 2, PointBelongsLine2D2N, typename std::conditional<TNumNodes == 3, PointBelongsTriangle3D3N, PointBelongsQuadrilateral3D4N>::type>::type BelongType;
    
    typedef PointBelong<TNumNodes>                                                                 PointBelongType;
    
    typedef Geometry<PointBelongType>                                                      GeometryPointBelongType;
    
    typedef array_1d<PointBelongType,TDim>                                                      ConditionArrayType;
    
    typedef typename std::vector<ConditionArrayType>                                        ConditionArrayListType;
    
    typedef Line2D2<PointType>                                                                            LineType;
    
    typedef Triangle3D3<PointType>                                                                    TriangleType;
    
    typedef typename std::conditional<TDim == 2, LineType, TriangleType >::type                  DecompositionType;
    
    typedef typename std::conditional<TFrictional == FrictionalCase::FRICTIONAL, DerivativeDataFrictional<TDim, TNumNodes, TNormalVariation>, DerivativeData<TDim, TNumNodes, TNormalVariation> >::type DerivativeDataType;
    
    static constexpr unsigned int MatrixSize = (TFrictional == FrictionalCase::FRICTIONLESS) ? TDim * (TNumNodes + TNumNodes) + TNumNodes : TDim * (TNumNodes + TNumNodes + TNumNodes);
    
    static constexpr unsigned int IsFrictional  = (TFrictional == FrictionalCase::FRICTIONAL) ? true: false;
    
    typedef MortarKinematicVariablesWithDerivatives<TDim, TNumNodes>                              GeneralVariables;
    
    typedef DualLagrangeMultiplierOperatorsWithDerivatives<TDim, TNumNodes, IsFrictional, TNormalVariation> AeData;
    
    typedef MortarOperatorWithDerivatives<TDim, TNumNodes, IsFrictional, TNormalVariation> MortarConditionMatrices;
    
    typedef ExactMortarIntegrationUtility<TDim, TNumNodes, true>                                IntegrationUtility;
    
    typedef DerivativesUtilities<TDim, TNumNodes, IsFrictional, TNormalVariation>         DerivativesUtilitiesType;
         
    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor
    AugmentedLagrangianMethodMortarContactCondition()
        : PairedCondition(),
          mIntegrationOrder(2)
    {}
    
    // Constructor 1
    AugmentedLagrangianMethodMortarContactCondition(
        IndexType NewId, 
        GeometryType::Pointer pGeometry
        ) :PairedCondition(NewId, pGeometry),
           mIntegrationOrder(2)
    {}
    
    // Constructor 2
    AugmentedLagrangianMethodMortarContactCondition(
        IndexType NewId, 
        GeometryType::Pointer pGeometry, 
        PropertiesType::Pointer pProperties
        ) :PairedCondition( NewId, pGeometry, pProperties ),
           mIntegrationOrder(2)
    {}
    
    // Constructor 3
    AugmentedLagrangianMethodMortarContactCondition(
        IndexType NewId, 
        GeometryType::Pointer pGeometry, 
        PropertiesType::Pointer pProperties, 
        GeometryType::Pointer pMasterGeometry
        )
        :PairedCondition( NewId, pGeometry, pProperties, pMasterGeometry),
         mIntegrationOrder(2)
    {}

    ///Copy constructor
    AugmentedLagrangianMethodMortarContactCondition( AugmentedLagrangianMethodMortarContactCondition const& rOther){}

    /// Destructor.
    ~AugmentedLagrangianMethodMortarContactCondition() override;

    /**
     * Flags related to the element computation
     */

    KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_RHS_VECTOR );
    KRATOS_DEFINE_LOCAL_FLAG( COMPUTE_LHS_MATRIX );

    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{
    
   /**
    * Called at the beginning of each solution step
    */
    void Initialize() override;

   /**
    * Called at the beginning of each solution step
    * @param rCurrentProcessInfo the current process info instance
    */
    void InitializeSolutionStep(ProcessInfo& rCurrentProcessInfo) override;

   /**
    * Called at the beginning of each iteration
    * @param rCurrentProcessInfo the current process info instance
    */
    void InitializeNonLinearIteration(ProcessInfo& rCurrentProcessInfo) override;

    /**
    * Called at the ending of each solution step
    * @param rCurrentProcessInfo the current process info instance
    */
    void FinalizeSolutionStep(ProcessInfo& rCurrentProcessInfo) override;
    
   /**
    * Called at the end of each iteration
    * @param rCurrentProcessInfo the current process info instance
    */
    void FinalizeNonLinearIteration(ProcessInfo& rCurrentProcessInfo) override;

    /**
    * This method computes the mass matrix
    * @param rMassMatrix The mass matrix to be computed
    * @param rCurrentProcessInfo the current process info instance
    */
    
    void CalculateMassMatrix( 
        MatrixType& rMassMatrix, 
        ProcessInfo& rCurrentProcessInfo
        ) override;
    
    /**
     * Initialize Damping Matrix
     * @param rDampingMatrix The damping matrix to be computed
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void CalculateDampingMatrix( 
        MatrixType& rDampingMatrix,
        ProcessInfo& rCurrentProcessInfo
        ) override;
    
    /**
     * Creates a new element pointer from an arry of nodes
     * @param NewId the ID of the new element
     * @param rThisNodes the nodes of the new element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    
    Condition::Pointer Create( 
        IndexType NewId,
        NodesArrayType const& rThisNodes,
        PropertiesType::Pointer pProperties 
        ) const override;
    
    /**
     * Creates a new element pointer from an existing geometry
     * @param NewId the ID of the new element
     * @param pGeom the  geometry taken to create the condition
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    
    Condition::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties
        ) const override;
        
    /**
     * Creates a new element pointer from an existing geometry
     * @param NewId the ID of the new element
     * @param pGeom the  geometry taken to create the condition
     * @param pProperties the properties assigned to the new element
     * @param pMasterGeom the paired geometry
     * @return a Pointer to the new element
     */
    Condition::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties,
        GeometryType::Pointer pMasterGeom
        ) const override;
       
    /**
     * this is called during the assembling process in order
     * to calculate the condition contribution in explicit calculation.
     * NodalData is modified Inside the function, so the
     * The "AddEXplicit" FUNCTIONS THE ONLY FUNCTIONS IN WHICH A CONDITION
     * IS ALLOWED TO WRITE ON ITS NODES.
     * the caller is expected to ensure thread safety hence
     * SET/UNSETLOCK MUST BE PERFORMED IN THE STRATEGY BEFORE CALLING THIS FUNCTION
     * @param rCurrentProcessInfo the current process info instance
     */
    void AddExplicitContribution(ProcessInfo& rCurrentProcessInfo) override;
        
    /******************************************************************/
    /********** AUXILLIARY METHODS FOR GENERAL CALCULATIONS ***********/
    /******************************************************************/

    /**
     * Sets on rResult the ID's of the element degrees of freedom
     * @param rResult The result vector with the ID's of the DOF
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void EquationIdVector( 
        EquationIdVectorType& rResult,
        ProcessInfo& rCurrentProcessInfo
        ) override;

    /**
     * Sets on ConditionalDofList the degrees of freedom of the considered element geometry
     * @param rConditionalDofList The list of DoF
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void GetDofList( 
        DofsVectorType& rConditionalDofList,
        ProcessInfo& rCurrentProcessInfo 
        ) override;

    /**
     * Get on rVariable a double Value
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rValues The values of interest (doubles)
     */
    
    void GetValueOnIntegrationPoints( 
        const Variable<double>& rVariable,
        std::vector<double>& rValues,
        const ProcessInfo& rCurrentProcessInfo 
        ) override;
    
    /**
     * Get on rVariable a array_1d Value
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rValues The values of interest (array_1d)
     */
    
    void GetValueOnIntegrationPoints( 
        const Variable<array_1d<double, 3 > >& rVariable,
        std::vector<array_1d<double, 3 > >& rValues,
        const ProcessInfo& rCurrentProcessInfo
        ) override;
    
    /**
     * Get on rVariable a Vector Value
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rValues The values of interest (vector)
     */
    
    void GetValueOnIntegrationPoints( 
        const Variable<Vector>& rVariable,
        std::vector<Vector>& rValues,
        const ProcessInfo& rCurrentProcessInfo
        ) override;

    /**
     * Calculate a double Variable
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rOutput The values of interest (doubles)
     */
    
    void CalculateOnIntegrationPoints( 
        const Variable<double>& rVariable,
        std::vector<double>& rOutput,
        const ProcessInfo& rCurrentProcessInfo 
        ) override;
    
    /**
     * Calculate a array_1d Variable
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rOutput The values of interest (array_1d)
     */
    
    void CalculateOnIntegrationPoints( 
        const Variable<array_1d<double, 3 > >& rVariable,
        std::vector< array_1d<double, 3 > >& rOutput,
        const ProcessInfo& rCurrentProcessInfo
        ) override;
    
    /**
     * Calculate a Vector Variable
     * @param rVariable Internal values
     * @param rCurrentProcessInfo The current process information
     * @param rOutput The values of interest (vector)
     */
    
    void CalculateOnIntegrationPoints( 
        const Variable<Vector>& rVariable, 
        std::vector<Vector>& rOutput, 
        const ProcessInfo& rCurrentProcessInfo
        ) override;

    /**
     * This function provides the place to perform checks on the completeness of the input.
     * It is designed to be called only once (or anyway, not often) typically at the beginning
     * of the calculations, so to verify that nothing is missing from the input
     * or that no common error is found.
     * @param rCurrentProcessInfo The current process information
     */
    int Check( const ProcessInfo& rCurrentProcessInfo ) override;
        
    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

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
    
    Flags  mCalculationFlags;                            // Calculation flags
    
    IntegrationMethod mThisIntegrationMethod;            // Integration order of the element
    
    unsigned int mIntegrationOrder;                      // The integration order to consider
    
    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    /******************************************************************/
    /*********************** COMPUTING  METHODS ***********************/
    /******************************************************************/

    /**
     * This is called during the assembling process in order
     * to calculate all condition contributions to the global system
     * matrix and the right hand side
     * @param rLeftHandSideMatrix the condition left hand side matrix
     * @param rRightHandSideVector the condition right hand side
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void CalculateLocalSystem( 
        MatrixType& rLeftHandSideMatrix,
        VectorType& rRightHandSideVector,
        ProcessInfo& rCurrentProcessInfo 
        ) override;

    /**
     * This is called during the assembling process in order
     * to calculate the condition right hand side vector only
     * @param rRightHandSideVector the condition right hand side vector
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void CalculateRightHandSide(
        VectorType& rRightHandSideVector,
        ProcessInfo& rCurrentProcessInfo 
        ) override;

    /**
     * This is called during the assembling process in order
     * to calculate the condition left hand side matrix only
     * @param rLeftHandSideMatrix the condition left hand side matrix
     * @param rCurrentProcessInfo the current process info instance
     */
    
    void CalculateLeftHandSide( 
        MatrixType& rLeftHandSideMatrix,
        ProcessInfo& rCurrentProcessInfo 
        ) override;

    /**
     * Calculates the condition contribution
     */
    
    void CalculateConditionSystem( 
        MatrixType& rLeftHandSideMatrix,
        VectorType& rRightHandSideVector,
        const ProcessInfo& CurrentProcessInfo 
        );
    
    /**
     * Calculate condition kinematics (shape functions, jacobians, ...)
     */
    
    void CalculateKinematics( 
        GeneralVariables& rVariables,
        const DerivativeDataType& rDerivativeData,
        const array_1d<double, 3>& NormalMaster,
        const PointType& LocalPointDecomp,
        const PointType& LocalPointParent,
        GeometryPointType& GeometryDecomp,
        const bool DualLM = true
        );

    /********************************************************************************/
    /**************** METHODS TO CALCULATE MORTAR CONDITION MATRICES ****************/
    /********************************************************************************/

    /**
     * Calculates the local contibution of the LHS
     * @param rLocalLHS The local LHS to compute
     * @param rMortarConditionMatrices The mortar operators to be considered
     * @param rDerivativeData The class containing all the derivatives uses to compute the jacobian 
     * @param rActiveInactive The integer that is used to identify which case is the currectly computed
     */
    
    virtual void CalculateLocalLHS(
        Matrix& rLocalLHS, 
        const MortarConditionMatrices& rMortarConditionMatrices,
        const DerivativeDataType& rDerivativeData,
        const unsigned int rActiveInactive
        );
    
    /**
     * Calculates the local contibution of the RHS
     * @param rLocalRHS The local RHS to compute
     * @param rMortarConditionMatrices The mortar operators to be considered
     * @param rDerivativeData The class containing all the derivatives uses to compute the jacobian 
     * @param rActiveInactive The integer that is used to identify which case is the currectly computed
     */
    
    virtual void CalculateLocalRHS(
        Vector& rLocalRHS,
        const MortarConditionMatrices& rMortarConditionMatrices,
        const DerivativeDataType& rDerivativeData,
        const unsigned int rActiveInactive
        );
    
    /***********************************************************************************/
    /**************** AUXILLIARY METHODS FOR CONDITION LHS CONTRIBUTION ****************/
    /***********************************************************************************/

    /**
     * Calculates the values of the shape functions for the master element
     */
    
    void MasterShapeFunctionValue(
        GeneralVariables& rVariables,
        const array_1d<double, 3>& NormalMaster,
        const PointType& LocalPoint
        );
    
    /******************************************************************/
    /********** AUXILLIARY METHODS FOR GENERAL CALCULATIONS ***********/
    /******************************************************************/
    
    /**
     * Returns a value depending of the active/inactive set
     * @param CurrentGeometry The geometry containing the nodes that are needed to be checked as active or inactive
     * @return The integer that can be used to identify the case to compute 
     */
    
    virtual unsigned int GetActiveInactiveValue(GeometryType& CurrentGeometry) const
    {
        KRATOS_ERROR << "You are calling to the base class method GetActiveInactiveValue, you are evil, and your seed must be eradicated from the face of the earth" << std::endl;
        
        return 0;
    }

    /**
     * It checks if the element is isolated or not
     * @param DeltaTime The increment of time in each time step
     * @param HalfJump If the increment of time considered is just half or the whole time step
     */
    
    bool CheckIsolatedElement(
        const double DeltaTime, 
        const bool HalfJump = true
        );
    
    /**
     * It returns theintegration method considered
     */
    
    IntegrationMethod GetIntegrationMethod() override
    {        
        // Setting the auxiliar integration points
        switch (mIntegrationOrder) {
        case 1: return GeometryData::GI_GAUSS_1;
        case 2: return GeometryData::GI_GAUSS_2;
        case 3: return GeometryData::GI_GAUSS_3;
        case 4: return GeometryData::GI_GAUSS_4;
        case 5: return GeometryData::GI_GAUSS_5;
        default: return GeometryData::GI_GAUSS_2;
        }
    }
    
    /**
     * This functions computes the integration weight to consider
     * @param rVariables The kinematic variables
     */
    
    virtual double GetAxisymmetricCoefficient(const GeneralVariables& rVariables) const;
    
    /**
     * This method just resizes the LHS matrix
     * @param rLeftHandSideMatrix The LHS matrix
     */
    virtual void ResizeLHS(MatrixType& rLeftHandSideMatrix);
    
    /**
     * This method just resizes the RHS vector
     * @param rRightHandSideVector The RHS vector
     */
    virtual void ResizeRHS(VectorType& rRightHandSideVector);
    
    /**
     * This method just sets as zero the LHS matrix
     * @param rLeftHandSideMatrix The LHS matrix
     */
    virtual void ZeroLHS(MatrixType& rLeftHandSideMatrix);
    
    /**
     *This method just sets as zero the RHS vector
     * @param rRightHandSideVector The RHS vector
     */
    virtual void ZeroRHS(VectorType& rRightHandSideVector);
    
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

    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{

    // Serialization 
    
    friend class Serializer;
    
    void save(Serializer& rSerializer) const override
    {
        KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, PairedCondition );
        rSerializer.save("CalculationFlags", mCalculationFlags);
        rSerializer.save("IntegrationOrder", mIntegrationOrder);
    }

    void load(Serializer& rSerializer) override
    {
        KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, PairedCondition );
        rSerializer.load("CalculationFlags", mCalculationFlags);
        rSerializer.load("IntegrationOrder", mIntegrationOrder);
    }

    ///@}

}; // Class AugmentedLagrangianMethodMortarContactCondition

///@}

///@name Type Definitions
///@{

///@}
///@name Input and output
///@{

///@}

}// namespace Kratos.

#endif // KRATOS_ALM_MORTAR_CONTACT_CONDITION_H_INCLUDED  defined 
