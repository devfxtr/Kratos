// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:		 BSD License
//					 license: structural_mechanics_application/license.txt
//
//  Main authors:    Vicente Mataix Ferrándiz
//

// System includes

// External includes

// Project includes
#include "custom_elements/updated_lagrangian.h"
#include "utilities/math_utils.h"
#include "structural_mechanics_application_variables.h"
#include "custom_utilities/structural_mechanics_math_utilities.hpp"

namespace Kratos
{

//******************************CONSTRUCTOR*******************************************
/***********************************************************************************/

UpdatedLagrangian::UpdatedLagrangian( IndexType NewId, GeometryType::Pointer pGeometry )
        : BaseSolidElement( NewId, pGeometry )
{
    // NOTE: DO NOT ADD DOFS HERE!!!
}

/***********************************************************************************/
/***********************************************************************************/

UpdatedLagrangian::UpdatedLagrangian( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties )
        : BaseSolidElement( NewId, pGeometry, pProperties )
{
    // NOTE: DO NOT ADD DOFS HERE!!!
}

/***********************************************************************************/
/***********************************************************************************/

Element::Pointer UpdatedLagrangian::Create( IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties ) const
{
    return Element::Pointer( new UpdatedLagrangian( NewId, GetGeometry().Create( ThisNodes ), pProperties ) );
}

/***********************************************************************************/
/***********************************************************************************/

UpdatedLagrangian::~UpdatedLagrangian()
{
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::Initialize( )
{
    BaseSolidElement::Initialize();
    
    const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(  );
    
    const unsigned int integration_points_number = integration_points.size();
    
    if ( mDetF0.size() !=  integration_points_number)
        mDetF0.resize( integration_points_number );
    if ( mF0.size() !=  integration_points_number)
        mF0.resize( integration_points_number );
    
    const unsigned int dimension = GetGeometry().WorkingSpaceDimension();
    
    for (unsigned int point_number = 0; point_number < integration_points.size(); ++point_number) {
        mDetF0[point_number] = 1.0;  
        mF0[point_number] = IdentityMatrix(dimension);  
    }
    
    mF0Computed = false;
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::InitializeSolutionStep(ProcessInfo& rCurrentProcessInfo)
{
    BaseSolidElement::InitializeSolutionStep(rCurrentProcessInfo);
    
    mF0Computed = false;
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::FinalizeSolutionStep( ProcessInfo& rCurrentProcessInfo )
{
    // Create and initialize element variables:
    const unsigned int number_of_nodes = GetGeometry().size();
    const unsigned int dimension = GetGeometry().WorkingSpaceDimension();
    const unsigned int strain_size = mConstitutiveLawVector[0]->GetStrainSize();
        
    KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
    ConstitutiveVariables this_constitutive_variables(strain_size);

    // Create constitutive law parameters:
    ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);

    // Set constitutive law flags:
    Flags& ConstitutiveLawOptions=Values.GetOptions();
    ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, false);
    ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
    ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

    Values.SetStrainVector(this_constitutive_variables.StrainVector);
    
    // Reading integration points
    const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(  );
    
    // Displacements vector
    Vector displacements;
    GetValuesVector(displacements);
        
    // Reading integration points
    for ( unsigned int point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
        // Compute element kinematics B, F, DN_DX ...
        this->CalculateKinematicVariables(this_kinematic_variables, point_number, integration_points);
        
        // Call the constitutive law to update material variables
        mConstitutiveLawVector[point_number]->FinalizeMaterialResponse(Values, GetStressMeasure());
        
        mConstitutiveLawVector[point_number]->FinalizeSolutionStep( 
        GetProperties(),
        GetGeometry(),
        row( GetGeometry().ShapeFunctionsValues(  ), point_number ),
        rCurrentProcessInfo 
        );
        
        // Update the element internal variables
        this->UpdateHistoricalDatabase(this_kinematic_variables, point_number);
    }
    
    mF0Computed = true;
}

/***********************************************************************************/
/***********************************************************************************/

ConstitutiveLaw::StressMeasure UpdatedLagrangian::GetStressMeasure() const
{
    return ConstitutiveLaw::StressMeasure_Cauchy;
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::UpdateHistoricalDatabase(
    KinematicVariables& rThisKinematicVariables,
    const unsigned int PointNumber
    )
{
    mDetF0[PointNumber] = rThisKinematicVariables.detF;
    noalias(mF0[PointNumber]) = rThisKinematicVariables.F;
}
    
/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::CalculateAll( 
    MatrixType& rLeftHandSideMatrix,
    VectorType& rRightHandSideVector,
    ProcessInfo& rCurrentProcessInfo,
    const bool CalculateStiffnessMatrixFlag,
    const bool CalculateResidualVectorFlag 
    )
{
    KRATOS_TRY;

    const unsigned int number_of_nodes = GetGeometry().size();
    const unsigned int dimension = GetGeometry().WorkingSpaceDimension();
    const unsigned int strain_size = GetProperties().GetValue( CONSTITUTIVE_LAW )->GetStrainSize();

    KinematicVariables this_kinematic_variables(strain_size, dimension, number_of_nodes);
    ConstitutiveVariables this_constitutive_variables(strain_size);
    
    // Resizing as needed the LHS
    const unsigned int mat_size = number_of_nodes * dimension;

    if ( CalculateStiffnessMatrixFlag == true ) { // Calculation of the matrix is required
        if ( rLeftHandSideMatrix.size1() != mat_size )
            rLeftHandSideMatrix.resize( mat_size, mat_size, false );

        noalias( rLeftHandSideMatrix ) = ZeroMatrix( mat_size, mat_size ); //resetting LHS
    }

    // Resizing as needed the RHS
    if ( CalculateResidualVectorFlag == true ) { // Calculation of the matrix is required
        if ( rRightHandSideVector.size() != mat_size )
            rRightHandSideVector.resize( mat_size, false );

        rRightHandSideVector = ZeroVector( mat_size ); //resetting RHS
    }

    // Reading integration points
    const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(  );
    
    ConstitutiveLaw::Parameters Values(GetGeometry(),GetProperties(),rCurrentProcessInfo);
    
    // Set constitutive law flags:
    Flags& ConstitutiveLawOptions=Values.GetOptions();
    ConstitutiveLawOptions.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, false);
    ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
    ConstitutiveLawOptions.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);
    
    // If strain has to be computed inside of the constitutive law with PK2
    Values.SetStrainVector(this_constitutive_variables.StrainVector); //this is the input  parameter

    for ( unsigned int point_number = 0; point_number < integration_points.size(); ++point_number ) {
        // Contribution to external forces
        const Vector body_force = this->GetBodyForce(integration_points, point_number);
        
        // Compute element kinematics B, F, DN_DX ...
        this->CalculateKinematicVariables(this_kinematic_variables, point_number, integration_points);
        
        // Compute material reponse
        this->CalculateConstitutiveVariables(this_kinematic_variables, this_constitutive_variables, Values, point_number, integration_points, this->GetStressMeasure());

        // Calculating weights for integration on the reference configuration
        double int_to_reference_weight = this->GetIntegrationWeight(integration_points, point_number, this_kinematic_variables.detJ0); 

        if ( dimension == 2 && GetProperties().Has( THICKNESS )) 
            int_to_reference_weight *= this->GetProperties()[THICKNESS];

        if ( CalculateStiffnessMatrixFlag == true ) { // Calculation of the matrix is required
            // Contributions to stiffness matrix calculated on the reference config
            /* Material stiffness matrix */
            this->CalculateAndAddKm( rLeftHandSideMatrix, this_kinematic_variables.B, this_constitutive_variables.D, int_to_reference_weight );

            /* Geometric stiffness matrix */
            this->CalculateAndAddKg( rLeftHandSideMatrix, this_kinematic_variables.DN_DX, this_constitutive_variables.StressVector, int_to_reference_weight );
        }

        if ( CalculateResidualVectorFlag == true ) { // Calculation of the matrix is required
            this->CalculateAndAddResidualVector(rRightHandSideVector, this_kinematic_variables, rCurrentProcessInfo, body_force, this_constitutive_variables.StressVector, int_to_reference_weight);
        }
    }

    KRATOS_CATCH( "" )
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::CalculateKinematicVariables(
    KinematicVariables& rThisKinematicVariables,
    const unsigned int PointNumber,
    const GeometryType::IntegrationPointsArrayType& IntegrationPoints
    )
{
    const IntegrationMethod this_integration_method = this->GetGeometry().GetDefaultIntegrationMethod();
    
    // Shape functions
    rThisKinematicVariables.N = row(GetGeometry().ShapeFunctionsValues(this_integration_method), PointNumber);
    
    rThisKinematicVariables.detJ0 = this->CalculateDerivativesOnReferenceConfiguration(rThisKinematicVariables.J0, rThisKinematicVariables.InvJ0, rThisKinematicVariables.DN_DX, PointNumber, this_integration_method);
    
    // Calculating jacobian
    Matrix J, inv_J;
    rThisKinematicVariables.detJ0 = this->CalculateDerivativesOnCurrentConfiguration(J, inv_J, rThisKinematicVariables.DN_DX, PointNumber, this_integration_method);
    
    KRATOS_ERROR_IF(rThisKinematicVariables.detJ0 < 0.0) << "WARNING:: ELEMENT ID: " << this->Id() << " INVERTED. DETJ0: " << rThisKinematicVariables.detJ0 << std::endl;
    
    // Deformation gradient
    const unsigned int strain_size = (rThisKinematicVariables.B).size1();
    Matrix DF = prod( J, rThisKinematicVariables.InvJ0 );
    
    // Axisymmetric case
    if (strain_size == 4) {
        DF.resize(3, 3); // We keep the old values
        for (unsigned int index = 0; index < 1; index++) {
            DF(index, 2) = 0.0;
            DF(2, index) = 0.0;
        }

        rThisKinematicVariables.N = row(GetGeometry().ShapeFunctionsValues(this_integration_method), PointNumber);
        const double current_radius = StructuralMechanicsMathUtilities::CalculateRadius(rThisKinematicVariables.N, GetGeometry(), Current);
        const double initial_radius = StructuralMechanicsMathUtilities::CalculateRadius(rThisKinematicVariables.N, GetGeometry(), Initial);
        DF(2, 2) = current_radius/initial_radius;
    }
    
    const double detDF = MathUtils<double>::Det(DF);
    rThisKinematicVariables.detF = detDF * this->ReferenceConfigurationDeformationGradientDeterminant(PointNumber);
    noalias(rThisKinematicVariables.F) = prod(DF, this->ReferenceConfigurationDeformationGradient(PointNumber));
    
    // Calculating operator B
    this->CalculateB( rThisKinematicVariables.B, rThisKinematicVariables.DN_DX, strain_size, IntegrationPoints, PointNumber );
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::CalculateConstitutiveVariables(
    KinematicVariables& rThisKinematicVariables, 
    ConstitutiveVariables& rThisConstitutiveVariables, 
    ConstitutiveLaw::Parameters& rValues,
    const unsigned int PointNumber,
    const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
    const ConstitutiveLaw::StressMeasure ThisStressMeasure,
    const Vector Displacements
    )
{       
    // Here we essentially set the input parameters
    rValues.SetDeterminantF(rThisKinematicVariables.detF); // Assuming the determinant is computed somewhere else
    rValues.SetDeformationGradientF(rThisKinematicVariables.F); //F computed somewhere else
    
    // Here we set the space on which the results shall be written
    rValues.SetConstitutiveMatrix(rThisConstitutiveVariables.D); // Assuming the determinant is computed somewhere else
    rValues.SetStressVector(rThisConstitutiveVariables.StressVector); //F computed somewhere else
    
    // Actually do the computations in the ConstitutiveLaw    
    mConstitutiveLawVector[PointNumber]->CalculateMaterialResponse(rValues, ThisStressMeasure); //here the calculations are actually done 
}

/***********************************************************************************/
/***********************************************************************************/

double UpdatedLagrangian::CalculateDerivativesOnReferenceConfiguration(
    Matrix& J0, 
    Matrix& InvJ0, 
    Matrix& DN_DX, 
    const unsigned int PointNumber,
    IntegrationMethod ThisIntegrationMethod
    )
{
    J0.clear();
    
    double detJ0;
    
    Matrix delta_displacement;
    delta_displacement = this->CalculateDeltaDisplacement(delta_displacement);
    
    J0 = this->GetGeometry().Jacobian( J0, PointNumber, ThisIntegrationMethod, delta_displacement);
    
    const Matrix& DN_De = this->GetGeometry().ShapeFunctionsLocalGradients(ThisIntegrationMethod)[PointNumber];
    
    MathUtils<double>::InvertMatrix( J0, InvJ0, detJ0 );
    
    noalias( DN_DX ) = prod( DN_De, InvJ0);
    
    return detJ0;
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::CalculateB(
    Matrix& rB,
    const Matrix& rDN_DX,
    const unsigned int StrainSize,
    const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
    const unsigned int PointNumber
    )
{
    KRATOS_TRY
    
    const unsigned int number_of_nodes = GetGeometry().PointsNumber();
    const unsigned int dimension = GetGeometry().WorkingSpaceDimension();

    // For axisymmetric case
    Vector N;
    double Radius = 0.0;
    
    if ( StrainSize == 4 ) {
        N = row(GetGeometry().ShapeFunctionsValues(), PointNumber);
        Radius = StructuralMechanicsMathUtilities::CalculateRadius(N, GetGeometry());
    }
    
    for ( unsigned int i = 0; i < number_of_nodes; i++ ) {
        const unsigned int index = dimension * i;

        if ( StrainSize == 3 ) {
            rB( 0, index + 0 ) = rDN_DX( i, 0 );
            rB( 1, index + 1 ) = rDN_DX( i, 1 );
            rB( 2, index + 0 ) = rDN_DX( i, 1 );
            rB( 2, index + 1 ) = rDN_DX( i, 0 );
        } else if ( StrainSize == 4 ) {
            rB( 0, index + 0 ) = rDN_DX( i, 0 );
            rB( 1, index + 1 ) = rDN_DX( i, 1 );
            rB( 2, index + 0 ) = N[i]/Radius;
            rB( 3, index + 0 ) = rDN_DX( i, 1 );
            rB( 3, index + 1 ) = rDN_DX( i, 0 );
        } else {
            rB( 0, index + 0 ) = rDN_DX( i, 0 );
            rB( 1, index + 1 ) = rDN_DX( i, 1 );
            rB( 2, index + 2 ) = rDN_DX( i, 2 );

            rB( 3, index + 0 ) = rDN_DX( i, 1 );
            rB( 3, index + 1 ) = rDN_DX( i, 0 );

            rB( 4, index + 1 ) = rDN_DX( i, 2 );
            rB( 4, index + 2 ) = rDN_DX( i, 1 );

            rB( 5, index + 0 ) = rDN_DX( i, 2 );
            rB( 5, index + 2 ) = rDN_DX( i, 0 );
        }
    }

    KRATOS_CATCH( "" )
}

/***********************************************************************************/
/***********************************************************************************/

double UpdatedLagrangian::ReferenceConfigurationDeformationGradientDeterminant(const unsigned PointNumber) const
{
    if (mF0Computed == false) 
        return mDetF0[PointNumber];
    
    return 1.0;
}

/***********************************************************************************/
/***********************************************************************************/
    
Matrix UpdatedLagrangian::ReferenceConfigurationDeformationGradient(const unsigned PointNumber) const
{
    if (mF0Computed == false) 
        return mF0[PointNumber];
    
    const unsigned int dimension = this->GetGeometry().WorkingSpaceDimension();
    
    return IdentityMatrix(dimension);
}

/***********************************************************************************/
/***********************************************************************************/
    
void UpdatedLagrangian::CalculateOnIntegrationPoints(
    const Variable<double>& rVariable, 
    std::vector<double>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    )
{
    if (rVariable == REFERENCE_DEFORMATION_GRADIENT_DETERMINANT) {
        if (rValues.size() != mConstitutiveLawVector.size())
            rValues.resize(mConstitutiveLawVector.size());
        
        for ( unsigned int point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number )
            rValues[point_number] = mDetF0[point_number];
    } else {
        BaseSolidElement::CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }
}

/***********************************************************************************/
/***********************************************************************************/
    
void UpdatedLagrangian::CalculateOnIntegrationPoints(
    const Variable<Matrix>& rVariable, 
    std::vector<Matrix>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    ) 
{        
    if (rVariable == REFERENCE_DEFORMATION_GRADIENT) {
        if (rValues.size() != mConstitutiveLawVector.size())
            rValues.resize(mConstitutiveLawVector.size());
        
        for ( unsigned int point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number )
            rValues[point_number] = mF0[point_number];
    } else {
        BaseSolidElement::CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::SetValueOnIntegrationPoints(
    const Variable<double>& rVariable, 
    std::vector<double>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    )
{
    if (rVariable == REFERENCE_DEFORMATION_GRADIENT_DETERMINANT) {
        KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "Can not set REFERENCE_DEFORMATION_GRADIENT_DETERMINANT, expected size: " << mConstitutiveLawVector.size() << " current size: " << rValues.size() << std::endl;
        
        for ( unsigned int point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
            mDetF0[point_number] = rValues[point_number];
        }
    } else {
        BaseSolidElement::SetValueOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }
}

/***********************************************************************************/
/***********************************************************************************/
    
void UpdatedLagrangian::SetValueOnIntegrationPoints(
    const Variable<Matrix>& rVariable, 
    std::vector<Matrix>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    )
{
    if (rVariable == REFERENCE_DEFORMATION_GRADIENT) {
        KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "Can not set REFERENCE_DEFORMATION_GRADIENT, expected size: " << mConstitutiveLawVector.size() << " current size: " << rValues.size() << std::endl;
        
        for ( unsigned int point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number )
            mF0[point_number] = rValues[point_number];
    } else {
        BaseSolidElement::SetValueOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
    }
}

/***********************************************************************************/
/***********************************************************************************/
    
void UpdatedLagrangian::GetValueOnIntegrationPoints(
    const Variable<double>& rVariable, 
    std::vector<double>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    )
{
    this->CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
}

/***********************************************************************************/
/***********************************************************************************/
    
void UpdatedLagrangian::GetValueOnIntegrationPoints(
    const Variable<Matrix>& rVariable, 
    std::vector<Matrix>& rValues, 
    const ProcessInfo& rCurrentProcessInfo
    ) 
{
    this->CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
}

/***********************************************************************************/
/***********************************************************************************/

int UpdatedLagrangian::Check( const ProcessInfo& rCurrentProcessInfo )
{
    KRATOS_TRY

    int ier = BaseSolidElement::Check(rCurrentProcessInfo);

    return ier;

    KRATOS_CATCH( "" );
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::save( Serializer& rSerializer ) const
{
    KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseSolidElement );
    rSerializer.save("mF0Computed", mF0Computed);
    rSerializer.save("mDetF0", mDetF0);
    rSerializer.save("mF0", mF0);
}

/***********************************************************************************/
/***********************************************************************************/

void UpdatedLagrangian::load( Serializer& rSerializer )
{
    KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseSolidElement );
    rSerializer.load("mF0Computed", mF0Computed);
    rSerializer.load("mDetF0", mDetF0);
    rSerializer.load("mF0", mF0);
}

} // Namespace Kratos


