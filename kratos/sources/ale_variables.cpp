//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    msandre
//

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "includes/ale_variables.h"
#include "includes/kernel.h"

namespace Kratos
{
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(MESH_DISPLACEMENT)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(MESH_ACCELERATION)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(MESH_REACTION)
    KRATOS_CREATE_3D_VARIABLE_WITH_COMPONENTS(MESH_RHS)

    KRATOS_CREATE_VARIABLE(int, LAPLACIAN_DIRECTION)


void KratosApplication::RegisterALEVariables()
{
    KRATOS_REGISTER_3D_VARIABLE_WITH_COMPONENTS(MESH_DISPLACEMENT)
    KRATOS_REGISTER_3D_VARIABLE_WITH_COMPONENTS(MESH_ACCELERATION)
    KRATOS_REGISTER_3D_VARIABLE_WITH_COMPONENTS(MESH_REACTION)
    KRATOS_REGISTER_3D_VARIABLE_WITH_COMPONENTS(MESH_RHS)

    KRATOS_REGISTER_VARIABLE(LAPLACIAN_DIRECTION)

}

}  // namespace Kratos.
