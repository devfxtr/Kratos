//
//   Project Name:        Kratos
//   Last Modified by:    $Author: Anna Rehr $
//   Date:                $Date: 2017-10-13 08:56:42 $
//   email:               anna.rehr@gmx.de
//
//
//README::::look to the key word "VERSION" if you want to find all the points where you have to change something so that you can pass from a kdtree to a bin data search structure;

// #if !defined(SET_NODAL_CONTACT )
// #define  SET_NODAL_CONTACT

// /* External includes */
// #include "boost/smart_ptr.hpp"

// System includes
#include <string>
#include <iostream>
#include <stdlib.h>


// Project includes
#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "includes/condition.h"
#include "includes/mortar_classes.h"
#include "custom_utilities/contact_utilities.h"

namespace Kratos
{

/*
This class sets a boolean variable for all nodes which are in contact
*/

class SetNodalContact
{
public:

    SetNodalContact(ModelPart& mp): mThisModelPart (mp)
    {
    }

    void Execute()
    {
        std::cout<<"set bool for nodal contact"<<std::endl;

        // slave side: mark as in contact if a node has a Lagrange Multiplier unequal to zero
        ModelPart::NodesContainerType& rNodes = mThisModelPart.Nodes();
        for(ModelPart::NodesContainerType::iterator i_nodes = rNodes.begin(); i_nodes!=rNodes.end(); i_nodes++){
            if(i_nodes->Has(AUGMENTED_NORMAL_CONTACT_PRESSURE)== true){
                std::cout<<"node "<<i_nodes->Id()<<" has contact pressure "<<i_nodes->GetValue(AUGMENTED_NORMAL_CONTACT_PRESSURE)<<std::endl;
                i_nodes->SetValue(CONTACT_PRESSURE,i_nodes->GetValue(AUGMENTED_NORMAL_CONTACT_PRESSURE));
            }
        }   
        
        ModelPart::ConditionsContainerType conditions = mThisModelPart.Conditions();
        for(auto i_condition = conditions.begin(); i_condition!= conditions.end(); i_condition++)
        {
            unsigned int pair_number;
            boost::shared_ptr<ConditionMap>& all_conditions_maps = i_condition->GetValue( MAPPING_PAIRS );
            pair_number = all_conditions_maps->size();
            std::cout<<"condition"<<i_condition->Id()<<" has "<<pair_number<<" master elements."<<std::endl;
            
            for (auto i_master = all_conditions_maps->begin(); i_master != all_conditions_maps->end(); i_master++){
                if (i_master->second == true){
                    for(int i=0; i < i_master->first->GetGeometry().size(); i++){
                        
                        array_1d<double,3> i_nodes= i_master->first->GetGeometry()[i].Coordinates();
                        // Projection of the master node to the slave node
                        //ContactUtilities::FastProjectDirection()
                    
                    }
                }
            }

        }
    }

private:
    ModelPart mThisModelPart;
};



}  // namespace Kratos.

// #endif // KRATOS_PROJECTION  defined 

