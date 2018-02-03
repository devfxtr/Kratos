from __future__ import print_function, absolute_import, division  # makes KM backward compatible with python 2.6 and 2.7
#import kratos core and applications
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

# Check that KM was imported in the main script
KM.CheckForPreviousImport()

class ContactSearchUtility:
    """This class is used in order to compute the contact search

    This class constructs the model parts containing the contact conditions and 
    initializes parameters and variables related with the contact. The class creates
    search utilities to be used to create the contact pairs

    Only the member variables listed below should be accessed directly.

    Public member variables:
    model_part -- the model part used to construct the process.
    settings -- Kratos parameters containing solver settings.
    """
    
    __normal_computation = {
        # JSON input
        "NO_DERIVATIVES_COMPUTATION": CSMA.NormalDerivativesComputation.NO_DERIVATIVES_COMPUTATION,
        "ELEMENTAL_DERIVATIVES":  CSMA.NormalDerivativesComputation.ELEMENTAL_DERIVATIVES,
        "NODAL_ELEMENTAL_DERIVATIVES": CSMA.NormalDerivativesComputation.NODAL_ELEMENTAL_DERIVATIVES
        }

    __type_search = {
        # JSON input
        "KdtreeInRadius": CSMA.SearchTreeType.KdtreeInRadius,
        "KdtreeInBox":  CSMA.SearchTreeType.KdtreeInBox,
        "Kdop": CSMA.SearchTreeType.Kdop
        }

    __check_gap = {
        # JSON input
        "NoCheck": CSMA.CheckGap.NoCheck,
        "DirectCheck":  CSMA.CheckGap.DirectCheck,
        "MappingCheck": CSMA.CheckGap.MappingCheck
        }
    
    def __init__(self, main_model_part, custom_settings): 
        
        self.main_model_part = main_model_part    
        
        ##settings string in json format
        default_settings = KM.Parameters("""
        {
            "contact_settings" :
            {
                "mortar_type"                            : "",
                "contact_model_part"                     : "Contact_Part",
                "axisymmetric"                           : false,
                "frictional_law"                         : "Coulomb",
                "assume_master_slave"                    : "",
                "normal_variation"                       : "NO_DERIVATIVES_COMPUTATION",
                "manual_ALM"                             : false,
                "stiffness_factor"                       : 1.0,
                "penalty_scale_factor"                   : 1.0,
                "use_scale_factor"                       : true,
                "penalty"                                : 0.0,
                "scale_factor"                           : 1.0e0,
                "tangent_factor"                         : 0.1,
                "integration_order"                      : 3,
                "active_check_factor"                    : 0.01,
                "max_gap_factor"                         : 1.0e-3,
                "remeshing_with_contact_bc"              : false,
                "gidio_debug"                            : false
            },
            "contact_search_settings" :
            {
                "max_number_results"                     : 1000,
                "bucket_size"                            : 4,
                "search_factor"                          : 2.0,
                "type_search"                            : "InRadius",
                "check_gap"                              : "MappingCheck",
                "database_step_update"                   : 1
            }
        }
        """)
        
        ## Overwrite the default settings with user-provided parameters
        custom_settings.ValidateAndAssignDefaults(default_settings)
        self.contact_settings = custom_settings["contact_settings"]
        self.contact_search_settings = custom_settings["contact_search_settings"]

        # A check necessary for axisymmetric cases (the domain can not be 3D)
        if (self.contact_settings["axisymmetric"].GetBool() == True) and (self.dimension == 3):
            raise NameError("3D and axisymmetric makes no sense")
        
        # Getting the normal variation flag
        self.normal_variation = self.__get_enum_flag(self.contact_settings, "normal_variation", self.__normal_computation)

        self.dimension = self.main_model_part.ProcessInfo[KM.DOMAIN_SIZE]

        self.frictional_law = self.contact_settings["frictional_law"].GetString()
        
        print("Contact search built finished")
        
    def Initialize(self, computing_model_part):          
        # The contact model part
        name_contact_model_part = self.contact_settings["contact_model_part"].GetString()
        self.contact_model_part = self.main_model_part.GetSubModelPart(name_contact_model_part)
          
        # We initialize the serach
        self.database_step = 0
        self._search_initialize(computing_model_part)
        
    def ExecuteSearch(self):
        # After predict we execute the search
        self.database_step += 1
        global_step = self.main_model_part.ProcessInfo[KM.STEP]
        
        mortar_type = self.contact_settings["mortar_type"].GetString()
        if ("Contact" in mortar_type or global_step == 1):
            if (self.database_step >= self.contact_search_settings["database_step_update"].GetInt() or global_step == 1):
                # We solve one linear step with a linear strategy if needed
                # Clear current pairs
                self.contact_search.ClearMortarConditions()
                # Update database
                self.contact_search.UpdateMortarConditions()
                #self.contact_search.CheckMortarConditions()
                    
                # Debug
                if (self.contact_settings["gidio_debug"].GetBool() == True):
                    self._debug_output(global_step, "")
                
    def ExecutePostTimeStep(self):
        mortar_type = self.contact_settings["mortar_type"].GetString()
        if ("Contact" in mortar_type):
            # In case of remeshing
            if (self.contact_settings["remeshing_with_contact_bc"].GetBool() == True):
                self._transfer_slave_to_master()
            
            # We can just reset
            modified = self.main_model_part.Is(KM.MODIFIED)
            if (modified == False and (self.database_step >= self.contact_search_settings["database_step_update"].GetInt() or self.global_step == 1)):
                self.contact_search.ClearMortarConditions()
                self.database_step = 0
        
    def _search_initialize(self, computing_model_part):
        # We compute NODAL_H that can be used in the search and some values computation
        self.find_nodal_h = KM.FindNodalHProcess(computing_model_part)
        self.find_nodal_h.Execute()
        
        # Assigning master and slave sides
        self._assign_slave_nodes()
        
        # Appending the conditions created to the self.main_model_part
        if (computing_model_part.HasSubModelPart("Contact")):
            preprocess = False
            interface_model_part = computing_model_part.GetSubModelPart("Contact")
        else:
            preprocess = True
            interface_model_part = computing_model_part.CreateSubModelPart("Contact")

        # We consider frictional contact (We use the SLIP flag because was the easiest way)
        if "ALMContactFrictional" in self.contact_settings["mortar_type"].GetString():
            computing_model_part.Set(KM.SLIP, True) 
        else:
            computing_model_part.Set(KM.SLIP, False) 
            
        # We call the process info
        process_info = self.main_model_part.ProcessInfo
            
        # We recompute the normal at each iteration (false by default)
        process_info[CSMA.CONSIDER_NORMAL_VARIATION] = self.normal_variation
        # We set the max gap factor for the gap adaptation
        max_gap_factor = self.contact_settings["max_gap_factor"].GetDouble()
        process_info[CSMA.ADAPT_PENALTY] = (max_gap_factor > 0.0)
        process_info[CSMA.MAX_GAP_FACTOR] = max_gap_factor
        process_info[CSMA.ACTIVE_CHECK_FACTOR] = self.contact_settings["active_check_factor"].GetDouble()
        
        # We set the value that scales in the tangent direction the penalty and scale parameter
        if "ALMContactFrictional" in self.contact_settings["mortar_type"].GetString():
            process_info[CSMA.TANGENT_FACTOR] = self.contact_settings["tangent_factor"].GetDouble()
        
        # Copying the properties in the contact model part
        self.contact_model_part.SetProperties(computing_model_part.GetProperties())
        
        # Setting the integration order and active check factor
        for prop in computing_model_part.GetProperties():
            prop[CSMA.INTEGRATION_ORDER_CONTACT] = self.contact_settings["integration_order"].GetInt()
            
        # We set the interface flag
        KM.VariableUtils().SetFlag(KM.INTERFACE, True, self.contact_model_part.Nodes)
        
        #If the conditions doesn't exist we create them
        if (preprocess == True):
            self._interface_preprocess(computing_model_part)
        else:
            master_slave_process = CSMA.MasterSlaveProcess(computing_model_part) 
            master_slave_process.Execute()
        
        # We initialize the contact values
        self._initialize_contact_values(computing_model_part)

        # When all conditions are simultaneously master and slave
        self._assign_slave_conditions()

        # We initialize the ALM parameters
        self._initialize_alm_parameters(computing_model_part)

        # We copy the conditions to the ContactSubModelPart
        if (preprocess == True):
            for cond in self.contact_model_part.Conditions:
                interface_model_part.AddCondition(cond)    
            del(cond)
            for node in self.contact_model_part.Nodes:
                interface_model_part.AddNode(node, 0)   
            del(node)

        # Creating the search
        self._create_main_search(computing_model_part)
        
        # We initialize the conditions    
        alm_init_var = CSMA.ALMFastInit(self.contact_model_part) 
        alm_init_var.Execute()
        
        # We initialize the search utility
        self.contact_search.CreatePointListMortar()
        self.contact_search.InitializeMortarConditions()
    
    def _interface_preprocess(self, computing_model_part):
        """ This method creates the process used to compute the contact interface

        Keyword arguments:
        self -- It signifies an instance of a class.
        computing_model_part -- The model part that contains the structural problem to be solved
        """
        
        # We create the process for creating the interface
        self.interface_preprocess = CSMA.InterfacePreprocessCondition(computing_model_part)
        
        # It should create the conditions automatically
        interface_parameters = KM.Parameters("""{"simplify_geometry": false}""")
        if (self.dimension == 2):
            self.interface_preprocess.GenerateInterfacePart2D(computing_model_part, self.contact_model_part, interface_parameters) 
        else:
            self.interface_preprocess.GenerateInterfacePart3D(computing_model_part, self.contact_model_part, interface_parameters) 
    
    def _assign_slave_conditions(self):
        """ This method initializes assigment of the slave conditions

        Keyword arguments:
        self -- It signifies an instance of a class.
        """
        
        if (self.contact_settings["assume_master_slave"].GetString() == ""):
            KM.VariableUtils().SetFlag(KM.SLAVE, True, self.contact_model_part.Conditions)

    def _assign_slave_nodes(self):
        """ This method initializes assigment of the slave nodes

        Keyword arguments:
        self -- It signifies an instance of a class.
        """
        
        if (self.contact_settings["assume_master_slave"].GetString() != ""):
            KM.VariableUtils().SetFlag(KM.SLAVE, False, self.contact_model_part.Nodes)
            KM.VariableUtils().SetFlag(KM.MASTER, True, self.contact_model_part.Nodes)
            model_part_slave = self.main_model_part.GetSubModelPart(self.contact_settings["assume_master_slave"].GetString())
            KM.VariableUtils().SetFlag(KM.SLAVE, True, model_part_slave.Nodes)
            KM.VariableUtils().SetFlag(KM.MASTER, False, model_part_slave.Nodes)
    
    def _initialize_contact_values(self, computing_model_part):
        """ This method initializes some values and variables used during contact computations

        Keyword arguments:
        self -- It signifies an instance of a class.
        computing_model_part -- The model part that contains the structural problem to be solved
        """
        
        # We consider frictional contact (We use the SLIP flag because was the easiest way)
        if "ALMContactFrictional" in self.contact_settings["mortar_type"].GetString():
            computing_model_part.Set(KM.SLIP, True)
        else:
            computing_model_part.Set(KM.SLIP, False)
            
        # We call the process info
        process_info = self.main_model_part.ProcessInfo
            
        # We recompute the normal at each iteration (false by default)
        process_info[CSMA.DISTANCE_THRESHOLD] = 1.0e24
        process_info[CSMA.CONSIDER_NORMAL_VARIATION] = self.normal_variation
        # We set the max gap factor for the gap adaptation
        max_gap_factor = self.contact_settings["max_gap_factor"].GetDouble()
        process_info[CSMA.ADAPT_PENALTY] = (max_gap_factor > 0.0)
        process_info[CSMA.MAX_GAP_FACTOR] = max_gap_factor
        
        # We set the value that scales in the tangent direction the penalty and scale parameter
        if "ALMContactFrictional" in self.contact_settings["mortar_type"].GetString():
            process_info[CSMA.TANGENT_FACTOR] = self.contact_settings["tangent_factor"].GetDouble()
        
        # Copying the properties in the contact model part
        self.contact_model_part.SetProperties(computing_model_part.GetProperties())
        
        # Setting the integration order and active check factor
        for prop in computing_model_part.GetProperties():
            prop[CSMA.INTEGRATION_ORDER_CONTACT] = self.contact_settings["integration_order"].GetInt() 
            prop[CSMA.ACTIVE_CHECK_FACTOR] = self.contact_settings["active_check_factor"].GetDouble()
    
    def _create_main_search(self, computing_model_part):
        """ This method creates the search process that will be use during contact search

        Keyword arguments:
        self -- It signifies an instance of a class.
        computing_model_part -- The model part that contains the structural problem to be solved
        """
        
        # We define the condition name to be used
        mortar_type = self.contact_settings["mortar_type"].GetString()
        axisymmetric = self.contact_settings["axisymmetric"].GetBool()
        double_formulation = ("DALM" in mortar_type)
        if "ALMContactFrictionless" in mortar_type:
            if self.normal_variation == 2:
                if axisymmetric == True:
                    condition_name = "ALMNVFrictionlessAxisymMortarContact"
                else:
                    condition_name = "ALMNVFrictionlessMortarContact"
                    if double_formulation:
                        condition_name = "D" + condition_name
            else:
                if axisymmetric == True:
                    condition_name = "ALMFrictionlessAxisymMortarContact"
                else:
                    condition_name = "ALMFrictionlessMortarContact"
                    if double_formulation:
                        condition_name = "D" + condition_name
        elif "ALMContactFrictional" in mortar_type:
            if self.normal_variation == 2:
                if axisymmetric == True:
                    condition_name = "ALMNVFrictionalAxisymMortarContact"
                else:
                    condition_name = "ALMNVFrictionalMortarContact"
            else:
                if axisymmetric == True:
                    condition_name = "ALMFrictionalAxisymMortarContact"
                else:
                    condition_name = "ALMFrictionalMortarContact"
        search_parameters = KM.Parameters("""{"condition_name": "", "final_string": "", "double_formulation" : false}""")
        search_parameters.AddValue("type_search",self.contact_search_settings["type_search"])
        search_parameters.AddValue("check_gap",self.contact_search_settings["check_gap"])
        search_parameters.AddValue("allocation_size",self.contact_search_settings["max_number_results"])
        search_parameters.AddValue("bucket_size",self.contact_search_settings["bucket_size"])
        search_parameters.AddValue("search_factor",self.contact_search_settings["search_factor"])
        search_parameters["double_formulation"].SetBool(double_formulation)
        search_parameters["condition_name"].SetString(condition_name)
        
        # We compute the number of nodes of the geometry
        number_nodes = len(computing_model_part.Conditions[1].GetNodes())
        
        # We create the search process
        if (self.dimension == 2):
            self.contact_search = CSMA.TreeContactSearch2D2N(computing_model_part, search_parameters)
        else:
            if (number_nodes == 3):
                self.contact_search = CSMA.TreeContactSearch3D3N(computing_model_part, search_parameters)
            else:
                self.contact_search = CSMA.TreeContactSearch3D4N(computing_model_part, search_parameters)
    
    def _initialize_alm_parameters(self, computing_model_part):
        """ This method initializes the ALM parameters from the process info 

        Keyword arguments:
        self -- It signifies an instance of a class.
        computing_model_part -- The model part that contains the structural problem to be solved
        """
        
        # We call the process info
        process_info = self.main_model_part.ProcessInfo
        
        if (self.contact_settings["manual_ALM"].GetBool() == False):
            # Computing the scale factors or the penalty parameters (StiffnessFactor * E_mean/h_mean)
            alm_var_parameters = KM.Parameters("""{}""")
            alm_var_parameters.AddValue("stiffness_factor",self.contact_settings["stiffness_factor"])
            alm_var_parameters.AddValue("penalty_scale_factor",self.contact_settings["penalty_scale_factor"])
            self.alm_var_process = CSMA.ALMVariablesCalculationProcess(self.contact_model_part, KM.NODAL_H, alm_var_parameters)
            self.alm_var_process.Execute()
            # We don't consider scale factor
            if (self.contact_settings["use_scale_factor"].GetBool() == False):
                process_info[KM.SCALE_FACTOR] = 1.0
        else:
            # We set the values in the process info
            process_info[KM.INITIAL_PENALTY] = self.contact_settings["penalty"].GetDouble()
            process_info[KM.SCALE_FACTOR] = self.contact_settings["scale_factor"].GetDouble()
            
        # We print the parameters considered
        print("The parameters considered finally are: ")            
        print("SCALE_FACTOR: ", "{:.2e}".format(process_info[KM.SCALE_FACTOR]))
        print("INITIAL_PENALTY: ", "{:.2e}".format(process_info[KM.INITIAL_PENALTY]))
    
    def _transfer_slave_to_master(self):
        """ This method to transfer information from the slave side to the master side

        Keyword arguments:
        self -- It signifies an instance of a class.
        """
        
        # We compute the number of nodes of the geometry
        num_nodes = len(self.contact_model_part.Conditions[1].GetNodes())
    
        # We use the search utility
        self._reset_search()
        self.contact_search.UpdateMortarConditions()
        #self.contact_search.CheckMortarConditions()
        
        map_parameters = KM.Parameters("""
        {
            "echo_level"                       : 0,
            "absolute_convergence_tolerance"   : 1.0e-9,
            "relative_convergence_tolerance"   : 1.0e-4,
            "max_number_iterations"            : 10,
            "integration_order"                : 2,
            "inverted_master_slave_pairing"    : true
        }
        """)
        
        computing_model_part = self.main_model_part.GetSubModelPart(self.computing_model_part_name)
        interface_model_part = computing_model_part.GetSubModelPart("Contact")
        if (self.dimension == 2): 
            mortar_mapping1 = KM.SimpleMortarMapperProcess2D2NDoubleNonHistorical(interface_model_part, CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE, map_parameters)
        else:
            if (num_nodes == 3): 
                mortar_mapping1 = KM.SimpleMortarMapperProcess3D3NDoubleNonHistorical(interface_model_part, CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE, map_parameters)
            else:
                mortar_mapping1 = KM.SimpleMortarMapperProcess3D4NDoubleNonHistorical(interface_model_part, CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE, map_parameters)
                    
        mortar_mapping1.Execute()
        
        # Transfering the AUGMENTED_NORMAL_CONTACT_PRESSURE to NORMAL_CONTACT_STRESS
        KM.VariableUtils().CopyScalarVar(CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE, KM.NORMAL_CONTACT_STRESS, interface_model_part.Nodes)

        self._reset_search()

    def _reset_search(self):
        """ It resets the search process.

        Keyword arguments:
        self -- It signifies an instance of a class.
        """

        self.contact_search.InvertSearch()
        self.contact_search.ResetContactOperators()
        self.contact_search.CreatePointListMortar()
        self.contact_search.InitializeMortarConditions()

    def __get_enum_flag(self, param, label, dictionary):
        """ Parse enums settings using an auxiliary dictionary of acceptable values.

        Keyword arguments:
        self -- It signifies an instance of a class.
        param -- The label to add to the postprocess file
        label -- The label used to get the string
        dictionary -- The dictionary containing the list of possible candidates
        """

        keystring = param[label].GetString()
        try:
            value = dictionary[keystring]
        except KeyError:
            msg = "{0} Error: Unknown value \"{1}\" read for parameter \"{2}\"".format(self.__class__.__name__, value, label)
            raise Exception(msg)

        return value

    def _debug_output(self, label, name):
        """ This method is used for debugging pourposes, it creates a postprocess file when called, in sucha  way that it can.

        Keyword arguments:
        self -- It signifies an instance of a class.
        label -- The label to add to the postprocess file
        name -- The name to append to the file 
        """

        output_file = "POSTSEARCH"

        gid_mode = KM.GiDPostMode.GiD_PostBinary
        singlefile = KM.MultiFileFlag.SingleFile
        deformed_mesh_flag = KM.WriteDeformedMeshFlag.WriteUndeformed
        write_conditions = KM.WriteConditionsFlag.WriteElementsOnly

        gid_io = KM.GidIO(output_file + name + "_STEP_" + str(label), gid_mode, singlefile, deformed_mesh_flag, write_conditions)

        gid_io.InitializeMesh(label)
        gid_io.WriteMesh(self.main_model_part.GetMesh())
        gid_io.FinalizeMesh()
        gid_io.InitializeResults(label, self.main_model_part.GetMesh())

        gid_io.WriteNodalFlags(KM.INTERFACE, "INTERFACE", self.main_model_part.Nodes, label)
        gid_io.WriteNodalFlags(KM.ACTIVE, "ACTIVE", self.main_model_part.Nodes, label)
        gid_io.WriteNodalFlags(KM.ISOLATED, "ISOLATED", self.main_model_part.Nodes, label)
        gid_io.WriteNodalFlags(KM.SLAVE, "SLAVE", self.main_model_part.Nodes, label)
        gid_io.WriteNodalResults(KM.NORMAL, self.main_model_part.Nodes, label, 0)
        gid_io.WriteNodalResultsNonHistorical(CSMA.AUGMENTED_NORMAL_CONTACT_PRESSURE, self.main_model_part.Nodes, label)
        gid_io.WriteNodalResultsNonHistorical(KM.NODAL_AREA, self.main_model_part.Nodes, label)
        gid_io.WriteNodalResults(KM.DISPLACEMENT, self.main_model_part.Nodes, label, 0)
        if (self.main_model_part.Nodes[1].SolutionStepsDataHas(KM.VELOCITY_X)):
            gid_io.WriteNodalResults(KM.VELOCITY, self.main_model_part.Nodes, label, 0)
            gid_io.WriteNodalResults(KM.ACCELERATION, self.main_model_part.Nodes, label, 0)
        gid_io.WriteNodalResults(KM.NORMAL_CONTACT_STRESS, self.main_model_part.Nodes, label, 0)
        gid_io.WriteNodalResults(CSMA.WEIGHTED_GAP, self.main_model_part.Nodes, label, 0)
        gid_io.WriteNodalResultsNonHistorical(CSMA.NORMAL_GAP, self.main_model_part.Nodes, label)
        gid_io.WriteNodalResultsNonHistorical(CSMA.AUXILIAR_COORDINATES, self.main_model_part.Nodes, label)
        gid_io.WriteNodalResultsNonHistorical(CSMA.DELTA_COORDINATES, self.main_model_part.Nodes, label)
        
        gid_io.FinalizeResults()
        
        #raise NameError("DEBUG") 
