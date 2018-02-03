from __future__ import print_function, absolute_import, division  # makes KM backward compatible with python 2.6 and 2.7
#import kratos core and applications
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

# Check that KM was imported in the main script
KM.CheckForPreviousImport()

# Import the implicit solver (the explicit one is derived from it)
import structural_mechanics_static_solver

def CreateSolver(main_model_part, custom_settings):
    return StaticMechanicalSolver(main_model_part, custom_settings)

class StaticMechanicalSolver(structural_mechanics_static_solver.StaticMechanicalSolver):
    """The structural mechanics contact static solver.

    This class creates the mechanical solvers for contact static analysis. It currently
    supports line search, linear, arc-length, form-finding and Newton-Raphson
    strategies.

    Public member variables:
    arc_length_settings -- settings for the arc length method.

    See structural_mechanics_solver.py for more information.
    """
    def __init__(self, main_model_part, custom_settings):

        self.main_model_part = main_model_part

        # Settings string in json format
        contact_related_settings = KM.Parameters("""
        {
            "contact_strategy_settings" :
            {
                "mortar_type"                            : "",
                "condn_convergence_criterion"            : false,
                "fancy_convergence_criterion"            : true,
                "print_convergence_criterion"            : false,
                "ensure_contact"                         : false,
                "gidio_debug"                            : false,
                "adaptative_strategy"                    : false,
                "split_factor"                           : 10.0,
                "max_number_splits"                      : 3,
                "contact_displacement_relative_tolerance": 1.0e-4,
                "contact_displacement_absolute_tolerance": 1.0e-9,
                "contact_residual_relative_tolerance"    : 1.0e-4,
                "contact_residual_absolute_tolerance"    : 1.0e-9
            },
            "contact_strategy_settings" :
            {
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
                "remeshing_with_contact_bc"              : false
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
        self.settings = custom_settings
        self.validate_and_transfer_matching_settings(self.settings, contact_related_settings)
        self.contact_strategy_settings = contact_related_settings["contact_strategy_settings"]
        
        # Construct the base solver.
        super().__init__(self.main_model_part, self.settings)
        
        # Setting default configurations true by default
        if (self.settings["clear_storage"].GetBool() == False):
            print("WARNING:: Storage must be cleared each step. Switching to True")
            self.settings["clear_storage"].SetBool(True)
        if (self.settings["reform_dofs_at_each_step"].GetBool() == False):
            print("WARNING:: DoF must be reformed each time step. Switching to True")
            self.settings["reform_dofs_at_each_step"].SetBool(True)
        if (self.settings["block_builder"].GetBool() == False):
            print("WARNING:: EliminationBuilderAndSolver can not used with the current implementation. Switching to BlockBuilderAndSolver")
            self.settings["block_builder"].SetBool(True)

        # Initialize the processes list
        self.processes_list = None
        
        # Create an auxiliary Kratos parameters object for the search utility settings.
        search_utility_params = KM.Parameters("{}")
        search_utility_params.AddValue("contact_strategy_settings",contact_related_settings["contact_strategy_settings"])
        search_utility_params.AddValue("contact_search_settings",contact_related_settings["contact_search_settings"])
        search_utility_params["contact_strategy_settings"].AddValue("mortar_type",self.contact_strategy_settings["mortar_type"])
        search_utility_params["contact_strategy_settings"].AddValue("gidio_debug",self.contact_strategy_settings["gidio_debug"])
        import contact_search_utility
        self.contact_search_utility = contact_search_utility.ContactSearchUtility(self.main_model_part, search_utility_params)

        print("Construction of ContactMechanicalSolver finished")

    def AddVariables(self):

        super().AddVariables()

        mortar_type = self.contact_strategy_settings["mortar_type"].GetString()
        if  mortar_type != "":
            self.main_model_part.AddNodalSolutionStepVariable(KM.NORMAL)  # Add normal
            self.main_model_part.AddNodalSolutionStepVariable(KM.NODAL_H) # Add nodal size variable
            if  "ALMContactFrictionless" in mortar_type:
                self.main_model_part.AddNodalSolutionStepVariable(KM.NORMAL_CONTACT_STRESS)       # Add normal contact stress
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_GAP)              # Add normal contact gap
            elif "ALMContactFrictional" in mortar_type: 
                self.main_model_part.AddNodalSolutionStepVariable(KM.VECTOR_LAGRANGE_MULTIPLIER)  # Add normal contact stress 
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_GAP)              # Add normal contact gap 
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_SLIP)             # Add normal contact gap 
            elif "ScalarMeshTying" in mortar_type:
                self.main_model_part.AddNodalSolutionStepVariable(KM.SCALAR_LAGRANGE_MULTIPLIER)  # Add scalar LM
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_SCALAR_RESIDUAL)  # Add scalar LM residual
            elif "ComponentsMeshTying" in mortar_type:
                self.main_model_part.AddNodalSolutionStepVariable(KM.VECTOR_LAGRANGE_MULTIPLIER)  # Add vector LM
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_VECTOR_RESIDUAL)  # Add vector LM residual
                
        print("::[Contact Mechanical Solver]:: Variables ADDED")
    
    def AddDofs(self):

        super().AddDofs()
        
        mortar_type = self.contact_strategy_settings["mortar_type"].GetString()
        if ("ALMContactFrictionless" in mortar_type):
            KM.VariableUtils().AddDof(KM.NORMAL_CONTACT_STRESS, CSMA.WEIGHTED_GAP, self.main_model_part)
        elif ("ALMContactFrictional" in mortar_type): 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_X, self.main_model_part) 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Y, self.main_model_part) 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Z, self.main_model_part) 
        elif ("ScalarMeshTying" in mortar_type):
            KM.VariableUtils().AddDof(KM.SCALAR_LAGRANGE_MULTIPLIER, CSMA.WEIGHTED_SCALAR_RESIDUAL, self.main_model_part)
        elif ("ComponentsMeshTying" in mortar_type):
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_X, CSMA.WEIGHTED_VECTOR_RESIDUAL_X, self.main_model_part)
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Y, CSMA.WEIGHTED_VECTOR_RESIDUAL_Y, self.main_model_part)
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Z, CSMA.WEIGHTED_VECTOR_RESIDUAL_Z, self.main_model_part)

        print("::[Contact Mechanical Solver]:: DOF's ADDED")
    
    def Initialize(self):
        super().Initialize() # The mechanical solver is created here.
          
        # Local matrices and vectors (for manual predict and update)
        self.A = KM.CompressedMatrix()
        self.Dx = KM.Vector()
        self.b = KM.Vector()
          
        # The contact search
        computing_model_part = self.GetComputingModelPart()
        self.contact_search_utility.Initialize(computing_model_part)
    
    def Solve(self):
        if self.settings["clear_storage"].GetBool():
            self.Clear()
        
        mechanical_solver = self.get_mechanical_solver()
        
        # We predict before searching 
        self.mechanical_scheme.Predict(self.main_model_part, self.builder_and_solver.GetDofSet(), self.A, self.Dx, self.b)
        mechanical_solver.MoveMesh()
        
        # After predict we execute the search
        self.contact_search_utility.ExecuteSearch()
            
        # The steps of the solve are Initialize(), InitializeSolutionStep(), Predict(), SolveSolutionStep(), FinalizeSolutionStep()
        mechanical_solver.Initialize()
        mechanical_solver.InitializeSolutionStep()
        #mechanical_solver.Predict() # Predict is performed before search
        # We solve the problem
        mechanical_solver.SolveSolutionStep()
        mechanical_solver.FinalizeSolutionStep()
        
        # The post time step executions in the ContactSearchUtility
        self.contact_search_utility.ExecutePostTimeStep()
        
    def AddProcessesList(self, processes_list):
        self.processes_list = CSMA.ProcessFactoryUtility(processes_list)
        
    def _create_convergence_criterion(self):
        # Create an auxiliary Kratos parameters object to store the convergence settings.
        conv_params = KM.Parameters("{}")
        conv_params.AddValue("convergence_criterion",self.settings["convergence_criterion"])
        conv_params.AddValue("rotation_dofs",self.settings["rotation_dofs"])
        conv_params.AddValue("echo_level",self.settings["echo_level"])
        conv_params.AddValue("displacement_relative_tolerance",self.settings["displacement_relative_tolerance"])
        conv_params.AddValue("displacement_absolute_tolerance",self.settings["displacement_absolute_tolerance"])
        conv_params.AddValue("residual_relative_tolerance",self.settings["residual_relative_tolerance"])
        conv_params.AddValue("residual_absolute_tolerance",self.settings["residual_absolute_tolerance"])
        conv_params.AddValue("contact_displacement_relative_tolerance",self.contact_strategy_settings["contact_displacement_relative_tolerance"])
        conv_params.AddValue("contact_displacement_absolute_tolerance",self.contact_strategy_settings["contact_displacement_absolute_tolerance"])
        conv_params.AddValue("contact_residual_relative_tolerance",self.contact_strategy_settings["contact_residual_relative_tolerance"])
        conv_params.AddValue("contact_residual_absolute_tolerance",self.contact_strategy_settings["contact_residual_absolute_tolerance"])
        conv_params.AddValue("mortar_type",self.contact_strategy_settings["mortar_type"])
        conv_params.AddValue("condn_convergence_criterion",self.contact_strategy_settings["condn_convergence_criterion"])
        conv_params.AddValue("fancy_convergence_criterion",self.contact_strategy_settings["fancy_convergence_criterion"])
        conv_params.AddValue("print_convergence_criterion",self.contact_strategy_settings["print_convergence_criterion"])
        conv_params.AddValue("ensure_contact",self.contact_strategy_settings["ensure_contact"])
        conv_params.AddValue("gidio_debug",self.contact_strategy_settings["gidio_debug"])
        import contact_convergence_criteria_factory
        convergence_criterion = contact_convergence_criteria_factory.convergence_criterion(conv_params)
        return convergence_criterion.mechanical_convergence_criterion
        
    def _create_builder_and_solver(self):
        if  self.contact_strategy_settings["mortar_type"].GetString() != "":
            linear_solver = self.get_linear_solver()
            if self.settings["block_builder"].GetBool():
                if self.settings["multi_point_constraints_used"].GetBool():
                    raise Exception("MPCs not compatible with contact")
                else:
                    builder_and_solver = CSMA.ContactResidualBasedBlockBuilderAndSolver(linear_solver)
            else:
                raise Exception("Contact not compatible with EliminationBuilderAndSolver")
        else:
            builder_and_solver = super()._create_builder_and_solver()
            
        return builder_and_solver
        
    def _create_mechanical_solver(self):
        if  self.contact_strategy_settings["mortar_type"].GetString() != "":
            if self.settings["analysis_type"].GetString() == "linear":
                mechanical_solver = self._create_linear_strategy()
            else:
                if(self.settings["line_search"].GetBool()):
                    mechanical_solver = self._create_contact_line_search_strategy()
                else:
                    mechanical_solver = self._create_contact_newton_raphson_strategy()
        else:
            mechanical_solver = super()._create_mechanical_solver()
                    
        return mechanical_solver
    
    def _create_contact_line_search_strategy(self):
        computing_model_part = self.GetComputingModelPart()
        self.mechanical_scheme = self.get_solution_scheme()
        self.linear_solver = self.get_linear_solver()
        self.mechanical_convergence_criterion = self.get_convergence_criterion()
        self.builder_and_solver = self.get_builder_and_solver()
        newton_parameters = KM.Parameters("""{}""")
        return CSMA.LineSearchContactStrategy(computing_model_part, 
                                                self.mechanical_scheme, 
                                                self.linear_solver, 
                                                self.mechanical_convergence_criterion, 
                                                self.builder_and_solver, 
                                                self.settings["max_iteration"].GetInt(), 
                                                self.settings["compute_reactions"].GetBool(), 
                                                self.settings["reform_dofs_at_each_step"].GetBool(), 
                                                self.settings["move_mesh_flag"].GetBool(),
                                                newton_parameters
                                                )
    
    def _create_contact_newton_raphson_strategy(self):
        computing_model_part = self.GetComputingModelPart()
        self.mechanical_scheme = self.get_solution_scheme()
        self.linear_solver = self.get_linear_solver()
        self.mechanical_convergence_criterion = self.get_convergence_criterion()
        self.builder_and_solver = self.get_builder_and_solver()
        newton_parameters = KM.Parameters("""{}""")
        newton_parameters.AddValue("adaptative_strategy",self.contact_strategy_settings["adaptative_strategy"])
        newton_parameters.AddValue("split_factor",self.contact_strategy_settings["split_factor"])
        newton_parameters.AddValue("max_number_splits",self.contact_strategy_settings["max_number_splits"])
        return CSMA.ResidualBasedNewtonRaphsonContactStrategy(computing_model_part, 
                                                                self.mechanical_scheme, 
                                                                self.linear_solver, 
                                                                self.mechanical_convergence_criterion, 
                                                                self.builder_and_solver, 
                                                                self.settings["max_iteration"].GetInt(), 
                                                                self.settings["compute_reactions"].GetBool(), 
                                                                self.settings["reform_dofs_at_each_step"].GetBool(), 
                                                                self.settings["move_mesh_flag"].GetBool(),
                                                                newton_parameters,
                                                                self.processes_list
                                                                )
