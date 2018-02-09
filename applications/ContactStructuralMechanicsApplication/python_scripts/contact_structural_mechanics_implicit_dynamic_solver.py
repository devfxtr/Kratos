from __future__ import print_function, absolute_import, division  # makes KM backward compatible with python 2.6 and 2.7
#import kratos core and applications
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

# Check that KM was imported in the main script
KM.CheckForPreviousImport()

# Import the implicit solver (the explicit one is derived from it)
import structural_mechanics_implicit_dynamic_solver

def CreateSolver(main_model_part, custom_settings):
    return ImplicitMechanicalSolver(main_model_part, custom_settings)

class ImplicitMechanicalSolver(structural_mechanics_implicit_dynamic_solver.ImplicitMechanicalSolver):
    """The structural mechanics contact implicit dynamic solver.

    This class creates the mechanical solvers for contact implicit dynamic analysis.
    It currently supports Newmark, Bossak and dynamic relaxation schemes.

    Public member variables:
    dynamic_settings -- settings for the implicit dynamic solvers.

    See structural_mechanics_solver.py for more information.
    """
    def __init__(self, main_model_part, custom_settings): 
        
        self.main_model_part = main_model_part    
        
        ##settings string in json format
        contact_settings = KM.Parameters("""
        {
            "contact_settings" :
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
            }
        }
        """)
        
        ## Overwrite the default settings with user-provided parameters
        self.settings = custom_settings
        self.validate_and_transfer_matching_settings(self.settings, contact_settings)
        self.contact_settings = contact_settings["contact_settings"]

        # Construct the base solver.
        super().__init__(self.main_model_part, self.settings)
        
        # Setting default configurations true by default
        if (self.settings["clear_storage"].GetBool() == False):
            KM.Logger.PrintWarning("Storage must be cleared each step. Switching to True")
            self.settings["clear_storage"].SetBool(True)
        if (self.settings["reform_dofs_at_each_step"].GetBool() == False):
            KM.Logger.PrintWarning("DoF must be reformed each time step. Switching to True")
            self.settings["reform_dofs_at_each_step"].SetBool(True)
        if (self.settings["block_builder"].GetBool() == False):
            KM.Logger.PrintWarning("EliminationBuilderAndSolver can not used with the current implementation. Switching to BlockBuilderAndSolver")
            self.settings["block_builder"].SetBool(True)
        
        # Setting echo level
        self.echo_level =  self.settings["echo_level"].GetInt()
    
        # Initialize the processes list
        self.processes_list = None
        
        KM.Logger.PrintInfo("Construction of ContactMechanicalSolver finished")

    def AddVariables(self):
        
        super().AddVariables()
        
        mortar_type = self.contact_settings["mortar_type"].GetString()
        if  mortar_type != "":
            self.main_model_part.AddNodalSolutionStepVariable(KM.NORMAL)  # Add normal
            self.main_model_part.AddNodalSolutionStepVariable(KM.NODAL_H) # Add nodal size variable
            if  mortar_type == "ALMContactFrictionless":
                self.main_model_part.AddNodalSolutionStepVariable(KM.NORMAL_CONTACT_STRESS)       # Add normal contact stress
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_GAP)              # Add normal contact gap
            elif mortar_type == "ALMContactFrictionlessComponents": 
                self.main_model_part.AddNodalSolutionStepVariable(KM.VECTOR_LAGRANGE_MULTIPLIER)  # Add normal contact stress 
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_GAP)              # Add normal contact gap 
            elif mortar_type == "ALMContactFrictional": 
                self.main_model_part.AddNodalSolutionStepVariable(KM.VECTOR_LAGRANGE_MULTIPLIER)  # Add normal contact stress 
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_GAP)              # Add normal contact gap 
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_SLIP)             # Add contact slip
            elif  mortar_type == "ScalarMeshTying":
                self.main_model_part.AddNodalSolutionStepVariable(KM.SCALAR_LAGRANGE_MULTIPLIER)  # Add scalar LM
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_SCALAR_RESIDUAL)  # Add scalar LM residual
            elif  mortar_type == "ComponentsMeshTying":
                self.main_model_part.AddNodalSolutionStepVariable(KM.VECTOR_LAGRANGE_MULTIPLIER)  # Add vector LM
                self.main_model_part.AddNodalSolutionStepVariable(CSMA.WEIGHTED_VECTOR_RESIDUAL)  # Add vector LM residual
   
        KM.Logger.PrintInfo("::[Contact Mechanical Solver]:: Variables ADDED")
        
    def AddDofs(self):

        super().AddDofs()
        
        mortar_type = self.contact_settings["mortar_type"].GetString()
        if (mortar_type == "ALMContactFrictionless"):
            KM.VariableUtils().AddDof(KM.NORMAL_CONTACT_STRESS, CSMA.WEIGHTED_GAP, self.main_model_part)
        elif (mortar_type == "ALMContactFrictional" or mortar_type == "ALMContactFrictionlessComponents"): 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_X, self.main_model_part) 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Y, self.main_model_part) 
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Z, self.main_model_part) 
        elif (mortar_type == "ScalarMeshTying"):
            KM.VariableUtils().AddDof(KM.SCALAR_LAGRANGE_MULTIPLIER,CSMA.WEIGHTED_SCALAR_RESIDUAL, self.main_model_part)
        elif (mortar_type == "ComponentsMeshTying"):
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_X, CSMA.WEIGHTED_VECTOR_RESIDUAL_X, self.main_model_part)
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Y, CSMA.WEIGHTED_VECTOR_RESIDUAL_Y, self.main_model_part)
            KM.VariableUtils().AddDof(KM.VECTOR_LAGRANGE_MULTIPLIER_Z, CSMA.WEIGHTED_VECTOR_RESIDUAL_Z, self.main_model_part)

        KM.Logger.PrintInfo("::[Contact Mechanical Solver]:: DOF's ADDED")
    
    def Initialize(self):
        super().Initialize() # The mechanical solver is created here.
    
    def Solve(self):
        if self.settings["clear_storage"].GetBool():
            self.Clear()
            
        mechanical_solver = self.get_mechanical_solver()
            
        # The steps of the solve are Initialize(), InitializeSolutionStep(), Predict(), SolveSolutionStep(), FinalizeSolutionStep()        
        mechanical_solver.Solve()
        #mechanical_solver.Initialize()
        #mechanical_solver.InitializeSolutionStep()
        #mechanical_solver.Predict()
        #mechanical_solver.SolveSolutionStep()
        #mechanical_solver.FinalizeSolutionStep()
    
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
        conv_params.AddValue("contact_displacement_relative_tolerance",self.contact_settings["contact_displacement_relative_tolerance"])
        conv_params.AddValue("contact_displacement_absolute_tolerance",self.contact_settings["contact_displacement_absolute_tolerance"])
        conv_params.AddValue("contact_residual_relative_tolerance",self.contact_settings["contact_residual_relative_tolerance"])
        conv_params.AddValue("contact_residual_absolute_tolerance",self.contact_settings["contact_residual_absolute_tolerance"])
        conv_params.AddValue("mortar_type",self.contact_settings["mortar_type"])
        conv_params.AddValue("condn_convergence_criterion",self.contact_settings["condn_convergence_criterion"])
        conv_params.AddValue("fancy_convergence_criterion",self.contact_settings["fancy_convergence_criterion"])
        conv_params.AddValue("print_convergence_criterion",self.contact_settings["print_convergence_criterion"])
        conv_params.AddValue("ensure_contact",self.contact_settings["ensure_contact"])
        conv_params.AddValue("gidio_debug",self.contact_settings["gidio_debug"])
        import contact_convergence_criteria_factory
        convergence_criterion = contact_convergence_criteria_factory.convergence_criterion(conv_params)
        return convergence_criterion.mechanical_convergence_criterion
    
    def _create_builder_and_solver(self):
        if  self.contact_settings["mortar_type"].GetString() != "":
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
        if  self.contact_settings["mortar_type"].GetString() != "":
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
        newton_parameters.AddValue("adaptative_strategy",self.contact_settings["adaptative_strategy"])
        newton_parameters.AddValue("split_factor",self.contact_settings["split_factor"])
        newton_parameters.AddValue("max_number_splits",self.contact_settings["max_number_splits"])
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
