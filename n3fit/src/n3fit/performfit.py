"""
    Fit action controller
"""

# Backend-independent imports
import copy
import logging
import numpy as np
import scipy as sp
import n3fit.checks
from n3fit.vpinterface import N3PDF

import yaml

log = logging.getLogger(__name__)

from validphys.initialisation_specs import AnalyticInitialisation
from validphys.core import PDF
from validphys.convolution import predictions

import pandas as pd
import os

from validphys.loader import Loader

l = Loader()

def analytic_solution(data, theorySM, theorylin, covmat):
    """
    Returns the minimum of the chi2 function:

      chi2 = (data - theorySM - theorylin c)^T invcovmat (data - theorySM - theorylin c),

    """

    diff = data.to_numpy() - theorySM.to_numpy()

    theorylin = theorylin.to_numpy()

    part1 = np.linalg.solve(covmat, theorylin)
    part2 = np.linalg.solve(covmat, diff)

    sol = np.linalg.solve(theorylin.T @ part1, theorylin.T @ part2)

    minval = (diff - theorylin @ sol).T @ np.linalg.solve(covmat, diff - theorylin @ sol)
    minval = minval / len(diff)

    return (sol, minval)

# Action to be called by validphys
# All information defining the NN should come here in the "parameters" dict
@n3fit.checks.can_run_multiple_replicas
def performfit(
    *,
    analytic_check,
    data,
    groups_covmat,
    groups_index,
    n3fit_checks_action, # wrapper for all checks
    replicas, # checks specific to performfit
    replicas_nnseed_fitting_data_dict,
    posdatasets_fitting_pos_dict,
    integdatasets_fitting_integ_dict,
    theoryid,
    n_simu_parameters,
    basis,
    fitbasis,
    simu_parameters_scales,
    bsm_fac_initialisations,
    use_th_covmat=False,
    analytic_initialisation_pdf=None,
    fixed_pdf_fit=False,
    sum_rules=True,
    parameters,
    replica_path,
    output_path,
    save=None,
    load_weights_from_fit=None,
    hyperscanner=None,
    hyperopt=None,
    kfold_parameters,
    tensorboard=None,
    debug=False,
    maxcores=None,
    parallel_models=False, 
    simu_parameters_names=None,
    bsm_initialisation_seed=0,
):
    """
        This action will (upon having read a validcard) process a full PDF fit
        for a set of replicas.

        The input to this function is provided by validphys
        and/or defined in the runcards or commandline arguments.

        This controller is provided with:
        1. Seeds generated using the replica number and the seeds defined in the runcard.
        2. Loaded datasets with replicas generated.
            2.1 Loaded positivity/integrability sets.

        The workflow of this controller is as follows:
        1. Generate a ModelTrainer object holding information to create the NN and perform a fit
            (at this point no NN object has been generated)
            1.1 (if hyperopt) generates the hyperopt scanning dictionary
                    taking as a base the fitting dictionary and the runcard's hyperscanner dictionary
        2. Pass the dictionary of parameters to ModelTrainer
                                        for the NN to be generated and the fit performed
            2.1 (if hyperopt) Loop over point 4 for `hyperopt` number of times
        3. Once the fit is finished, output the PDF grid and accompanying files

        Parameters
        ----------
            genrep: bool
                Whether or not to generate MC replicas. (Only used for checks)
            data: validphys.core.DataGroupSpec
                containing the datasets to be included in the fit. (Only used
                for checks)
            replicas_nnseed_fitting_data_dict: list[tuple]
                list with element for each replica (typically just one) to be
                fitted. Each element
                is a tuple containing the replica number, nnseed and
                ``fitted_data_dict`` containing all of the data, metadata
                for each group of datasets which is to be fitted.
            posdatasets_fitting_pos_dict: list[dict]
                list of dictionaries containing all data and metadata for each
                positivity dataset
            integdatasets_fitting_integ_dict: list[dict]
                list of dictionaries containing all data and metadata for each
                integrability dataset
            theoryid: validphys.core.TheoryIDSpec
                Theory which is used to generate theory predictions from model
                during fit. Object also contains some metadata on the theory
                settings.
            basis: list[dict]
                preprocessing information for each flavour to be fitted.
            fitbasis: str
                Valid basis which the fit is to be ran in. Available bases can
                be found in :py:mod:`validphys.pdfbases`.
            sum_rules: bool
                Whether to impose sum rules in fit. By default set to True
            parameters: dict
                Mapping containing parameters which define the network
                architecture/fitting methodology.
            replica_path: pathlib.Path
                path to the output of this run
            output_path: str
                name of the fit
            save: None, str
                model file where weights will be saved, used in conjunction with
                ``load``.
            load_weights_from_fit: None, str
                PDF fit from which to load weights from.
            hyperscanner: dict
                dictionary containing the details of the hyperscanner
            hyperopt: int
                if given, number of hyperopt iterations to run
            kfold_parameters: None, dict
                dictionary with kfold settings used in hyperopt.
            tensorboard: None, dict
                mapping containing tensorboard settings if it is to be used. By
                default it is None and tensorboard is not enabled.
            debug: bool
                activate some debug options
            maxcores: int
                maximum number of (logical) cores that the backend should be aware of
            parallel_models: bool
                whether to run models in parallel
    """
    from n3fit.backends import set_initial_state

    # If debug is active, the initial state will be fixed so that the run is reproducible
    set_initial_state(debug=debug, max_cores=maxcores)

    from n3fit.stopwatch import StopWatch

    stopwatch = StopWatch()

    # All potentially backend dependent imports should come inside the fit function
    # so they can eventually be set from the runcard
    from n3fit.model_trainer import ModelTrainer
    from n3fit.io.writer import WriterWrapper

    # Note: there are three possible scenarios for the loop of replicas:
    #   1.- Only one replica is being run, in this case the loop is only evaluated once
    #   2.- Many replicas being run, in this case each will have a replica_number, seed, etc
    #       and they will be fitted sequentially
    #   3.- Many replicas being run in parallel. In this case the loop will be evaluated just once
    #       but a model per replica will be generated
    #
    # In the main scenario (1) replicas_nnseed_fitting_data_dict is a list of just one element
    # case (3) is similar but the one element of replicas_nnseed_fitting_data_dict will be modified
    # to be (
    #       [list of all replica idx],
    #       one experiment with data=(replicas, ndata),
    #       [list of all NN seeds]
    #       )
    #

    rep_num = replicas_nnseed_fitting_data_dict[0][0]

    compute_analytic = False
    for ini in bsm_fac_initialisations:
        if isinstance(ini, AnalyticInitialisation):
            compute_analytic = True
            use_analytic_initialisation = True
        else:
            use_analytic_initialisation = False

    if compute_analytic or analytic_check:
        # Compute the initialisations
        sm_predictions = []
        linear_bsm = []
        th_covmat = []
        cuts_dict = {}
        for ds in data.datasets:
            cuts_dict[ds.name] = ds.cuts.load()
            pred_values = predictions(ds, PDF(analytic_initialisation_pdf))[rep_num].to_numpy()
            ndat = len(pred_values)
            for label in groups_index:
                if ds.name == label[1]:
                    group_name = label[0]
                    break
            new_index = pd.MultiIndex.from_tuples([(group_name, ds.name, x) for x in cuts_dict[ds.name]])
            sm_predictions += [pd.DataFrame(pred_values, index=new_index)]
            simu_path = l.datapath / ('theory_' + data.thspec.id) / 'simu_factors' / ('SIMU_' + ds.name + '.yaml')
            nop = n_simu_parameters
            if os.path.exists(simu_path) and ds.simu_parameters_linear_combinations is not None:
                with open(simu_path, 'r') as f:
                     simu_info = yaml.safe_load(f)

                columns = []
                for param in ds.simu_parameters_linear_combinations:
                    model = '_'.join(param.split("_")[:-1])
                    column = np.zeros((ndat,))
                    cuts = ds.cuts.load()
                    for key in ds.simu_parameters_linear_combinations[param]:
                        if key in simu_info[model].keys():
                            model_values = [simu_info[model][key][i] for i in cuts]
                            column += np.array(model_values * ds.simu_parameters_linear_combinations[param][key])
                    column = column / np.array([simu_info[model]['SM'][i] for i in cuts]) * pred_values
                    columns += [column]
                linear_bsm += [pd.DataFrame(np.array(columns).T, index=new_index)]

                if use_th_covmat == True and 'theory_cov' in simu_info.keys() and len(simu_info['theory_cov']) > 0:
                    th_covmat += [np.array(simu_info['theory_cov'])]
                else:
                    th_covmat += [np.zeros((ndat, ndat))]
            else:
                linear_bsm += [pd.DataFrame(np.zeros((ndat, nop)), index=new_index)]
                th_covmat += [np.zeros((ndat, ndat))]

        exp_data = []
        for i in range(len(replicas_nnseed_fitting_data_dict[0][1])):
            dictionary = replicas_nnseed_fitting_data_dict[0][1]
            exp_name = dictionary[i]['name']
            exp_index = []
            for dataset in dictionary[i]['datasets']:
                # Must apply both training mask *and* cuts
                tr_mask = dataset['ds_tr_mask']
                indices = [(exp_name, dataset['name'], x) for x in cuts_dict[dataset['name']]]
                masked_indices = []
                for s in range(len(indices)):
                    if tr_mask[s]:
                        masked_indices += [indices[s]] 
                exp_index += masked_indices
            exp_index = pd.MultiIndex.from_tuples(exp_index)
            exp_data += [pd.DataFrame(replicas_nnseed_fitting_data_dict[0][1][i]['expdata'], columns=exp_index)]

        exp_data = pd.concat(exp_data, axis=1).T
        index_with_cuts_and_tr = exp_data.index

        # Take only the prediction corresponding to the replica we are interested in
        #sm_predictions = pd.DataFrame(pd.concat(sm_predictions).to_numpy()[:,rep_num])
        sm_predictions = pd.concat(sm_predictions)

        linear_bsm = pd.concat(linear_bsm)

        th_covmat = sp.linalg.block_diag(*th_covmat)
        th_covmat = pd.DataFrame(th_covmat)
        th_covmat.index = sm_predictions.index
        th_covmat = th_covmat.T
        th_covmat.index = sm_predictions.index

        # Now we need to reindex everything, because NNPDF is annoying
        sm_predictions = sm_predictions.loc[index_with_cuts_and_tr]
        linear_bsm = linear_bsm.loc[index_with_cuts_and_tr]
        th_covmat = th_covmat.loc[index_with_cuts_and_tr]
        th_covmat = th_covmat.T.loc[index_with_cuts_and_tr]

        covmat = groups_covmat
        covmat = covmat.loc[index_with_cuts_and_tr]
        covmat = covmat.T.loc[index_with_cuts_and_tr]

        total_covmat = covmat + th_covmat

        analytic_initialisation, minval = analytic_solution(exp_data, sm_predictions, linear_bsm, total_covmat) 

    else:
        analytic_initialisation = None


    if analytic_check:
        log.info("The analytic solution is " + str(analytic_initialisation.T.tolist()))
        log.info("The minimum is achieved at chi2=" + str(minval))

    if not use_analytic_initialisation:
        analytic_initialisation = None

    n_models = len(replicas_nnseed_fitting_data_dict)
    if parallel_models and n_models != 1:
        replicas, replica_experiments, nnseeds = zip(*replicas_nnseed_fitting_data_dict)
        # Parse the experiments so that the output data contain information for all replicas
        # as the only different from replica to replica is the experimental training/validation data
        all_experiments = copy.deepcopy(replica_experiments[0])
        for i_exp in range(len(all_experiments)):
            training_data = []
            validation_data = []
            for i_rep in range(n_models):
                training_data.append(replica_experiments[i_rep][i_exp]['expdata'])
                validation_data.append(replica_experiments[i_rep][i_exp]['expdata_vl'])
            all_experiments[i_exp]['expdata'] = np.concatenate(training_data, axis=0)
            all_experiments[i_exp]['expdata_vl'] = np.concatenate(validation_data, axis=0)
        log.info(
            "Starting parallel fits from replica %d to %d",
            replicas[0],
            replicas[0] + n_models - 1,
        )
        replicas_info = [(replicas, all_experiments, nnseeds)]
    else:
        replicas_info = replicas_nnseed_fitting_data_dict

    for replica_idxs, exp_info, nnseeds in replicas_info:
        if not parallel_models or n_models == 1:
            # Cases 1 and 2 above are a special case of 3 where the replica idx and the seed should
            # be a list of just one element
            replica_idxs = [replica_idxs]
            nnseeds = [nnseeds]
            log.info("Starting replica fit %d", replica_idxs[0])

        # Generate a ModelTrainer object
        # this object holds all necessary information to train a PDF (up to the NN definition)
        the_model_trainer = ModelTrainer(
            exp_info,
            posdatasets_fitting_pos_dict,
            integdatasets_fitting_integ_dict,
            basis,
            fitbasis,
            nnseeds,
            replicas,
            fixed_pdf=fixed_pdf_fit,
            debug=debug,
            kfold_parameters=kfold_parameters,
            max_cores=maxcores,
            model_file=load_weights_from_fit,
            sum_rules=sum_rules,
            parallel_models=n_models,
            n_simu_parameters=n_simu_parameters,
            simu_parameters_names=simu_parameters_names, 
            simu_parameters_scales=simu_parameters_scales,
            bsm_fac_initialisations=bsm_fac_initialisations,
            analytic_initialisation=analytic_initialisation,
            bsm_initialisation_seed=bsm_initialisation_seed,
        )

        # This is just to give a descriptive name to the fit function
        pdf_gen_and_train_function = the_model_trainer.hyperparametrizable

        # Read up the parameters of the NN from the runcard
        stopwatch.register_times("replica_set")

        ########################################################################
        # ### Hyperopt                                                         #
        # If hyperopt is active the parameters of NN will be substituted by the#
        # hyoperoptimizable variables.                                         #
        # Hyperopt will run for --hyperopt number of iterations before leaving #
        # this block                                                           #
        ########################################################################
        if hyperopt:
            from n3fit.hyper_optimization.hyper_scan import hyper_scan_wrapper

            # Note that hyperopt will not run in parallel or with more than one model _for now_
            replica_path_set = replica_path / f"replica_{replica_idxs[0]}"
            true_best = hyper_scan_wrapper(
                replica_path_set, the_model_trainer, hyperscanner, max_evals=hyperopt
            )
            print("##################")
            print("Best model found: ")
            for k, i in true_best.items():
                print(f" {k} : {i} ")

            # In general after we do the hyperoptimization we do not care about the fit
            # so just let this die here
            break
        ####################################################################### end of hyperopt

        # Ensure hyperopt is off
        the_model_trainer.set_hyperopt(False)

        # Enable the tensorboard callback
        if tensorboard is not None:
            profiling = tensorboard.get("profiling", False)
            weight_freq = tensorboard.get("weight_freq", 0)
            if parallel_models and n_models != 1:
                # If using tensorboard when running in parallel
                # dump the debugging data to the nnfit folder
                replica_path_set = replica_path
            else:
                replica_path_set = replica_path / f"replica_{replica_idxs[0]}"
            log_path = replica_path_set / "tboard"
            the_model_trainer.enable_tensorboard(log_path, weight_freq, profiling)

        #############################################################################
        # ### Fit                                                                   #
        # This function performs the actual fit, it reads all the parameters in the #
        # "parameters" dictionary, uses them to generate the NN and trains the net  #
        #############################################################################
        result = pdf_gen_and_train_function(parameters)
        stopwatch.register_ref("replica_fitted", "replica_set")

        stopping_object = result["stopping_object"]
        log.info("Stopped at epoch=%d", stopping_object.stop_epoch)

        final_time = stopwatch.stop()
        all_training_chi2, all_val_chi2, all_exp_chi2 = the_model_trainer.evaluate(stopping_object)

        pdf_models = result["pdf_models"]
        for i, (replica_number, pdf_model) in enumerate(zip(replica_idxs, pdf_models)):
            # Each model goes into its own replica folder
            replica_path_set = replica_path / f"replica_{replica_number}"

            # Create a pdf instance
            q0 = theoryid.get_description().get("Q0")
            pdf_instance = N3PDF(pdf_model, fit_basis=basis, Q=q0)

            bsm_fac_df=result["bsm_fac_df"]

            # Generate the writer wrapper
            writer_wrapper = WriterWrapper(
                replica_number,
                pdf_instance,
                stopping_object,
                q0**2,
                final_time,
            )

            # Get the right chi2s
            training_chi2 = np.take(all_training_chi2, i)
            val_chi2 = np.take(all_val_chi2, i)
            exp_chi2 = np.take(all_exp_chi2, i)

            # And write the data down
            writer_wrapper.write_data(
                replica_path_set, output_path.name, training_chi2, val_chi2, exp_chi2, bsm_fac_df 
            )
            log.info(
                    "Best fit for replica #%d, chi2=%.3f (tr=%.3f, vl=%.3f)",
                    replica_number,
                    exp_chi2,
                    training_chi2,
                    val_chi2
                    )


            # Save the weights to some file for the given replica
            if save:
                model_file_path = replica_path_set / save
                log.info(" > Saving the weights for future in %s", model_file_path)
                # Need to use "str" here because TF 2.2 has a bug for paths objects (fixed in 2.3)
                pdf_model.save_weights(str(model_file_path), save_format="h5")

        if tensorboard is not None:
            log.info("Tensorboard logging information is stored at %s", log_path)
