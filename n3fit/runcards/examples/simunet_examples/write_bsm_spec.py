def write_bsm_spec_file(filename, operators, datasets):
    with open(filename, 'w') as f:

        operator_names = list(operators.keys())
        operator_scales = [operators[op] for op in operator_names]
        dataset_names = list(datasets.keys())

        firstline = tuple(["Operator"] + operator_names)
        f.write(("%50s    " + "%15s    " * (len(firstline) - 1) + "\n") % firstline)

        secondline = tuple(["Scale"] + operator_scales)
        f.write(("%50s    " + "%15s    " * (len(secondline) - 1) + "\n") % secondline)

        for dataname in dataset_names:
            line = [dataname]
            for op in operator_names:
                if op in datasets[dataname][1]:
                    line += [datasets[dataname][0]]
                else:
                    line += ["None"]

            f.write(("%50s    " + "%15s    " * (len(line) - 1) + "\n") % tuple(line))             

operators = {
    'OtG': 1.0,
    'Opt': 1.0,
    'OtZ': 1.0,
    'OtW': 1.0,
    'O3pQ3': 1.0,
    'OpQM': 1.0,
    'O8qd': 1.0,
    'O1qd': 1.0,
    'O1qu': 1.0,
    'O8qu': 1.0,
    'O1dt': 1.0, 
    'O8dt': 1.0, 
    'O1qt': 1.0, 
    'O8qt': 1.0, 
    'O1ut': 1.0, 
    'O8ut': 1.0, 
    'O11qq': 1.0, 
    'O13qq': 1.0, 
    'O81qq': 1.0, 
    'O83qq': 1.0,
    # 'OQt8': 1.0, 
    # 'OQQ1': 1.0, 
    # 'OQQ8': 1.0, 
    # 'OQt1': 1.0, 
    # 'Ott1': 1.0,
}

ttbar_operators = ['OtG', 'O8qd', 'O1qd', 'O1qu', 'O8qu', 'O1dt', 'O8dt', 'O1qt', 'O8qt', 'O1ut', 'O8ut', 'O11qq', 'O13qq', 'O81qq', 'O83qq']

ttz_operators = ['OtG', 'Opt', 'OtZ', 'O3pQ3', 'OpQM', 'O8qd', 'O1qd', 'O1qu', 'O8qu', 'O1dt', 'O8dt', 'O1qt', 'O8qt', 'O1ut', 'O8ut', 'O11qq', 'O13qq', 'O81qq', 'O83qq']

ttw_operators = ['OtG', 'O1qt', 'O8qt', 'O11qq', 'O13qq', 'O81qq', 'O83qq']

tz_operators = ['Opt', 'OtZ', 'OtW', 'O3pQ3', 'OpQM', 'O13qq', 'O83qq']

tw_operators = ['O3pQ3', 'OtG', 'OtW']

tta_operators = ['OtG', 'OtZ', 'OtW', 'O8qd', 'O1qd', 'O1qu', 'O8qu', 'O1dt', 'O8dt', 'O1qt', 'O8qt', 'O1ut', 'O8ut', 'O11qq', 'O13qq', 'O81qq', 'O83qq']

tttt_operators = ['OtG', 'O8qd', 'O1qd', 'O1qu', 'O8qu', 'O1dt', 'O8dt', 'O1qt', 'O8qt', 'O1ut', 'O8ut', 'O11qq', 'O13qq', 'O81qq', 'O83qq', 'OQt8', 'OQQ1', 'OQQ8', 'OQt1', 'Ott1']

ttbb_operators = ['OtG', 'O8qd', 'O1qd', 'O1qu', 'O8qu', 'O1dt', 'O8dt', 'O1qt', 'O8qt', 'O1ut', 'O8ut', 'O11qq', 'O13qq', 'O81qq', 'O83qq', 'OQt8', 'OQQ1', 'OQQ8', 'OQt1']

singletop_operators = ['O3pQ3', 'OtW', 'O13qq', 'O83qq']


datasets = {
    # ttbar
    'ATLASTTBARTOT7TEV':   ('LO_LIN', ttbar_operators),
    'ATLASTTBARTOT8TEV':   ('LO_LIN', ttbar_operators),
    'ATLAS_TOPDIFF_DILEPT_8TEV_TTMNORM':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_8TEV_LJETS_TOTAL':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_13TEV_DILEPTON_TOTAL':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_13TEV_HADRONIC_TOTAL':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_13TEV_HADRONIC_2D_TTM_ABSYTTNORM':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_13TEV_LJETS_TOTAL':   ('LO_LIN', ttbar_operators),
    'ATLAS_TTBAR_13TEV_TTMNORM':   ('LO_LIN', ttbar_operators),
    'CMSTTBARTOT5TEV':   ('LO_LIN', ttbar_operators),
    'CMSTTBARTOT7TEV':   ('LO_LIN', ttbar_operators),
    'CMSTTBARTOT8TEV':   ('LO_LIN', ttbar_operators),
    'CMS_TTBAR_2D_DIFF_MTT_TTRAP_NORM':   ('LO_LIN', ttbar_operators),
    'CMSTOPDIFF8TEVTTRAPNORM':   ('LO_LIN', ttbar_operators),
    'CMSTTBARTOT13TEV':   ('LO_LIN', ttbar_operators),
    'CMS_TTB_DIFF_13TEV_2016_2L_TTMNORM':   ('LO_LIN', ttbar_operators),
    'CMS_TTBAR_13TEV_LJETS_TOTAL':   ('LO_LIN', ttbar_operators),
    'CMS_TTBAR_13TEV_LJETS_2D_TTM_ABSYTTNORM':   ('LO_LIN', ttbar_operators),

    # ttbar AC
    'ATLAS_CMS_TTBAR_8TEV_ASY': ('None', ttbar_operators),
    'ATLAS_TTBAR_13TEV_ASY': ('None', ttbar_operators),
    'ATLAS_TTBAR_8TEV_ASY': ('None', ttbar_operators),
    'CMS_TTBAR_13TEV_ASY': ('None', ttbar_operators),
    'CMS_TTBAR_8TEV_ASY': ('None', ttbar_operators),
    
    # Whel
    # 'ATLAS_CMS_WHEL_8TEV': ('LO_LIN', ttbar_operators),
    
    # ttZ
    'CMS_TTBARZ_8TEV_TOTAL': ('LO_LIN', ttz_operators),
    'CMS_TTBARZ_13TEV_TOTAL': ('LO_LIN', ttz_operators),
    'CMS_TTBARZ_13TEV_PTZNORM': ('LO_LIN', ttz_operators),
    'ATLAS_TTBARZ_8TEV_TOTAL': ('LO_LIN', ttz_operators),
    'ATLAS_TTBARZ_13TEV_TOTAL': ('LO_LIN', ttz_operators),
    'ATLAS_TTBARZ_13TEV_PTZNORM': ('LO_LIN', ttz_operators),

    # ttW
    'CMS_TTBARW_8TEV_TOTAL': ('LO_LIN', ttw_operators),
    'CMS_TTBARW_13TEV_TOTAL': ('LO_LIN', ttw_operators),
    'ATLAS_TTBARW_8TEV_TOTAL': ('LO_LIN', ttw_operators),
    'ATLAS_TTBARW_13TEV_TOTAL': ('LO_LIN', ttw_operators),

    # ttgamma
    # 'ATLAS_TTBARGAMMA_8TEV_TOTAL': ('LO_LIN', tta_operators),
    # 'CMS_TTBARGAMMA_8TEV_TOTAL': ('LO_LIN', tta_operators),
    # 'ATLAS_TTBARGAMMA_13TEV_PTGAMMA': ('LO_LIN', tta_operators),

    # singletop
    'CMS_SINGLETOP_TCH_8TEV_T': ('LO_LIN', singletop_operators),
    'CMS_SINGLETOP_TCH_8TEV_TB': ('LO_LIN', singletop_operators),
    'ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP': ('LO_LIN', singletop_operators),
    # 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP': ('LO_LIN', singletop_operators),
    'CMS_SINGLETOP_SCH_8TEV_TOTAL': ('LO_LIN', singletop_operators),
    'ATLAS_SINGLETOP_SCH_8TEV_TOTAL': ('LO_LIN', singletop_operators),
    'ATLAS_SINGLETOP_TCH_13TEV_T': ('LO_LIN', singletop_operators),
    'ATLAS_SINGLETOP_TCH_13TEV_TB': ('LO_LIN', singletop_operators),
    'CMS_SINGLETOP_TCH_13TEV_T': ('LO_LIN', singletop_operators),
    'CMS_SINGLETOP_TCH_13TEV_TB': ('LO_LIN', singletop_operators),
    'CMS_SINGLETOP_TCH_13TEV_YTNORM': ('LO_LIN', singletop_operators),

    # tW
    'ATLAS_SINGLETOPW_8TEV_TOTAL': ('LO_LIN', tw_operators),
    'ATLAS_SINGLETOPW_8TEV_SLEP_TOTAL': ('LO_LIN', tw_operators),
    'CMS_SINGLETOPW_8TEV_TOTAL': ('LO_LIN', tw_operators),
    'ATLAS_SINGLETOPW_13TEV_TOTAL': ('LO_LIN', tw_operators),
    'CMS_SINGLETOPW_13TEV_TOTAL': ('LO_LIN', tw_operators),
    'CMS_SINGLETOPW_13TEV_SLEP_TOTAL': ('LO_LIN', tw_operators),

    # tZ
    # 'ATLAS_SINGLETOPZ_13TEV_TOTAL': ('LO_LIN', tz_operators),
    # 'CMS_SINGLETOPZ_13TEV_TOTAL': ('LO_LIN', tz_operators),
    # 'CMS_SINGLETOPZ_13TEV_PTZ': ('LO_LIN', tz_operators),

    # tttt
    # 'CMS_4TOP_13TEV_MULTILEP_TOTAL': ('LO_LIN', tttt_operators),
    # 'ATLAS_4TOP_13TEV_MULTILEP_TOTAL': ('LO_LIN', tttt_operators),
    # 'ATLAS_4TOP_13TEV_SLEP_TOTAL': ('LO_LIN', tttt_operators),

    # ttbb
    # 'CMS_TTBB_8TEV_DILEPTON_TOTAL': ('LO_LIN', ttbb_operators),
    # 'CMS_TTBB_13TEV_ALLJET_TOTAL': ('LO_LIN', ttbb_operators),
    # 'CMS_TTBB_13TEV_LJETS_TOTAL': ('LO_LIN', ttbb_operators),
    # 'CMS_TTBB_13TEV_DILEPTON_TOTAL': ('LO_LIN', ttbb_operators),
}

write_bsm_spec_file("bsm_spec.dat", operators, datasets)