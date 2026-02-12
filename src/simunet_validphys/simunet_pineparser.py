def parse_theory_meta(metadata_path, observable_name):
    """
    From the metadata.yaml file parses something similar to validphys.commondataparser.TheoryMeta.
    This is then needed by the FKspec taken by the pineappl_reader function.
    """
    
    with open(metadata_path, "r") as f:
        meta_card = yaml.safe_load(f)
    
    # Get the theory metadata
    for obs in meta_card['implemented_observables']:
        if obs['observable_name'] == observable_name:
            obs_metadata = obs
    
    # TODO: still figure out how to fill shifts and normalization
    th_meta = TheoryMeta(
        FK_tables= obs_metadata['theory']['FK_tables'],
        operation=obs_metadata['theory']['operation'],
        conversion_factor=obs_metadata['theory']['conversion_factor'],
        shifts=None,
        normalization=None,
        comment=None
    )
    
    return th_meta