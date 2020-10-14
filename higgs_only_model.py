def getXGBModel():
    import json
    import xgboost

    model = xgboost.XGBClassifier()

    # model_path = "/home/kunath/iLCSoft/projects/scratches/xgboost-ZH/data/neutrino_reco_selected_columns"
    # model.load_model(f"{model_path}/xgb.model")
    # params = json.load(open(f"{model_path}/params.json"))
    # training_columns = params["feature_names"]

    # If an error occurs when loading the model, make sure the xgboost version
    # used for building is the same one that is used here.
    model.load_model("/home/kunath/Zprojection/data/best_booster.bin")
    training_columns = [
    'mH', 'mHRecoil', 'cosTZ',
    'nChargedHadrons', 'nNeutralHadrons', 'nGamma', 'nElectrons', 'nMuons',
    'principleThrustZ', 'principleThrust', 'majorThrust', 'minorThrust',
    'sphericity', 'aplanarity',
    'cosTIsoLep', 'eHighestIsoLep', 'nIsoLeptons',
]

    # Column names must be adapted for the non-invisible Z decay samples.
    nuToEl = dict(mVis="mH", mMiss="mHRecoil", cosTMiss="cosTH")
    for i in range(len(training_columns)):
        if training_columns[i] in nuToEl:
            training_columns[i] = nuToEl[training_columns[i]]

    return model, training_columns