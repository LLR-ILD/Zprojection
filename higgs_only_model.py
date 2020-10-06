def getXGBModel():
    import json
    import xgboost

    model_path = "/home/kunath/iLCSoft/projects/scratches/xgboost-ZH/data/neutrino_reco_selected_columns"

    model = xgboost.XGBClassifier()
    model.load_model(f"{model_path}/xgb.model")
    params = json.load(open(f"{model_path}/params.json"))

    training_columns = params["feature_names"]
    # Column names must be adapted for the non-invisible Z decay samples.
    nuToEl = dict(mVis="mH", mMiss="mHRecoil", cosTMiss="cosTH")
    for i in range(len(training_columns)):
        if training_columns[i] in nuToEl:
            training_columns[i] = nuToEl[training_columns[i]]

    return model, training_columns