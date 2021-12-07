from pathlib import Path

import xgboost


training_columns_options = {
    "2020-10-12_16:16:25_55": [
        "mH", "mHRecoil", "cosTZ",
        "nChargedHadrons", "nNeutralHadrons", "nGamma", "nElectrons", "nMuons",
        "principleThrustZ", "principleThrust", "majorThrust", "minorThrust",
        "sphericity", "aplanarity",
        "cosTIsoLep", "eHighestIsoLep", "nIsoLeptons",
    ],
    "2020-10-25_23:16:24_46": [
        "mH",
        "nChargedHadrons", "nNeutralHadrons", "nGamma", "nElectrons", "nMuons",
        "nIsoLeptons",

    ]
}


def getXGBModel(
    model_location=Path(__file__).parent / "data/best_booster.bin", 
    training_columns=training_columns_options["2020-10-25_23:16:24_46"],
):
    model = xgboost.XGBClassifier()
    # If an error occurs when loading the model, make sure the xgboost version
    # used for building is the same one that is used here.
    model.load_model(model_location)

    # Column names must be adapted for the non-invisible Z decay samples.
    nuToEl = dict(mVis="mH", mMiss="mHRecoil", cosTMiss="cosTH")
    training_columns = list(training_columns)  # Avoid overwriting the input.
    for i in range(len(training_columns)):
        if training_columns[i] in nuToEl:
            training_columns[i] = nuToEl[training_columns[i]]
    return model, training_columns
