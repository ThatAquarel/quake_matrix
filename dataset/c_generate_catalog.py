import pandas as pd

import a_generate_spectrogram as a


PRE_CATALOG_LUNAR = (
    f"{a.DATA_DIR}lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
)
PRE_CATALOG_MARS = (
    f"{a.DATA_DIR}mars/training/catalogs/Mars_InSight_training_catalog_final.csv"
)

CATALOG_LUNAR = f"{a.PREPROCESS_DIR}lunar/training/catalog.csv"
CATALOG_MARS = f"{a.PREPROCESS_DIR}mars/training/catalog.csv"


def process(pre_catalog, catalog):
    c = pd.read_csv(pre_catalog)

    if "lunar" in pre_catalog:
        c.drop(c[c["evid"] == "evid00029"].index)

        c["filename"] = c["filename"] + ".csv"
    else:
        pass

    c.to_csv(catalog, index=False)


def main():
    for pre_catalog, catalog in zip(
        [PRE_CATALOG_LUNAR, PRE_CATALOG_MARS], [CATALOG_LUNAR, CATALOG_MARS]
    ):
        process(pre_catalog, catalog)


if __name__ == "__main__":
    main()
