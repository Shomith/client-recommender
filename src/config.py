"""Settings for the next-best-product recommender (Santander Product Recommendation data)."""

DATA_PATH = "data/train_ver2.csv"   # from Kaggle: santander-product-recommendation
SEED = 42
CLIENT_FRACTION = 0.2   # share of clients kept (hash of client id, same clients in every month)

MONTHS = ["2015-10", "2015-11", "2015-12", "2016-01", "2016-02", "2016-03", "2016-04", "2016-05"]

# (previous month, feature month, target month)
TRAIN_SPLITS = [("2015-10", "2015-11", "2015-12"),
                ("2015-11", "2015-12", "2016-01"),
                ("2015-12", "2016-01", "2016-02"),
                ("2016-01", "2016-02", "2016-03")]
VAL_SPLIT = ("2016-02", "2016-03", "2016-04")
TEST_SPLIT = ("2016-03", "2016-04", "2016-05")

N_CANDIDATES = 24   # candidates per client: every product not yet held (only 24 exist)
TOP_N = 10   
KS = (3, 5, 7)
COLD_START_MAX_TENURE = 6

ALS_FACTORS = 8
ALS_REG = 1.0
ALS_ITERS = 20

PRODUCTS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

ID_COL = "ncodpers"
DATE_COL = "fecha_dato"
NUM_COLS = ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente"]
CAT_COLS = ["sexo", "segmento", "canal_entrada", "tiprel_1mes"]
