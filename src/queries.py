import pandas as pd


def create_patient_query(x_train, rcount=0, gender='F', facid='A',
                          hemo=0, irondef=0, psychologicaldisordermajor=0,
                          hematocrit=40, glucose=100, bloodureanitro=15,
                          creatinine=1.0, bmi=22, pulse=72,
                          respiration=16, sodium=140):
    """
    Kreira upit za predikciju dužine hospitalizacije.

    Parametri:
        x_train: DataFrame — enkodovani trening skup (koristi se kao template)
        rcount: int — broj prethodnih hospitalizacija (0, 1, 2, 3, 4, 5+)
        gender: str — pol pacijenta ('F' ili 'M')
        facid: str — bolnica ('A', 'B', 'C', 'D', 'E')
        hemo, irondef, psychologicaldisordermajor: int — binarne dijagnoze (0 ili 1)
        hematocrit, glucose, bloodureanitro, creatinine: float — laboratorijski parametri
        bmi, pulse, respiration, sodium: float — vitalni znaci
    """
    query = x_train.iloc[[0]].copy()

    # rcount one-hot encoding
    for col in [c for c in query.columns if c.startswith('rcount_')]:
        query[col] = 0
    rcount_col = f'rcount_{rcount}'
    if rcount_col in query.columns:
        query[rcount_col] = 1

    # gender one-hot encoding (F je baseline)
    for col in [c for c in query.columns if c.startswith('gender_')]:
        query[col] = 0
    if gender == 'M' and 'gender_M' in query.columns:
        query['gender_M'] = 1

    # facid one-hot encoding (A je baseline)
    for col in [c for c in query.columns if c.startswith('facid_')]:
        query[col] = 0
    facid_col = f'facid_{facid}'
    if facid_col in query.columns:
        query[facid_col] = 1

    # binarne dijagnoze
    query['hemo'] = hemo
    query['irondef'] = irondef
    query['psychologicaldisordermajor'] = psychologicaldisordermajor

    # laboratorijski parametri i vitalni znaci
    query['hematocrit'] = hematocrit
    query['glucose'] = glucose
    query['bloodureanitro'] = bloodureanitro
    query['creatinine'] = creatinine
    query['bmi'] = bmi
    query['pulse'] = pulse
    query['respiration'] = respiration
    query['sodium'] = sodium

    return query