from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, FloatField
from wtforms.validators import NumberRange, InputRequired


class DataInput(FlaskForm):
    # Demographics
    demo = "Demographics"
    age = FloatField(
        label='Age (years)',
        id='AGE',
        validators=[InputRequired(), NumberRange(-1, 110)],
        default=60
        )
    sex = SelectField(
        'Sex',
        id='Sex',
        validators=[InputRequired()],
        choices=[
            (1, 'Female'),
            (0, 'Male'),
        ]
        )
    sex.dtype = "demographic"
    bmi = FloatField(
        'Body Mass Index',
        id='BMI',
        validators=[InputRequired(), NumberRange(-1, 110)],
        default=25
        )
    bmi.dtype = "demographic"
    race = SelectField(
        'Indigenous Australian',
        id='Race1',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    race.dtype = "demographic"
    insur = SelectField(
        'Insurance',
        id='Insur',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, 'Private'),
            (2, 'DVA'),
            (3, 'Medicare'),
            (4, 'Self Insured'),
            (5, 'Overseas'),
            (6, 'Other'),
        ],
        default=-1
        )
    insur.dtype = "demographic"

    # History
    pop = SelectField(
        'Previous Cardiothoracic Procedure',
        id='POP',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    pop.dtype = "history"
    arrt = SelectField(
        'History of Arrhythmia',
        id='ARRT',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    arrt.dtype = "history"
    smoh = SelectField(
        'History of Smoking',
        id='SMO_H',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    smoh.dtype = "history"
    smoc = SelectField(
        'Current Smoker',
        id='SMO_C',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    smoc.dtype = "history"

    db = SelectField(
        'History of Diabetes',
        id='DB',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    db.dtype = "history"

    db_con = SelectField(
        'Diabetes Control',
        id='DB_CON',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, 'None'),
            (2, 'Diet Controlled'),
            (3, 'Oral Hypoglycaemics'),
            (4, 'Insulin requiring'),
        ],
        default=-1
        )
    db_con.dtype = "history"

    hchol = SelectField(
        'History of Hypercholesterolaemia',
        id='HCHOL',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    hchol.dtype = "history"
    htn = SelectField(
        'History of Hypertension',
        id='HYT',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    htn.dtype = "history"
    ld = SelectField(
        'Lung Disease',
        id='LD_T',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown/None'),
            (2, "Mild (chronic puffer use)"),
            (3, "Moderate (chronic oral steroid)"),
            (4, "Severe Room air p0 2 < 60 or room air pC0 2 > 50"),
        ],
        default=-1
        )
    ld.dtype = "history"
    cbvd = SelectField(
        'Cerebrovascular Disease',
        id='CBVD_T',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown/None'),
            (1, "Coma"),
            (2, "CVA"),
            (3, "RIND/TIA"),
            (4, "Carotid Occlusive Disease")
        ],
        default=-1
        )
    cbvd.dtype = "history"
    pvd = SelectField(
        'History of Peripheral Vascular Disease',
        id='PVD',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    pvd.dtype = "history"
    chf = SelectField(
        'History of Heart Failure',
        id='CHF',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    chf.dtype = "history"
    dial = SelectField(
        'Preoperative Dialysis Requirement',
        id='DIAL',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    dial.dtype = "history"    
    trans = SelectField(
        'History of Renal Transplant',
        id='TRANS',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    trans.dtype = "history"
    procno = FloatField(
        'Number of Previous Procedures (this admission)',
        id='PROCNO',
        validators=[InputRequired(), NumberRange(-1, 10)],
        default=-1
        )
    procno.dtype = "history"

    # Physiology
    chf_c = SelectField(
        'Heart Failure on Admission',
        id='CHF_C',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    chf_c.dtype = "physiology"
    ef = FloatField(
        'Ejection Fraction (%)',
        id='EF',
        validators=[InputRequired(), NumberRange(-1, 100)],
        default=-1
        )
    ef.dtype = "physiology"    
    ef_est = SelectField(
        'Ejection Fraction Estimate',
        id='EF_EST',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, ">60"),
            (2, "46-60"),
            (3, "30-45"),
            (4, "<30")
        ],
        default=-1
        )
    ef_est.dtype = "physiology"
    nyha = SelectField(
        'NYHA Class',
        id='NYHA',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, "Class I"),
            (2, "Class II"),
            (3, "Class III"),
            (4, "Class IV")
        ],
        default=-1
        )
    nyha.dtype = "physiology"
    shock = SelectField(
        'Preoperative Shock',
        id='SHOCK',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    shock.dtype = "physiology"
    ie = SelectField(
        'Infective Endocarditis',
        id='IE_T',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, 'Active'),
            (2, 'Treated'),
        ],
        default=-1
        )
    ie.dtype = "physiology"

    # Operative Characteristics
    stat = SelectField(
        'Urgent Status',
        id='STAT',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, 'Elective'),
            (2, 'Urgent'),
            (3, 'Emergent'),
            (4, 'Salvage')
        ],
        default=-1
        )
    stat.dtype = "operative"
    tp = SelectField(
        'Type of Procedure',
        id='TP',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (1, 'CABG Alone'),
            (2, 'Isolated Valve'),
            (3, 'CABG and Valve'),
            (4, 'Other')
        ],
        default=-1
        )
    tp.dtype = "tp"
    aortic = SelectField(
        'Aortic Valve',
        id='AORTIC',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    aortic.dtype = "operative"
    mitral = SelectField(
        'Mitral Valve',
        id='MITRAL',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    mitral.dtype = "operative"
    tricus = SelectField(
        'Tricuspid Valve',
        id='TRICUS',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    tricus.dtype = "operative"
    pulm = SelectField(
        'Pulmonary Valve',
        id='PULM',
        validators=[InputRequired()],
        choices=[
            (-1, 'Unknown'),
            (0, 'No'),
            (1, 'Yes'),
        ],
        default=-1
        )
    pulm.dtype = "operative"
    cct = FloatField(
        'Cumulative Cross Clamp Time (mins)',
        id='CCT',
        validators=[InputRequired(), NumberRange(-1, 1000)],
        default=-1
        )
    cct.dtype = "operative"
    perf = FloatField(
        'Total Perfusion Time (mins)',
        id='PERF',
        validators=[InputRequired(), NumberRange(-1, 1000)],
        default=-1
        )
    perf.dtype = "operative"
    # Labs
    precr = FloatField(
        'Preperative Creatinine (micromol/L)',
        id='PRECR',
        validators=[InputRequired(), NumberRange(-1, 2000)],
        default=-1
        )
    precr.dtype = "labs"
    egfr = FloatField(
        'eGFR (mL/min per 1.73m2)',
        id='eGFR',
        validators=[InputRequired(), NumberRange(-1, 90)],
        default=-1
        )
    egfr.dtype = "labs"
    hg = FloatField(
        'Preoperative Haemaglobin (g/L)',
        id='HG',
        validators=[InputRequired(), NumberRange(-1, 250)],
        default=-1
        )
    hg.dtype = "labs"
    minht = FloatField(
        'Minimum intra-op haemaglobin (g/L)',
        id='MINHT',
        validators=[InputRequired(), NumberRange(-1, 250)],
        default=-1
        )
    minht.dtype = "labs"
    icu = FloatField(
        'Hours in ICU',
        id='ICU',
        validators=[InputRequired(), NumberRange(-1, 1000)],
        default=-1
        )
    # early postop
    icu.dtype = "postop"
    vent = FloatField(
        'Hours Ventilated Postoperatively',
        id='VENT',
        validators=[InputRequired(), NumberRange(-1, 1000)],
        default=-1
        )
    vent.dtype = "postop"
    drain = FloatField(
        'Total Chest Drain Output in First 4 Hours (mL)',
        id='DRAIN_4',
        validators=[InputRequired(), NumberRange(-1, 4000)],
        default=-1
        )
    drain.dtype = "postop"
    submit = SubmitField('Submit')
