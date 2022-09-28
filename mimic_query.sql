WITH surgery AS
-- Code for type of surgery
(
    SELECT
        ad.hadm_id

        -- CABG
        , MAX(CASE WHEN
            icd9_code IN (
                '3603','3610','3611','3612','3613','3614','3615','3616',
                '3617','3619','362','3631','3632','3633'
               )
            THEN 1 
            ELSE 0 END) AS CABG
        -- AORTIC
        , MAX(CASE WHEN
            icd9_code IN ('3511', '3521', '3522')
            THEN 1 
            ELSE 0 END) AS AORTIC
        -- MITRAL
        , MAX(CASE WHEN
            icd9_code IN ('3512', '3523', '3524')
            THEN 1 
            ELSE 0 END) AS MITRAL
        -- TRICUSPID
        , MAX(CASE WHEN
            icd9_code IN ('3514', '3527', '3528')
            THEN 1 
            ELSE 0 END) AS TRICUSPID
        -- PULMONARY
        , MAX(CASE WHEN
            icd9_code IN ('3513', '3525', '3526')
            THEN 1 
            ELSE 0 END) AS PULMONARY
        -- Thoracic Operation
        , MAX(CASE WHEN
            icd9_code IN ('3845')
       THEN 1 
       ELSE 0 END) AS THORACIC
    FROM `physionet-data.mimiciii_clinical.admissions` ad
    FULL OUTER JOIN `physionet-data.mimiciii_clinical.procedures_icd` AS proc ON ad.HADM_ID = proc.hadm_id
    GROUP BY ad.HADM_ID
)
, ag AS
-- prep age dataframe
(
    SELECT 
        hadm_id,
        icustay_id,
        admission_age,
        icustay_seq,

    FROM `physionet-data.mimiciii_derived.icustay_detail` id
)
, pc AS
-- Get procedure codes 
(
    SELECT array_agg(icd9_code) as ICD_CODES, hadm_id as HADM_ID
    FROM `physionet-data.mimiciii_clinical.procedures_icd`
    GROUP BY HADM_ID
)
, com AS
-- prepare comorbidites according to charleston comorb index
(
    SELECT
        ad.hadm_id
        -- Myocardial infarction
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN ('410','412')
            THEN 1 
            ELSE 0 END) AS myocardial_infarct

        -- Arrhythmia
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN ('427')
            THEN 1 
            ELSE 0 END) AS arrhythmia

        -- Congestive heart failure
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '428'
            OR
            SUBSTR(icd9_code, 1, 5) IN ('39891','40201','40211','40291','40401','40403',
                          '40411','40413','40491','40493')
            OR 
            SUBSTR(icd9_code, 1, 4) BETWEEN '4254' AND '4259'
            THEN 1 
            ELSE 0 END) AS congestive_heart_failure

        -- Peripheral vascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('440','441')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '4431' AND '4439'
            THEN 1 
            ELSE 0 END) AS peripheral_vascular_disease

        -- Cerebrovascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '430' AND '438'
            OR
            SUBSTR(icd9_code, 1, 5) = '36234'
            THEN 1 
            ELSE 0 END) AS cerebrovascular_disease

        -- Dementia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '290'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('2941','3312')
            THEN 1 
            ELSE 0 END) AS dementia

        -- Chronic pulmonary disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '490' AND '505'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('4168','4169','5064','5081','5088')
            THEN 1 
            ELSE 0 END) AS chronic_pulmonary_disease

        -- Rheumatic disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '725'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('4465','7100','7101','7102','7103',
                                                  '7104','7140','7141','7142','7148')
            THEN 1 
            ELSE 0 END) AS rheumatic_disease

        -- Peptic ulcer disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('531','532','533','534')
            THEN 1 
            ELSE 0 END) AS peptic_ulcer_disease

        -- Mild liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('570','571')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('0706','0709','5733','5734','5738','5739','V427')
            OR
            SUBSTR(icd9_code, 1, 5) IN ('07022','07023','07032','07033','07044','07054')
            THEN 1 
            ELSE 0 END) AS mild_liver_disease

        -- Diabetes without chronic complication
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('2500','2501','2502','2503','2508','2509') 
            THEN 1 
            ELSE 0 END) AS diabetes_without_cc

        -- Diabetes with chronic complication
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('2504','2505','2506','2507')
            THEN 1 
            ELSE 0 END) AS diabetes_with_cc

        -- T1DM
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN (
                '25001','25003','25011','25013','25021','25023','25031',
                '25033','25041','25043','25051','25053','25061','25063','25071',
                '25073','25081','25083','25091','25093')
            THEN 1 
            ELSE 0 END) AS t1dm
         -- T2DM
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN (
            '25000','25002','25010','25012','25020','25022','25030','25032','25040',
            '25042','25050','25052','25060','25062','25070','25072','25080','25082',
            '25090','25092')
            THEN 1 
            ELSE 0 END) AS t2dm
        -- Hemiplegia or paraplegia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('342','343')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('3341','3440','3441','3442',
                                                  '3443','3444','3445','3446','3449')
            THEN 1 
            ELSE 0 END) AS paraplegia

        -- Renal disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('582','585','586','V56')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('5880','V420','V451')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '5830' AND '5837'
            OR
            SUBSTR(icd9_code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')          
            THEN 1 
            ELSE 0 END) AS renal_disease

        -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '140' AND '172'
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '1740' AND '1958'
            OR
            SUBSTR(icd9_code, 1, 3) BETWEEN '200' AND '208'
            OR
            SUBSTR(icd9_code, 1, 4) = '2386'
            THEN 1 
            ELSE 0 END) AS malignant_cancer

        -- Moderate or severe liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('4560','4561','4562')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '5722' AND '5728'
            THEN 1 
            ELSE 0 END) AS severe_liver_disease

        -- Metastatic solid tumor
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('196','197','198','199')
            THEN 1 
            ELSE 0 END) AS metastatic_solid_tumor

        -- AIDS/HIV
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('042','043','044')
            THEN 1 
            ELSE 0 END) AS aids

        -- SMOKING
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('V15')
            THEN 1 
            ELSE 0 END) AS smoking
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` ad
    GROUP BY ad.hadm_id
)
, echos AS
(
  select
    s.hadm_id,
    (select array_agg(struct(CHARTTIME, STORETIME, CATEGORY, TEXT) order by CHARTTIME) from unnest(note) 
        where ((note is not null) and (CATEGORY = 'Echo'))) echo,
  from (
    SELECT notes.hadm_id,
           array_agg(struct(notes.CHARTTIME, notes.STORETIME, notes.CATEGORY, notes.TEXT)) note,
           min(icu.intime) as intime
    from `physionet-data.mimiciii_notes.noteevents` notes
    LEFT JOIN `physionet-data.mimiciii_clinical.icustays` AS icu ON notes.HADM_ID = icu.hadm_id
    group by notes.hadm_id
  ) s
)
, bloods AS
(
    select
      s.hadm_id
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51256)) neutrophils
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51244, 51245)) lymphocytes
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51300, 51301, 51755)) wcc
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50889)) crp
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51222)) hb
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51265)) plt
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50862) and VALUENUM <= 10) albumin
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50912) and VALUENUM <= 150) creatinine
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50931) and VALUENUM <= 10000) glucose
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50882) and VALUENUM <= 10000) bicarb
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51006) and VALUENUM <= 300) bun
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50971) and VALUENUM <= 10000) potassium
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50960) and VALUENUM <= 10000) magnesium
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50861)) alt
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50863)) alp
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50878)) ast
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50927)) ggt
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50885)) bilirubin_total
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50883)) bilirubin_direct
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50884)) bilirubin_indirect
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (50813) and VALUENUM <= 10000) lactate
      , (select array_agg(struct(charttime, VALUENUM as value) order by charttime) from unnest(bloods) 
      where ITEMID in (51237)) inr
    from (
        SELECT le.hadm_id,
               array_agg(struct(le.ITEMID, le.charttime, le.VALUENUM)) bloods,
        FROM `physionet-data.mimiciii_clinical.labevents` le
        GROUP BY le.hadm_id
    ) s
)
, cardiac_index AS
(
    select
      s.icustay_id,
      (select array_agg(struct(charttime, ci) order by charttime) from unnest(ci) where ci is not null) ci,
    from (
      select 
        g.icustay_id,
        array_agg(struct(g.charttime, g.valuenum as ci)) ci
      from `physionet-data.mimiciii_clinical.chartevents` g
      where itemid in (228177, 226859, 228368, 226859, 116, 7610)
      group by g.icustay_id
    ) s
)
, prbcs AS
(
    select
      s.ICUSTAY_ID,
      (select array_agg(struct(CHARTTIME, prbc) order by CHARTTIME) from unnest(prbc) where prbc is not null) prbc,
    from (
      select 
        g.ICUSTAY_ID,
        array_agg(struct(g.CHARTTIME, g.AMOUNT as prbc)) prbc
      from `physionet-data.mimiciii_clinical.inputevents_cv` g
      LEFT JOIN `physionet-data.mimiciii_clinical.inputevents_mv`h ON g.ICUSTAY_ID = h.ICUSTAY_ID
      where (
          g.ITEMID in (226370, 227070, 226368, 225168, 221013, 44560, 43901, 43010, 30002, 30106, 30179, 30001, 30004, 42588, 42239, 46407, 42186) or
          h.ITEMID in (226370, 227070, 226368, 225168, 221013, 44560, 43901, 43010, 30002, 30106, 30179, 30001, 30004, 42588, 42239, 46407, 42186)
        )
      group by g.ICUSTAY_ID
    ) s
)
, aki AS
(
    select
      s.icustay_id,
      (select array_agg(struct(charttime, aki_stage_creat, aki_stage_uo) order by charttime) from unnest(aki) where aki is not null) aki,
    from (
      select 
        g.icustay_id,
        array_agg(struct(g.charttime, g.aki_stage_creat, g.aki_stage_uo)) aki
      from `physionet-data.mimiciii_derived.kdigo_stages` g
      group by g.icustay_id
    ) s
)
-- select features for final dataset
SELECT 
    ad.subject_id
    , ad.hadm_id
    , icu.ICUSTAY_ID as stay_id
    # , icu.DBSOURCE
    , ag.admission_age as age
    , pat.gender
    , icu2.ethnicity_grouped as ethnicity
    , body.height_first as height
    , body.weight_first as weight
    , ad.admission_type
    , ad.admission_location
    , ad.admittime
    , ad.dischtime
    , icu.INTIME
    , icu.OUTTIME
    , ad.insurance
    , icu2.icustay_seq
    , icu.los
    , icu.first_careunit
    , icu.last_careunit
    , icu.DBSOURCE
    , ad.hospital_expire_flag
    , surgery.cabg
    , surgery.aortic
    , surgery.mitral
    , surgery.tricuspid
    , surgery.pulmonary
    , ad.DEATHTIME as deathtime
    , pat.dod
    , com.myocardial_infarct
    , com.arrhythmia
    , com.congestive_heart_failure
    , com.peripheral_vascular_disease
    , com.cerebrovascular_disease
    , com.dementia
    , com.chronic_pulmonary_disease
    , com.rheumatic_disease
    , com.peptic_ulcer_disease
    , com.mild_liver_disease
    , com.diabetes_without_cc
    , com.diabetes_with_cc
    , com.t1dm
    , com.t2dm
    , com.paraplegia
    , com.renal_disease
    , com.malignant_cancer
    , com.severe_liver_disease 
    , com.metastatic_solid_tumor 
    , com.aids
    , com.smoking
    , bloods.neutrophils
    , bloods.lymphocytes
    , bloods.wcc
    , bloods.crp
    , bloods.hb
    , bloods.plt
    , bloods.albumin
    , bloods.creatinine
    , bloods.glucose
    , bloods.bicarb
    , bloods.bun
    , bloods.potassium
    , bloods.magnesium
    , bloods.alt
    , bloods.alp
    , bloods.ast
    , bloods.ggt
    , bloods.bilirubin_total
    , bloods.bilirubin_direct
    , bloods.bilirubin_indirect
    , bloods.lactate
    , bloods.inr
    , cardiac_index.ci as cardiac_index
    , prbcs.prbc
    , aki.aki
    # , pc.ICD_CODES
FROM `physionet-data.mimiciii_clinical.admissions` ad
LEFT JOIN surgery ON ad.hadm_id = surgery.hadm_id
LEFT JOIN ag ON ad.hadm_id = ag.hadm_id
RIGHT JOIN `physionet-data.mimiciii_clinical.icustays` AS icu ON ag.hadm_id = icu.HADM_ID
LEFT JOIN `physionet-data.mimiciii_derived.icustay_detail` AS icu2 ON icu.ICUSTAY_ID = icu2.icustay_id
LEFT JOIN `physionet-data.mimiciii_clinical.patients` AS pat ON icu.SUBJECT_ID = pat.subject_id
LEFT JOIN `physionet-data.mimiciii_derived.elixhauser_quan` AS elix ON icu.HADM_ID = elix.hadm_id
LEFT JOIN pc ON icu.hadm_id = pc.hadm_id
LEFT JOIN com ON icu.hadm_id = com.hadm_id
LEFT JOIN bloods ON icu.hadm_id = bloods.hadm_id
LEFT JOIN cardiac_index  ON icu.ICUSTAY_ID = cardiac_index.icustay_id
LEFT JOIN aki  ON icu.ICUSTAY_ID = aki.icustay_id
LEFT JOIN prbcs ON icu.ICUSTAY_ID = prbcs.ICUSTAY_ID
LEFT JOIN `physionet-data.mimiciii_derived.heightweight`AS body ON icu.icustay_id = body.icustay_id
# LEFT JOIN echos ON icu.hadm_id = echos.hadm_id
WHERE (
    surgery.cabg = 1 or surgery.aortic = 1 or surgery.mitral = 1 or surgery.tricuspid = 1 or surgery.pulmonary = 1
)
